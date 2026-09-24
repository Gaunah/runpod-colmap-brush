#!/usr/bin/env python3
"""
roi_prep.py - restrict Brush training to an automatically detected region of interest.

Usage:  python3 roi_prep.py <dense_dir> [enu_model_dir]
        <dense_dir>     = image_undistorter output (contains images/ and sparse/)
        <enu_model_dir> = optional copy of dense/sparse aligned to GPS with
                          `colmap model_aligner --alignment_type enu`

Steps:
  1. Estimate the world 'up' axis. With a GPS-aligned model, 'up' is the ENU z
     axis mapped back into the model frame. Otherwise (or if the cameras are
     near-collinear, where GPS leaves the roll about the flight line open) it
     is inferred from camera orientations: drone gimbals hold roll ~0, so every
     camera's x-axis is horizontal; 'up' is the one direction perpendicular to
     all of them.
  2. ROI centre: the point the cameras look at (orbit / POI captures), or the
     median of well-observed sparse points (grids, facade passes).
  3. ROI = vertical cylinder. Radius from the camera spread around the centre,
     height from the sparse points inside it (ground -> tallest structure).
  4. Remove sparse points outside the cylinder (Brush's initialisation).
  5. Write one mask per image: a pixel is trainable only if its viewing ray
     passes through the ROI volume. Sky, horizon and far terrain are masked, so
     Brush neither fits them nor invents near-camera floaters to explain them.

Env overrides:
  ROI_SCALE    radius multiplier (default 1.1); larger keeps more context
  ROI_MASK_DS  downsample factor for mask computation (default 8)

Re-runnable: the original model is backed up once and always used as input,
so you can re-run with a different ROI_SCALE without redoing COLMAP.
"""
import os
import shutil
import sys

import numpy as np
import pycolmap
from PIL import Image


def cam_pose(img):
    """(R, t) of cam_from_world; handles pycolmap's property -> method change."""
    cfw = img.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation)


def gps_up(imgs, Rs, C_w, enu_dir):
    """World 'up' from a GPS/ENU-aligned copy of the model, or None if unusable.

    The alignment is a similarity transform, so a direction fixed in the world is
    the same vector in every camera frame in both models: the aligned camera sees
    ENU up as R_enu @ e_z, hence up in our frame is R^T @ R_enu @ e_z.
    """
    if not enu_dir or not os.path.isfile(os.path.join(enu_dir, "images.bin")):
        return None
    s = np.linalg.svd(C_w - C_w.mean(0), compute_uv=False)
    if s[1] < 0.05 * s[0]:
        print("WARNING: cameras are near-collinear; GPS alignment cannot fix the "
              "roll about the flight line, falling back to camera orientations.")
        return None
    enu = {i.name: i for i in pycolmap.Reconstruction(enu_dir).images.values()}
    ups = [R.T @ cam_pose(enu[img.name])[0][:, 2]
           for img, R in zip(imgs, Rs) if img.name in enu]
    if not ups:
        return None
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def estimate_up(Rs):
    rights, views = Rs[:, 0, :], Rs[:, 2, :]
    evals, evecs = np.linalg.eigh(rights.T @ rights)  # ascending
    if evals[1] > 0.01 * evals[2]:
        up = evecs[:, 0]
    else:
        # All cameras share one heading (straight pass / course lock): 'up' is only
        # known to lie in the plane perpendicular to it. Best guess: opposite of
        # the mean viewing direction, projected into that plane.
        print("WARNING: near-uniform camera heading; 'up' estimated from view "
              "direction and may be tilted by the gimbal pitch.")
        null = evecs[:, :2]
        up = null @ (null.T @ -views.mean(0))
        up /= np.linalg.norm(up)
    if up @ views.mean(0) > 0:  # drones look down, not up
        up = -up
    return up


def local_frame(up):
    """Rotation Q with local = world @ Q.T and local z = up."""
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])


def compute_roi(C, V, P, track_len, scale):
    """C camera centres, V viewing dirs, P sparse points - all in the local frame."""
    good = P[track_len >= 3]

    # Centre: least-squares point closest to all horizontal viewing lines.
    A, b = np.zeros((2, 2)), np.zeros(2)
    for c, d in zip(C[:, :2], V[:, :2]):
        w = np.linalg.norm(d)
        if w < 0.1:  # (near-)nadir camera: no horizontal direction
            continue
        d = d / w
        M = np.eye(2) - np.outer(d, d)
        A += w * M
        b += w * M @ c
    cam_mean = C[:, :2].mean(0)
    spread = np.linalg.norm(C[:, :2] - cam_mean, axis=1).max()
    centre, mode = None, None
    if np.trace(A) > 0 and np.linalg.cond(A) < 20:
        cand = np.linalg.solve(A, b)
        if np.linalg.norm(cand - cam_mean) < 2 * spread + 1e-9:
            centre, mode = cand, "look-at point (orbit/POI)"
    if centre is None:
        centre, mode = np.median(good[:, :2], axis=0), "sparse-point median (grid/pass)"

    radius = np.percentile(np.linalg.norm(C[:, :2] - centre, axis=1), 95)
    # Guard for captures with little horizontal spread (hover, tight spin).
    height = np.median(C[:, 2]) - np.percentile(good[:, 2], 5)
    radius = max(radius, height) * scale

    inside = np.linalg.norm(P[:, :2] - centre, axis=1) <= radius
    zs = P[inside & (track_len >= 3), 2]
    if len(zs) < 50:
        raise RuntimeError(f"only {len(zs)} well-observed points inside ROI")
    z_lo, z_hi = np.percentile(zs, [1, 99.5])
    pad = 0.1 * (z_hi - z_lo)
    return dict(centre=centre, radius=radius, z_bot=z_lo - pad, z_top=z_hi + pad,
                mode=mode)


def ray_hits_roi(o, d, roi):
    """True where ray o + t*d (t > 0) passes through the ROI cylinder volume."""
    p = o[:2] - roi["centre"]
    q = d[..., :2]
    qq, pq, pp, r2 = (q * q).sum(-1), (q * p).sum(-1), p @ p, roi["radius"] ** 2
    horiz = qq > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = pq ** 2 - qq * (pp - r2)
        s = np.sqrt(np.maximum(disc, 0))
        t0 = np.where(horiz, (-pq - s) / qq, -np.inf)
        t1 = np.where(horiz, (-pq + s) / qq, np.inf)
        hit_disk = np.where(horiz, disc >= 0, pp <= r2)

        dz = d[..., 2]
        vert = np.abs(dz) > 1e-12
        za, zb = (roi["z_bot"] - o[2]) / dz, (roi["z_top"] - o[2]) / dz
        s0 = np.where(vert, np.minimum(za, zb), -np.inf)
        s1 = np.where(vert, np.maximum(za, zb), np.inf)
        hit_slab = np.where(vert, True, roi["z_bot"] <= o[2] <= roi["z_top"])

    lo = np.maximum(np.maximum(t0, s0), 0.0)
    hi = np.minimum(t1, s1)
    return hit_disk & hit_slab & (hi > lo)


def main():
    dense = os.path.abspath(sys.argv[1])
    enu_dir = sys.argv[2] if len(sys.argv) > 2 else None
    scale = float(os.environ.get("ROI_SCALE", "1.1"))
    ds = int(os.environ.get("ROI_MASK_DS", "8"))
    model_dir = os.path.join(dense, "sparse")
    backup = os.path.join(os.path.dirname(dense), "dense_sparse_full")
    if not os.path.isdir(backup):
        shutil.copytree(model_dir, backup)
    rec = pycolmap.Reconstruction(backup)

    imgs = list(rec.images.values())
    poses = [cam_pose(i) for i in imgs]
    Rs = np.array([R for R, _ in poses])
    C_w = np.array([-R.T @ t for R, t in poses])
    pids = np.array(list(rec.points3D.keys()))
    P_w = np.array([rec.points3D[p].xyz for p in pids])
    tl = np.array([rec.points3D[p].track.length() for p in pids])

    up = gps_up(imgs, Rs, C_w, enu_dir)
    if up is not None:
        print("Up axis from GPS (ENU) alignment")
    else:
        print("Up axis estimated from camera orientations")
        up = estimate_up(Rs)
    Q = local_frame(up)
    C, V, P = C_w @ Q.T, Rs[:, 2, :] @ Q.T, P_w @ Q.T
    roi = compute_roi(C, V, P, tl, scale)
    print(f"ROI centre via {roi['mode']}")
    print(f"ROI radius {roi['radius']:.3f}, height {roi['z_bot']:.3f} .. {roi['z_top']:.3f} "
          f"(model units; cameras at median {np.median(C[:, 2]):.3f})")
    if roi["z_top"] >= C[:, 2].min():
        print("WARNING: some cameras fly below the tallest structure; masking is "
              "weaker for those views (they see everything around them).")

    # Masks and cropped model are both written to temp dirs and swapped in only
    # once everything succeeded, so a failure leaves dense/ untouched.
    tmp, final = os.path.join(dense, "masks.tmp"), os.path.join(dense, "masks")
    tmp_model = os.path.join(dense, "sparse.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp_model, ignore_errors=True)

    # --- masks ---
    fracs = {}
    for img, (R, _), c in zip(imgs, poses, C):
        cam = rec.cameras[img.camera_id]
        W, H = cam.width, cam.height
        uu, vv = np.meshgrid(np.arange(ds / 2, W, ds), np.arange(ds / 2, H, ds))
        pix = np.stack([uu, vv, np.ones_like(uu)], -1)
        rays = pix @ np.linalg.inv(cam.calibration_matrix()).T  # camera frame
        d = rays @ R @ Q.T  # row-vector form of R^T (world), then local frame
        keep = ray_hits_roi(c, d, roi)
        fracs[img.name] = keep.mean()
        m = Image.fromarray(keep.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        out = os.path.join(tmp, os.path.splitext(img.name)[0] + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        m.save(out)

    # --- crop sparse initialisation points ---
    r_xy = np.linalg.norm(P[:, :2] - roi["centre"], axis=1)
    keep_pts = ((r_xy <= roi["radius"]) & (P[:, 2] >= roi["z_bot"])
                & (P[:, 2] <= roi["z_top"]))
    for pid in pids[~keep_pts]:
        rec.delete_point3D(int(pid))
    os.makedirs(tmp_model)
    rec.write(tmp_model)

    # --- swap in (model_dir is recoverable from the backup) ---
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    shutil.rmtree(model_dir)
    os.replace(tmp_model, model_dir)

    f = np.array(list(fracs.values()))
    print(f"Points kept: {keep_pts.sum()}/{len(keep_pts)}")
    print(f"Trainable pixels per image: mean {f.mean():.0%}, min {f.min():.0%}")
    low = [n for n, v in fracs.items() if v < 0.02]
    if low:
        print(f"WARNING: {len(low)} images have <2% trainable pixels (looking away "
              f"from the ROI), e.g. {low[:3]}")
    print(f"Masks: {final}  (white = train, black = ignore)")


if __name__ == "__main__":
    main()
