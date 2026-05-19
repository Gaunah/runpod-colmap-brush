# runpod-colmap-brush

Docker image and pipeline for turning drone photos into 3D Gaussian Splats on RunPod. Built around COLMAP for Structure-from-Motion and [Brush](https://github.com/ArthurBrussee/brush) for splat training, with PlayCanvas SOGS compression at the end.

Tuned for DJI drone footage (Mini 5 Pro and friends), but works with any photo set that has reasonable overlap.

## What's inside

- **COLMAP** (built from source, CUDA-enabled) for feature extraction, matching, and sparse reconstruction
- **Brush** for Gaussian Splat training (Rust, GPU-accelerated)
- **SOGS** for compressing the trained splat into a compact `.sog` (typically 10–20× smaller than the source PLY)
- **JupyterLab** for interactive access on RunPod
- A `run_pipeline.sh` script that runs the whole flow end-to-end

## Quick start (RunPod)

1. Deploy a RunPod GPU pod using this image: `gaunah/runpod-colmap-brush:latest`
2. Set the `JUPYTER_PASSWORD` environment variable to whatever token you want
3. Open JupyterLab on port 8888
4. From a terminal, run the pipeline against either a local folder or a zip URL:

```bash
# Local folder
bash /app/run_pipeline.sh /workspace/my_dataset/images

# Or a remote zip (expects an 'images/' folder inside)
bash /app/run_pipeline.sh https://example.com/dataset.zip
```

Everything is staged onto `/local_temp/dataset` for the run. Final outputs land in that project directory:

- `sparse_points.ply` — COLMAP sparse point cloud (for sanity checking)
- `export_60000.ply` — trained Gaussian Splat
- `scene.sog` — compressed splat, ready for web viewers

## Pipeline overview

1. **Stage data** — download/unzip or copy to local NVMe
2. **Feature extraction** — COLMAP with the OPENCV camera model and shared intrinsics (`single_camera 1`)
3. **Matching** — auto-selects strategy based on input:
   - Spatial matching when EXIF GPS is present (DJI default)
   - Vocab tree matching for sets > 150 images without GPS
   - Exhaustive matching for small sets
4. **Sparse mapping** — COLMAP mapper with principal-point refinement
5. **Undistortion** — produces pinhole-rectified images for Brush
6. **Splat training** — Brush, 60k iterations, exports every 20k
7. **Compression** — SOGS turns the final PLY into a `.sog` bundle

## Building the image

Local build:

```bash
docker build --build-arg COLMAP_REF=3.11.1 -t runpod-colmap-brush:dev .
```

## Capture tips for best results

The pipeline is only as good as your photos. For drone work:

- Lock exposure manually before takeoff (auto-exposure causes flicker in splats)
- Shoot in 4:3 to use the full sensor
- Capture at two altitudes / two pitch angles, not just a single orbit
- Aim for 60–80% overlap between consecutive frames
- Overcast light beats harsh sun
- Avoid pure top-down — splats need parallax

## Notes

- Brush trains on the **undistorted** dataset (`$DENSE`), not the original distorted images — important if you fork the pipeline
- If COLMAP produces multiple disconnected sub-models, the pipeline picks the largest and warns; this signals a capture coverage gap
- The image bundles `vocab_tree_flickr100K_words32K.bin` at `/app/` for vocab tree matching

## License

This is a glue layer around open-source projects — check the licenses of [COLMAP](https://github.com/colmap/colmap), [Brush](https://github.com/ArthurBrussee/brush), and [SOGS](https://github.com/playcanvas/sogs) for their respective terms.
