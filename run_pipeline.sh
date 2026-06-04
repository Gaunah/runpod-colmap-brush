#!/bin/bash
set -euo pipefail

# check for GPU
nvidia-smi -L >/dev/null 2>&1 || { echo "ERROR: no GPU visible; Brush needs CUDA."; exit 1; }

# ============================
# Parse input: local folder OR zip URL
# ============================
INPUT=${1:-}
if [ -z "$INPUT" ]; then
    echo "Usage: bash run_pipeline.sh <images_folder | zip_url>"
    echo "  Local:  bash run_pipeline.sh /workspace/images"
    echo "  Remote: bash run_pipeline.sh https://example.com/dataset.zip"
    exit 1
fi

VOCAB_TREE="/app/vocab_tree_flickr100K_words32K.bin"

# ============================
# Stage everything on fast local NVMe up front (skip if already staged)
# ============================
LOCAL_ROOT="/local_temp/dataset"
mkdir -p "$LOCAL_ROOT"

# Locate the staged image directory: the first top-level folder under
# $LOCAL_ROOT whose subtree contains image files. Takes the first folder from
# the zip that holds images. Recurses, so nested layouts (wrapper/sub/*.jpg)
# work: COLMAP reads --image_path recursively, so we hand it the top folder
# as-is. Skips: sparse/ + dense/ (created by later stages on a re-run), macOS
# "__MACOSX" wrapper dirs, and AppleDouble "._*" sidecar files (junk that
# carries a real image extension but isn't a readable image).
find_image_dir() {
    local d
    while IFS= read -r d; do
        if find "$d" -type f ! -name '._*' \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
               -o -iname '*.tif' -o -iname '*.tiff' \) -print -quit | grep -q .; then
            printf '%s\n' "$d"
            return 0
        fi
    done < <(find "$LOCAL_ROOT" -mindepth 1 -maxdepth 1 -type d \
                  ! -name sparse ! -name dense ! -name __MACOSX | sort)
    return 1
}

echo "============================"
echo "Staging Dataset on Local NVMe"
echo "============================"

# If a previous run already staged the images, reuse them.
EXISTING_IMAGES=$(find_image_dir || true)
if [ -n "$EXISTING_IMAGES" ]; then
    echo "Images already staged at: $EXISTING_IMAGES"
    echo "Skipping download/copy. (Delete $LOCAL_ROOT to force a fresh stage.)"
    IMAGES_DIR="$EXISTING_IMAGES"
elif [[ "$INPUT" =~ ^https?:// ]]; then
    # Remote zip: download + unzip directly on NVMe
    echo "Downloading: $INPUT"
    pushd "$LOCAL_ROOT" > /dev/null
    aria2c \
        --continue=true \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        "$INPUT"
    ZIP_FILE=$(ls -t ./*.zip 2>/dev/null | head -n 1 || true)
    if [ -z "$ZIP_FILE" ]; then
        echo "ERROR: no .zip file was downloaded."
        exit 1
    fi
    echo "Unzipping: $ZIP_FILE"
    7z x -y -bso0 -bsp0 "$ZIP_FILE"
    rm "$ZIP_FILE"
    popd > /dev/null

    IMAGES_DIR=$(find_image_dir || true)
    if [ -z "$IMAGES_DIR" ]; then
        echo "ERROR: no folder containing images found inside the archive."
        exit 1
    fi
else
    if [ ! -d "$INPUT" ]; then
        echo "ERROR: '$INPUT' is not a directory."
        exit 1
    fi
    echo "Copying: $INPUT -> $LOCAL_ROOT/images"
    cp -r "$INPUT" "$LOCAL_ROOT/images"
    IMAGES_DIR="$LOCAL_ROOT/images"
fi

PROJECT_DIR=$(dirname "$IMAGES_DIR")
DATABASE="$PROJECT_DIR/database.db"
SPARSE="$PROJECT_DIR/sparse"
DENSE="$PROJECT_DIR/dense"
mkdir -p "$SPARSE" "$DENSE"

NUM_IMAGES=$(find "$IMAGES_DIR" -type f ! -name '._*' \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)

echo "Project dir: $PROJECT_DIR"
echo "Images dir:  $IMAGES_DIR ($NUM_IMAGES images)"

echo "============================"
echo "COLMAP: Feature Extraction (OPENCV, single camera)"
echo "============================"
# OPENCV model handles DJI lens distortion; single_camera shares intrinsics
# across the whole set since all photos came from the same lens.
# affine_shape + domain_size_pooling produce stronger, more repeatable features.
#
# Retry loop: start at nproc/1 (full) and step the divisor up to nproc/$MAX_DIVISOR on failure
MAX_DIVISOR=10
for DIVISOR in $(seq 1 $MAX_DIVISOR); do
    THREADS=$(( $(nproc) / DIVISOR ))
    [ "$THREADS" -lt 1 ] && THREADS=1
    echo "Attempt $DIVISOR/$MAX_DIVISOR: nproc/$DIVISOR = $THREADS threads"
    if colmap feature_extractor \
        --database_path "$DATABASE" \
        --image_path "$IMAGES_DIR" \
        --ImageReader.camera_model "OPENCV" \
        --ImageReader.single_camera 1 \
        --FeatureExtraction.num_threads "$THREADS" \
        --SiftExtraction.estimate_affine_shape 1 \
        --SiftExtraction.domain_size_pooling 1
    then
        echo "Feature extraction succeeded."
        break
    fi
    echo "feature_extractor failed."
    if [ "$DIVISOR" -eq "$MAX_DIVISOR" ]; then
        echo "ERROR: feature extraction failed after $MAX_DIVISOR attempts."
        exit 1
    fi
    echo "Retrying with fewer threads..."
    sleep 2
done

# ============================
# COLMAP: Matching (auto-pick strategy)
# ============================
# Prefer spatial matching when GPS is in EXIF (DJI always has it). Fall back
# to vocab tree for larger sets, exhaustive only for very small ones.
HAS_POSE_PRIORS_TABLE=$(sqlite3 "$DATABASE" \
    "SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors';" \
    2>/dev/null || echo "")

if [ -n "$HAS_POSE_PRIORS_TABLE" ]; then
    # coordinate_system: 0 = WGS84 (GPS), 1 = Cartesian, -1 = undefined
    HAS_GPS=$(sqlite3 "$DATABASE" \
        "SELECT COUNT(*) FROM pose_priors WHERE position IS NOT NULL AND coordinate_system = 0;" \
        2>/dev/null || echo 0)
else
    HAS_GPS=$(sqlite3 "$DATABASE" \
        "SELECT COUNT(*) FROM images WHERE prior_tx IS NOT NULL AND (prior_tx != 0 OR prior_ty != 0 OR prior_tz != 0);" \
        2>/dev/null || echo 0)
fi

echo "============================"
if [ "$HAS_GPS" -gt 0 ]; then
    echo "COLMAP: Spatial Matching ($HAS_GPS images with GPS priors)"
    echo "============================"
    colmap spatial_matcher \
        --database_path "$DATABASE" \
        --SpatialMatching.max_num_neighbors 50 \
        --SpatialMatching.max_distance 100
elif [ "$NUM_IMAGES" -gt 150 ]; then
    echo "COLMAP: Vocab Tree Matching"
    echo "============================"
    colmap vocab_tree_matcher \
        --database_path "$DATABASE" \
        --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE"
else
    echo "COLMAP: Exhaustive Matching"
    echo "============================"
    colmap exhaustive_matcher \
        --database_path "$DATABASE"
fi

echo "============================"
echo "COLMAP: Sparse Mapping (initial)"
echo "============================"
colmap mapper \
    --database_path "$DATABASE" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE" \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.ba_use_gpu 1 \
    --Mapper.ba_gpu_index 0

# ============================
# Check for split reconstruction, attempt to bridge if found
# ============================
NUM_MODELS=$(find "$SPARSE" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ "$NUM_MODELS" -gt 1 ] && [ "$HAS_GPS" -gt 0 ]; then
    echo "============================"
    echo "WARNING: $NUM_MODELS sub-models after spatial matching."
    echo "Running supplemental vocab tree matching to find missed connections."
    echo "============================"
    colmap vocab_tree_matcher \
        --database_path "$DATABASE" \
        --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE"

    echo "============================"
    echo "COLMAP: Sparse Mapping (re-attempt with augmented matches)"
    echo "============================"
    # Mapper writes to a *separate* directory so we can compare and pick the
    # better result. Don't clobber the first attempt — it may still be the
    # better one if vocab tree introduces bad matches.
    SPARSE_V2="$PROJECT_DIR/sparse_v2"
    mkdir -p "$SPARSE_V2"
    colmap mapper \
        --database_path "$DATABASE" \
        --image_path "$IMAGES_DIR" \
        --output_path "$SPARSE_V2" \
        --Mapper.ba_refine_principal_point 1 \
        --Mapper.ba_use_gpu 1 \
        --Mapper.ba_gpu_index 0

    # Compare: prefer the run with the largest single model
    LARGEST_V1=$(find "$SPARSE"    -mindepth 1 -maxdepth 1 -type d -exec wc -c {}/images.bin \; 2>/dev/null | sort -rn | head -n1 | awk '{print $1}')
    LARGEST_V2=$(find "$SPARSE_V2" -mindepth 1 -maxdepth 1 -type d -exec wc -c {}/images.bin \; 2>/dev/null | sort -rn | head -n1 | awk '{print $1}')
    LARGEST_V1=${LARGEST_V1:-0}
    LARGEST_V2=${LARGEST_V2:-0}

    echo "Largest model size — v1: $LARGEST_V1 bytes, v2: $LARGEST_V2 bytes"
    if [ "$LARGEST_V2" -gt "$LARGEST_V1" ]; then
        echo "Second pass produced a larger model. Using sparse_v2."
        SPARSE="$SPARSE_V2"
    else
        echo "First pass model is still largest. Keeping sparse."
    fi
fi

# ============================
# Pick largest sub-model
# ============================
# COLMAP may split into sparse/0, sparse/1, ... if coverage is weak.
# Pick the one with the most registered images instead of blindly using /0.
LARGEST_MODEL=$(find "$SPARSE" -mindepth 1 -maxdepth 1 -type d \
    -exec sh -c 'echo "$(wc -c < "$1/images.bin" 2>/dev/null || echo 0) $1"' _ {} \; \
    | sort -rn | head -n1 | awk '{print $2}')

if [ -z "$LARGEST_MODEL" ] || [ ! -f "$LARGEST_MODEL/images.bin" ]; then
    echo "ERROR: COLMAP produced no usable sparse model."
    exit 1
fi

echo "Largest sparse model: $LARGEST_MODEL"

# Sanity check: surface a warning if reconstruction looks weak.
NUM_MODELS=$(find "$SPARSE" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$NUM_MODELS" -gt 1 ]; then
    echo "WARNING: COLMAP produced $NUM_MODELS disconnected sub-models."
    echo "         Capture probably has coverage gaps. Splat will only cover the largest one."
fi

echo "============================"
echo "COLMAP: Model Analysis"
echo "============================"
colmap model_analyzer --path "$LARGEST_MODEL" || true

echo "============================"
echo "COLMAP: Image Undistortion"
echo "============================"
# Brush expects pinhole images. Undistort produces $DENSE/images (rectified)
# and $DENSE/sparse (rectified model). We train Brush on $DENSE, NOT on the
# original distorted images.
colmap image_undistorter \
    --image_path "$IMAGES_DIR" \
    --input_path "$LARGEST_MODEL" \
    --output_path "$DENSE" \
    --output_type COLMAP

echo "============================"
echo "Start Brush Training (on undistorted dense dir)"
echo "============================"
# Pointing Brush at $DENSE, not $PROJECT_DIR, so it trains on rectified images
# matched to the rectified pinhole model.
brush "$DENSE" \
    --total-train-iters 60000 \
    --export-every 20000 \
    --max-splats 5000000 \
    --export-path "$PROJECT_DIR"

echo "Brush Gaussian Splat finished"

echo "============================"
echo "Compressing PLY -> SOG (splat-transform)"
echo "============================"
LATEST_PLY=$(ls -t "$PROJECT_DIR"/export_*.ply 2>/dev/null | head -n 1 || true)

if [ -z "$LATEST_PLY" ]; then
    echo "WARNING: no Brush export_*.ply found in $PROJECT_DIR, skipping SOG."
else
    SOG_OUT="$PROJECT_DIR/scene.sog"
    echo "Compressing: $LATEST_PLY -> $SOG_OUT"
    splat-transform "$LATEST_PLY" "$SOG_OUT"
    echo "Original PLY: $(du -h "$LATEST_PLY" | cut -f1)"
    echo "Compressed:   $(du -h "$SOG_OUT" | cut -f1)"
fi

echo "============================"
echo "Pipeline complete."
echo "Outputs in: $PROJECT_DIR"
echo "  - sparse model:     $LARGEST_MODEL"
echo "  - undistorted:      $DENSE"
echo "  - splat PLY:        ${LATEST_PLY:-<missing>}"
echo "  - compressed SOG:   $PROJECT_DIR/scene.sog"
echo "============================"

# Persist deliverables off ephemeral NVMe before the pod can vanish
PERSIST_DIR="/workspace/outputs/$(basename "$PROJECT_DIR")_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PERSIST_DIR"
cp -v "$PROJECT_DIR/scene.sog" "$PERSIST_DIR/" 2>/dev/null || true
[ -n "${LATEST_PLY:-}" ] && cp -v "$LATEST_PLY" "$PERSIST_DIR/" || true
cp -rv "$LARGEST_MODEL" "$PERSIST_DIR/sparse" || true
