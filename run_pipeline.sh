#!/bin/bash
set -euo pipefail

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
# Stage everything on fast local NVMe up front
# ============================
LOCAL_ROOT="/local_temp/dataset"
mkdir -p "$LOCAL_ROOT"

echo "============================"
echo "Staging Dataset on Local NVMe"
echo "============================"

if [[ "$INPUT" =~ ^https?:// ]]; then
    # Remote zip: download + unzip directly on NVMe
    echo "Downloading: $INPUT"
    pushd "$LOCAL_ROOT" > /dev/null
    curl -LJO "$INPUT"
    ZIP_FILE=$(ls -t ./*.zip 2>/dev/null | head -n 1 || true)
    if [ -z "$ZIP_FILE" ]; then
        echo "ERROR: no .zip file was downloaded."
        exit 1
    fi
    echo "Unzipping: $ZIP_FILE"
    unzip -q "$ZIP_FILE"
    rm "$ZIP_FILE"
    popd > /dev/null

    IMAGES_DIR=$(find "$LOCAL_ROOT" -type d -name "images" | head -n 1)
    if [ -z "$IMAGES_DIR" ]; then
        echo "ERROR: no 'images' folder found inside the archive."
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

NUM_IMAGES=$(find "$IMAGES_DIR" -maxdepth 1 -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)

echo "Project dir: $PROJECT_DIR"
echo "Images dir:  $IMAGES_DIR ($NUM_IMAGES images)"

echo "============================"
echo "COLMAP: Feature Extraction (OPENCV, single camera)"
echo "============================"
# OPENCV model handles DJI lens distortion; single_camera shares intrinsics
# across the whole set since all photos came from the same lens.
# affine_shape + domain_size_pooling produce stronger, more repeatable features.
colmap feature_extractor \
    --database_path "$DATABASE" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.camera_model "OPENCV" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

# ============================
# COLMAP: Matching (auto-pick strategy)
# ============================
# Prefer spatial matching when GPS is in EXIF (DJI always has it). Fall back
# to vocab tree for larger sets, exhaustive only for very small ones.
HAS_GPS=$(sqlite3 "$DATABASE" \
    "SELECT COUNT(*) FROM images WHERE prior_tx != 0 OR prior_ty != 0 OR prior_tz != 0;" \
    2>/dev/null || echo 0)

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
echo "COLMAP: Sparse Mapping"
echo "============================"
# Refining principal point matters for DJI sensors; tighter BA tolerance
# yields cleaner poses (slower, worth it).
colmap mapper \
    --database_path "$DATABASE" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE" \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.ba_global_function_tolerance 1e-6

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
echo "Export sparse PLY"
echo "============================"
colmap model_converter \
    --input_path "$LARGEST_MODEL" \
    --output_path "$PROJECT_DIR/sparse_points.ply" \
    --output_type PLY

echo "============================"
echo "Start Brush Training (on undistorted dense dir)"
echo "============================"
# Pointing Brush at $DENSE, not $PROJECT_DIR, so it trains on rectified images
# matched to the rectified pinhole model.
brush "$DENSE" \
    --total-train-iters 60000 \
    --export-every 20000 \
    --export-path "$PROJECT_DIR"

echo "Brush Gaussian Splat finished"

echo "============================"
echo "Compressing PLY -> SOG"
echo "============================"
LATEST_PLY=$(ls -t "$PROJECT_DIR"/export_*.ply 2>/dev/null | head -n 1)

if [ -z "$LATEST_PLY" ]; then
    echo "WARNING: no Brush export_*.ply found in $PROJECT_DIR, skipping SOG."
else
    SOG_DIR="$PROJECT_DIR/sog"
    mkdir -p "$SOG_DIR"
    echo "Compressing: $LATEST_PLY"
    sogs-compress \
        --ply "$LATEST_PLY" \
        --output-dir "$SOG_DIR"

    (cd "$SOG_DIR" && zip -qr "$PROJECT_DIR/scene.sog" .)
    echo "SOG written: $PROJECT_DIR/scene.sog"
    echo "Original PLY: $(du -h "$LATEST_PLY" | cut -f1)"
    echo "Compressed:   $(du -h "$PROJECT_DIR/scene.sog" | cut -f1)"
fi

echo "============================"
echo "Pipeline complete."
echo "Outputs in: $PROJECT_DIR"
echo "  - sparse model:     $LARGEST_MODEL"
echo "  - undistorted:      $DENSE"
echo "  - sparse PLY:       $PROJECT_DIR/sparse_points.ply"
echo "  - splat PLY:        ${LATEST_PLY:-<missing>}"
echo "  - compressed SOG:   $PROJECT_DIR/scene.sog"
echo "============================"
