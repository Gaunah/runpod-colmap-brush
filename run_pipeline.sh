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

    # Look for an 'images' folder anywhere inside the extracted tree
    IMAGES_DIR=$(find "$LOCAL_ROOT" -type d -name "images" | head -n 1)
    if [ -z "$IMAGES_DIR" ]; then
        echo "ERROR: no 'images' folder found inside the archive."
        exit 1
    fi
else
    # Local folder: copy to NVMe
    if [ ! -d "$INPUT" ]; then
        echo "ERROR: '$INPUT' is not a directory."
        exit 1
    fi
    echo "Copying: $INPUT -> $LOCAL_ROOT/images"
    cp -r "$INPUT" "$LOCAL_ROOT/images"
    IMAGES_DIR="$LOCAL_ROOT/images"
fi

# All work now lives on the NVMe
PROJECT_DIR=$(dirname "$IMAGES_DIR")
DATABASE="$PROJECT_DIR/database.db"
SPARSE="$PROJECT_DIR/sparse"
DENSE="$PROJECT_DIR/dense"
mkdir -p "$SPARSE" "$DENSE"

echo "Project dir: $PROJECT_DIR"
echo "Images dir:  $IMAGES_DIR"

echo "============================"
echo "COLMAP: Feature Extraction"
echo "============================"
colmap feature_extractor \
    --database_path "$DATABASE" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.camera_model "PINHOLE"

echo "============================"
echo "COLMAP: Matching"
echo "============================"
colmap exhaustive_matcher \
    --database_path "$DATABASE"

echo "============================"
echo "COLMAP: Sparse Mapping"
echo "============================"
colmap mapper \
    --database_path "$DATABASE" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE"

echo "============================"
echo "COLMAP: Dense Prep"
echo "============================"
colmap image_undistorter \
    --image_path "$IMAGES_DIR" \
    --input_path "$SPARSE/0" \
    --output_path "$DENSE" \
    --output_type COLMAP

echo "============================"
echo "Export PLY"
echo "============================"
colmap model_converter \
    --input_path "$SPARSE/0" \
    --output_path "$SPARSE/model.ply" \
    --output_type PLY

echo "============================"
echo "Start Brush Training"
echo "============================"
brush "$PROJECT_DIR" \
    --total-train-iters 30000 \
    --export-every 10000 \
    --export-path "$PROJECT_DIR"

echo "============================"
echo "Compressing PLY -> SOG"
echo "============================"

# Brush writes exports like export_10000.ply, export_20000.ply, export_30000.ply
# into $PROJECT_DIR. Grab the highest-iteration one.
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

    # Bundle the WebP+JSON output into a single .sog (zip) file for easy transfer
    (cd "$SOG_DIR" && zip -qr "$PROJECT_DIR/scene.sog" .)
    echo "SOG written: $PROJECT_DIR/scene.sog"
    echo "Original PLY: $(du -h "$LATEST_PLY" | cut -f1)"
    echo "Compressed:   $(du -h "$PROJECT_DIR/scene.sog" | cut -f1)"
fi

echo "Brush Gaussian Splat finished"
echo "Output: $PROJECT_DIR"
