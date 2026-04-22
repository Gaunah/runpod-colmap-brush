#!/bin/bash

# ============================
# Check for image folder
# ============================
IMAGES_DIR=$1
if [ -z "$IMAGES_DIR" ]; then
    echo "Please provide the path to your images folder."
    echo "Usage: bash run_pipeline.sh /workspace/images"
    exit 1
fi

# Set project folders relative to images
PROJECT_DIR=$(dirname "$IMAGES_DIR")
DATABASE="$PROJECT_DIR/database.db"
SPARSE="$PROJECT_DIR/sparse"
DENSE="$PROJECT_DIR/dense"

mkdir -p "$SPARSE"
mkdir -p "$DENSE"

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
 --database_path "$DATABASE" \

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
 --output_type COLMAP \
 --max_image_size 4000

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
 --total-steps 30000 \
 --max-resolution 4000 \
 --export-every 10000 \
 --export-path "$PROJECT_DIR"

echo "Brush Gaussian Splat finished"
