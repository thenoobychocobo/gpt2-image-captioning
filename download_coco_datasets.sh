#!/bin/bash

# Ensure script exits immediately if a command exits with a non-zero status
set -e

# === 1. Setup Directories ===

# Define directories
DATA_DIR="coco_data"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"


# === 2. Download Images (Parallel) ===

echo "--- Starting Image Downloads (Backgrounded)... ---"

# -c : Continue getting a partially-downloaded file.
# -q : Quiet (Turn off output) to prevent progress bars from clashing on screen.
wget -c -q -P "$DATA_DIR" http://images.cocodataset.org/zips/train2017.zip &
wget -c -q -P "$DATA_DIR" http://images.cocodataset.org/zips/val2017.zip &
wget -c -q -P "$DATA_DIR" http://images.cocodataset.org/zips/test2017.zip &

# Wait ensures the script pauses here until ALL background downloads above finish.
wait 
echo "--- Image Downloads Complete. Starting Unzip... ---"


# === 3. Unzip Images ===

# -q : Quiet mode for unzip so file lists don't flood the terminal.
unzip -q "$DATA_DIR/train2017.zip" -d "$DATA_DIR" &
unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR" &
unzip -q "$DATA_DIR/test2017.zip" -d "$DATA_DIR" &

wait
echo "--- Image Unzip Complete. Cleaning up... ---"


# === 4. Cleanup (remove zip files) ===

rm "$DATA_DIR/train2017.zip" "$DATA_DIR/val2017.zip" "$DATA_DIR/test2017.zip"


# === 5. Download Annotations (Parallel) ===

echo "--- Starting Annotation Downloads ---"

wget -c -q -P "$DATA_DIR" http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
wget -c -q -P "$DATA_DIR" http://images.cocodataset.org/annotations/image_info_test2017.zip &

wait
echo "--- Annotation Downloads Complete. Starting Unzip... ---"


# === 6. Unzip Annotations ===

unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR" &
unzip -q "$DATA_DIR/image_info_test2017.zip" -d "$DATA_DIR" &

wait
echo "--- Annotation Unzip Complete. Cleaning up... ---"


# === 7. Cleanup (Remove zip files) ===

rm "$DATA_DIR/annotations_trainval2017.zip" "$DATA_DIR/image_info_test2017.zip"

echo "--- All tasks finished successfully. ---"