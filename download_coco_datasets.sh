#!/bin/bash

# Ensure script exits immediately if a command exits with a non-zero status
set -e

# === 1. Setup Directories ===

# Define directories
DATA_DIR="coco_data"
TRAIN_DIR="$DATA_DIR/train2017"
VAL_DIR="$DATA_DIR/val2017"
TEST_DIR="$DATA_DIR/test2017"
ANN_DIR="$DATA_DIR/annotations"

# Create directories if they don't exist
mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR" "$ANN_DIR"


# === 2. Download Images (Parallel) ===

echo "--- Starting Image Downloads (Backgrounded)... ---"

# -c : Continue getting a partially-downloaded file.
# -q : Quiet (Turn off output) to prevent progress bars from clashing on screen.
wget -c -q -P "$TRAIN_DIR" http://images.cocodataset.org/zips/train2017.zip &
wget -c -q -P "$VAL_DIR" http://images.cocodataset.org/zips/val2017.zip &
wget -c -q -P "$TEST_DIR" http://images.cocodataset.org/zips/test2017.zip &

# Wait ensures the script pauses here until ALL background downloads above finish.
wait 
echo "--- Image Downloads Complete. Starting Unzip... ---"


# === 3. Unzip Images ===

# -q : Quiet mode for unzip so file lists don't flood the terminal.
unzip -q "$TRAIN_DIR/train2017.zip" -d "$TRAIN_DIR" &
unzip -q "$VAL_DIR/val2017.zip" -d "$VAL_DIR" &
unzip -q "$TEST_DIR/test2017.zip" -d "$TEST_DIR" &

wait
echo "--- Image Unzip Complete. Cleaning up... ---"


# === 4. Cleanup (remove zip files) ===

rm "$TRAIN_DIR/train2017.zip" "$VAL_DIR/val2017.zip" "$TEST_DIR/test2017.zip"


# === 5. Download Annotations (Parallel) ===

echo "--- Starting Annotation Downloads ---"

wget -c -q -P "$ANN_DIR" http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
wget -c -q -P "$ANN_DIR" http://images.cocodataset.org/annotations/image_info_test2017.zip &

wait
echo "--- Annotation Downloads Complete. Starting Unzip... ---"


# === 6. Unzip Annotations ===

unzip -q "$ANN_DIR/annotations_trainval2017.zip" -d "$ANN_DIR" &
unzip -q "$ANN_DIR/image_info_test2017.zip" -d "$ANN_DIR" &

wait
echo "--- Annotation Unzip Complete. Cleaning up... ---"

# === 7. Cleanup (Remove zip files) ===

rm "$ANN_DIR/annotations_trainval2017.zip" "$ANN_DIR/image_info_test2017.zip"

echo "--- All tasks finished successfully. ---"