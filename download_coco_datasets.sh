#!/bin/bash

# Ensure script exits immediately if a command exits with a non-zero status
set -e

# Download function
download() {
    outdir="$1"
    url="$2"

    mkdir -p "$outdir"

    if command -v curl >/dev/null 2>&1; then
        echo "curl found, using curl to download."
        filename=$(basename "$url")
        curl -L -C - "$url" -o "$outdir/$filename" 
    elif command -v wget >/dev/null 2>&1; then
        echo "wget found, using wget to download."
        wget -c -P "$outdir" "$url" 
    else
        echo "Error: curl and wget are not installed. Please install one of them to proceed." >&2
        exit 1
    fi
}

# === 1. Setup Directories ===

# Define directories
DATA_DIR="coco_data"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"


# === 2. Download Images (Parallel) ===

echo "--- Starting Image Downloads (Backgrounded)... ---"

# -c : Continue getting a partially-downloaded file.
# -q : Quiet (Turn off output) to prevent progress bars from clashing on screen.
## if wget not found, use curl as an alternative
# use curl if wget is not available
download "$DATA_DIR" http://images.cocodataset.org/zips/train2017.zip &
download "$DATA_DIR" http://images.cocodataset.org/zips/val2017.zip &
download "$DATA_DIR" http://images.cocodataset.org/zips/val2014.zip & # We use val2014 as the test set

# Wait ensures the script pauses here until ALL background downloads above finish.
wait 
echo "--- Image Downloads Complete. Starting Unzip... ---"


# === 3. Unzip Images ===

# -q : Quiet mode for unzip so file lists don't flood the terminal.
unzip "$DATA_DIR/train2017.zip" -d "$DATA_DIR" &
unzip "$DATA_DIR/val2017.zip" -d "$DATA_DIR" &
unzip "$DATA_DIR/val2014.zip" -d "$DATA_DIR" &

wait
echo "--- Image Unzip Complete. Cleaning up... ---"


# === 4. Cleanup (remove zip files) ===

rm "$DATA_DIR/train2017.zip" "$DATA_DIR/val2017.zip" "$DATA_DIR/val2014.zip"


# === 5. Download Annotations (Parallel) ===

echo "--- Starting Annotation Downloads ---"

download "$DATA_DIR" http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
download "$DATA_DIR" http://images.cocodataset.org/annotations/annotations_trainval2014.zip &

wait
echo "--- Annotation Downloads Complete. Starting Unzip... ---"


# === 6. Unzip Annotations ===

unzip "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR" &
unzip "$DATA_DIR/annotations_trainval2014.zip" -d "$DATA_DIR" &

wait
echo "--- Annotation Unzip Complete. Cleaning up... ---"


# === 7. Cleanup (Remove zip files) ===

rm "$DATA_DIR/annotations_trainval2017.zip" "$DATA_DIR/annotations_trainval2014.zip"

echo "--- All tasks finished successfully. ---"