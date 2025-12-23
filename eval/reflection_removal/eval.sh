#!/bin/bash

# ==========================================
# Configuration Area
# ==========================================

# 1. Path to the Python evaluation script
PYTHON_SCRIPT="eval.py"

# 2. Dataset path configuration
GT_DIR="data/SIR2/PostcardDataset/transmission_layer"  # Ground Truth folder
OUTPUT_DIR="results/full-2/output/postcard/results"    # Model inference results
SAVE_CSV="results/full-2/indicator/postcard.csv"       # Path to save the result CSV

# 3. Optional: Path for resized images (Leave as "" to disable saving)
RESIZE_DIR=""

# ==========================================
# Execution Logic
# ==========================================

echo "Starting metric calculation..."
echo "------------------------------------"
echo "GT Path:     $GT_DIR"
echo "Output Path: $OUTPUT_DIR"
echo "Save CSV:    $SAVE_CSV"

# Ensure the directory for the CSV file exists
CSV_DIR=$(dirname "$SAVE_CSV")
mkdir -p "$CSV_DIR"

# Construct the command
CMD="python \"$PYTHON_SCRIPT\" -g \"$GT_DIR\" -o \"$OUTPUT_DIR\" -s \"$SAVE_CSV\""

# Append resize directory argument if configured
if [ -n "$RESIZE_DIR" ]; then
    mkdir -p "$RESIZE_DIR"
    CMD="$CMD -r \"$RESIZE_DIR\""
fi

# Execute the command
eval $CMD

# Check execution status
if [ $? -eq 0 ]; then
    echo "------------------------------------"
    echo "✅ Calculation completed! Results saved to: $SAVE_CSV"
else
    echo "------------------------------------"
    echo "❌ Calculation failed. Please check the Python error messages above."
fi