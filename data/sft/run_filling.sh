#!/bin/bash
# Script to run filling.py with all template files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITEMS_FILE="items.csv"
OUTPUT_DIR="filled"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Template files to process
TEMPLATES=(
    "L1_Direct_template.csv"
    "L3_Sycophancy_template.csv"
    "L4_Quality_Trap_template.csv"
    "L5_Adversarial_template.csv"
)

# Check if items file exists
if [ ! -f "$ITEMS_FILE" ]; then
    echo "Error: Items file not found: $ITEMS_FILE"
    exit 1
fi

# Process each template
for template in "${TEMPLATES[@]}"; do
    if [ ! -f "$template" ]; then
        echo "Warning: Template file not found: $template, skipping..."
        continue
    fi
    
    # Extract base name (e.g., "L1_Direct" from "L1_Direct_template.csv")
    base_name=$(basename "$template" "_template.csv")
    
    # Create output filename
    output_file="$OUTPUT_DIR/${base_name}_filling.csv"
    
    echo "Processing $template -> $output_file"
    python3 filling.py --items "$ITEMS_FILE" --templates "$template" --output "$output_file"
done

echo ""
echo "All templates processed successfully!"
echo "Output files are in: $OUTPUT_DIR/"
