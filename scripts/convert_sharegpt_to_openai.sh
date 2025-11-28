#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/common.sh"

usage() {
    echo "Usage: $0 <input_file> <output_file> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  input_file              Path to input dataset file"
    echo "  output_file             Path to save converted dataset file"
    echo ""
    echo "Optional Arguments:"
    echo "  --max-items <int>       Maximum number of items in the output file (default: 1000)"
    echo "  --min-turns <int>       Minimum number of turns per conversation (default: 2)"
    echo "  --max-turns <int>       Maximum number of turns per conversation (optional)"
    echo "  -f, --force             Force overwrite existing output file"
    echo ""
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/input /path/to/output"
    echo "  $0 ./data/sharegpt ./data/converted --max-items=5000 --max-turns=10"
    echo "  $0 ./data/sharegpt ./data/converted --force"
    echo ""
    exit 1
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -lt 2 ]; then
    usage
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
shift 2

CONVERT_SCRIPT="$SCRIPT_DIR/convert_sharegpt_to_openai.py"

MAX_ITEMS=1000
MIN_TURNS=2
MAX_TURNS=""
FORCE=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-items=*)
            MAX_ITEMS="${1#*=}"
            shift
            ;;
        --max-items)
            MAX_ITEMS="$2"
            shift 2
            ;;
        --min-turns=*)
            MIN_TURNS="${1#*=}"
            shift
            ;;
        --min-turns)
            MIN_TURNS="$2"
            shift 2
            ;;
        --max-turns=*)
            MAX_TURNS="${1#*=}"
            shift
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        *)
            warn "Unknown option: $1"
            shift
            ;;
    esac
done

# Check if output file exists
if [ -f "$OUTPUT_FILE" ]; then
    if [ "$FORCE" = true ]; then
        warn "Output file already exists, will be overwritten: $OUTPUT_FILE"
        rm -f "$OUTPUT_FILE"
    else
        info "Output file already exists: $OUTPUT_FILE"
        info "Use --force to overwrite"
        exit 0
    fi
fi

info "Conversion script: $CONVERT_SCRIPT"
info "Input dataset path: $INPUT_FILE"
info "Output dataset path: $OUTPUT_FILE"

# Install dependencies
if ! python3 -c "import pandas" &> /dev/null; then
    info "Installing pandas..."
    pip3 install pandas --quiet
fi

if ! python3 -c "import tqdm" &> /dev/null; then
    info "Installing tqdm..."
    pip3 install tqdm --quiet
fi

if ! python3 -c "import transformers" &> /dev/null; then
    info "Installing transformers..."
    pip3 install transformers --quiet
fi

# Build the command
CMD="python3 \"$CONVERT_SCRIPT\" \"$INPUT_FILE\" \"$OUTPUT_FILE\" --max-items=$MAX_ITEMS --min-turns=$MIN_TURNS"

if [ -n "$MAX_TURNS" ]; then
    CMD="$CMD --max-turns=$MAX_TURNS"
fi

info "Starting conversion..."
info "Command: $CMD"

# Execute the conversion
eval "$CMD"

if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    info "Conversion completed successfully!"
    info "Output file: $OUTPUT_FILE"
    info "File size: $FILE_SIZE"
else
    error "Conversion failed"
    exit 1
fi
