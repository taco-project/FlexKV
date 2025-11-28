#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/common.sh"

usage() {
    echo "Usage: $0 [download_path] [filename]"
    echo ""
    echo "Arguments:"
    echo "  download_path    Path to save the dataset"
    echo "  filename         Dataset filename (default: ShareGPT_V3_unfiltered_cleaned_split.json)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/dataset                        # Download dataset with default filename"
    echo "  $0 /path/to/dataset custom_dataset.json   # Download dataset with custom filename"
    echo ""
    exit 1
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

if [ -z "$1" ]; then
    error "Download path not specified"
    usage
    exit 1
fi

SHAREGPT_DATASET_PATH="$1"

if [ -z "$2" ]; then
    DATASET_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
else
    DATASET_FILE="$2"
fi

DATASET_PATH="$SHAREGPT_DATASET_PATH/$DATASET_FILE"

if [ -f "$DATASET_PATH" ]; then
    info "Dataset file already exists: $DATASET_PATH"
    info "Skipping download"
    exit 0
fi

if [ ! -d "$SHAREGPT_DATASET_PATH" ]; then
    info "Creating directory: $SHAREGPT_DATASET_PATH"
    mkdir -p "$SHAREGPT_DATASET_PATH"
fi

DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

info "Starting ShareGPT dataset download..."
info "Download URL: $DATASET_URL"

if command -v wget &> /dev/null; then
    info "Downloading with wget..."
    wget --progress=bar:force:noscroll \
         --timeout=30 \
         --tries=3 \
         -O "$DATASET_PATH" \
         "$DATASET_URL"
elif command -v curl &> /dev/null; then
    info "Downloading with curl..."
    curl -L \
         --progress-bar \
         --connect-timeout 30 \
         --retry 3 \
         -o "$DATASET_PATH" \
         "$DATASET_URL"
else
    error "Neither wget nor curl found, please install one of them"
    exit 1
fi
