#!/bin/bash
# run_convert.sh - Convert LeRobot v3.0 datasets to v2.1 format
#
# This script:
# 1. Sets up the environment
# 2. Runs batch conversion from v3.0 to v2.1
# 3. Verifies the converted datasets
#
# Usage:
#   ./run_convert.sh [--dry-run] [--single RELATIVE_PATH]
#
# Examples:
#   # Dry run - list all v3.0 datasets
#   ./run_convert.sh --dry-run
#
#   # Convert all datasets
#   ./run_convert.sh
#
#   # Convert a single dataset
#   ./run_convert.sh --single benchmark1_0_compressed/agilex_3rgb/1_potatooven

set -e  # Exit on error

#############################################
# CONFIGURATION - Modify these paths as needed
#############################################

# Input directory containing v3.0 datasets
INPUT_ROOT="/cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/WQY_VLA_DATA/RoboMIND_lerobot/RoboMIND_lerobot"

# Output directory for v2.1 datasets (will be created)
OUTPUT_ROOT="/cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/WQY_VLA_DATA/RoboMIND_lerobot_v21_converted"

# Conda environment name
CONDA_ENV="any4lerobot"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Number of episodes to sample for verification
VERIFY_SAMPLE_SIZE=3

#############################################
# FUNCTIONS
#############################################

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_error "ffmpeg is not installed. Please install it first."
        log_error "  On CentOS/RHEL: sudo yum install ffmpeg"
        log_error "  On Ubuntu: sudo apt install ffmpeg"
        log_error "  Via conda: conda install -c conda-forge ffmpeg"
        exit 1
    fi
    
    # Check ffprobe
    if ! command -v ffprobe &> /dev/null; then
        log_error "ffprobe is not installed (usually comes with ffmpeg)."
        exit 1
    fi
    
    # Check Python dependencies
    python -c "import pyarrow; import jsonlines; import tqdm; import numpy" 2>/dev/null || {
        log_error "Missing Python dependencies. Install with:"
        log_error "  pip install pyarrow jsonlines tqdm numpy datasets"
        exit 1
    }
    
    log_info "All dependencies OK"
}

activate_conda() {
    log_info "Activating conda environment: $CONDA_ENV"
    
    # Try different conda initialization methods
    if [ -f "/cpfs04/shared/Group-m6/tongzai.hxt/miniforge3/etc/profile.d/conda.sh" ]; then
        source "/cpfs04/shared/Group-m6/tongzai.hxt/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        log_error "Cannot find conda. Please ensure conda is installed and in PATH."
        exit 1
    fi
    
    conda activate "$CONDA_ENV" || {
        log_error "Failed to activate conda environment: $CONDA_ENV"
        log_error "Create it with: conda create -n $CONDA_ENV python=3.10"
        exit 1
    }
    
    log_info "Conda environment activated: $(which python)"
}

run_dry_run() {
    log_info "DRY RUN: Listing v3.0 datasets..."
    python "$SCRIPT_DIR/batch_convert_v30_to_v21.py" \
        --input-root "$INPUT_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --dry-run
}

run_single_conversion() {
    local rel_path="$1"
    log_info "Converting single dataset: $rel_path"
    
    python "$SCRIPT_DIR/batch_convert_v30_to_v21.py" \
        --input-root "$INPUT_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --single "$rel_path"
    
    # Verify the converted dataset
    log_info "Verifying converted dataset..."
    python "$SCRIPT_DIR/verify_v21_dataset.py" \
        --root "$OUTPUT_ROOT/$rel_path" \
        --sample-size "$VERIFY_SAMPLE_SIZE" \
        --verbose
}

run_batch_conversion() {
    log_info "Starting batch conversion..."
    log_info "  Input:  $INPUT_ROOT"
    log_info "  Output: $OUTPUT_ROOT"
    
    # Create output directory
    mkdir -p "$OUTPUT_ROOT"
    
    # Run conversion
    python "$SCRIPT_DIR/batch_convert_v30_to_v21.py" \
        --input-root "$INPUT_ROOT" \
        --output-root "$OUTPUT_ROOT"
    
    # Verify all converted datasets
    log_info "Verifying all converted datasets..."
    python "$SCRIPT_DIR/verify_v21_dataset.py" \
        --root "$OUTPUT_ROOT" \
        --sample-size "$VERIFY_SAMPLE_SIZE" \
        --batch
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Convert LeRobot v3.0 datasets to v2.1 format"
    echo ""
    echo "Options:"
    echo "  --dry-run              List datasets without converting"
    echo "  --single RELATIVE_PATH Convert only a single dataset"
    echo "  --input-root PATH      Override input root directory"
    echo "  --output-root PATH     Override output root directory"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run"
    echo "  $0 --single benchmark1_0_compressed/agilex_3rgb/1_potatooven"
    echo "  $0"
}

#############################################
# MAIN
#############################################

# Parse arguments
DRY_RUN=false
SINGLE_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --single)
            SINGLE_PATH="$2"
            shift 2
            ;;
        --input-root)
            INPUT_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Change to script directory
cd "$SCRIPT_DIR"

# Activate conda environment
activate_conda

# Check dependencies
check_dependencies

# Run appropriate mode
if [ "$DRY_RUN" = true ]; then
    run_dry_run
elif [ -n "$SINGLE_PATH" ]; then
    run_single_conversion "$SINGLE_PATH"
else
    run_batch_conversion
fi

log_info "Done!"
