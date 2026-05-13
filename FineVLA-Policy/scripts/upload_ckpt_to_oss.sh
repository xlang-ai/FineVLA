#!/usr/bin/env bash
# Upload checkpoint folder (or selected checkpoints) to OSS via ossutil.
#
# Usage:
#   ./upload_ckpt_to_oss.sh <run_dir> [step1 step2 step3 ...]
#
# Arguments:
#   <run_dir>   Run directory name under CKPT_ROOT, e.g.
#               xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT
#   [steps]     Optional list of step numbers. If given, only those checkpoints
#               are uploaded. Otherwise ALL steps_*_pytorch_model.pt are uploaded.
#
# Examples:
#   # Upload specific checkpoints:
#   ./upload_ckpt_to_oss.sh xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT 40000 80000 100000
#
#   # Upload all checkpoints in the run:
#   ./upload_ckpt_to_oss.sh xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT
#
# Behavior:
#   - Uploads config.yaml and dataset_statistics.json from the run dir first
#   - Uploads each checkpoint sequentially (prints progress)
#   - Skips files that already exist on OSS (checked via ossutil stat)
#   - Uses parallel multipart upload (16 threads, 256MB parts) for speed
#
# Environment overrides:
#   OSS_BUCKET   default bucket        (default: ofasys-vla-shanghai)
#   OSS_PREFIX   default key prefix    (default: xintonghu/StarVLA)
#   CKPT_ROOT    local checkpoint root (default: <repo>/results/Checkpoints)

set -euo pipefail
unset http_proxy https_proxy all_proxy no_proxy

# ---------- defaults ----------
DEFAULT_BUCKET="ofasys-vla-shanghai"
DEFAULT_PREFIX="xintonghu/StarVLA"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CKPT_ROOT="${REPO_ROOT}/results/Checkpoints"

CKPT_ROOT="${CKPT_ROOT:-${DEFAULT_CKPT_ROOT}}"
BUCKET="${OSS_BUCKET:-${DEFAULT_BUCKET}}"
PREFIX="${OSS_PREFIX:-${DEFAULT_PREFIX}}"
PREFIX="${PREFIX%/}"

LOCAL_BASE_VLM="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B"
OSS_BASE_VLM="/cpfs01/xintonghu/StarVLA/_pretrained/Qwen3.5-4B"

# ---------- args ----------
RUN_DIR_NAME="${1:-}"
shift || true
STEP_LIST=("$@")

if [[ -z "${RUN_DIR_NAME}" ]]; then
    echo "Usage: $0 <run_dir> [step1 step2 step3 ...]" >&2
    echo "" >&2
    echo "  run_dir : folder name under ${CKPT_ROOT}/" >&2
    echo "  steps   : optional step numbers (e.g. 40000 80000 100000)" >&2
    echo "" >&2
    echo "  default bucket : ${DEFAULT_BUCKET}" >&2
    echo "  default prefix : ${DEFAULT_PREFIX}" >&2
    exit 1
fi

if ! command -v ossutil >/dev/null 2>&1; then
    echo "Error: ossutil not found in PATH." >&2
    exit 1
fi

# ---------- resolve run directory ----------
RUN_DIR="${CKPT_ROOT}/${RUN_DIR_NAME}"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Error: run directory not found: ${RUN_DIR}" >&2
    exit 1
fi

CKPT_DIR="${RUN_DIR}/checkpoints"
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "Error: checkpoints directory not found: ${CKPT_DIR}" >&2
    exit 1
fi

# ---------- helper: check if file exists on OSS ----------
oss_exists() {
    local dest="$1"
    ossutil stat "${dest}" >/dev/null 2>&1
}

# ---------- helper: upload a single file ----------
upload_file() {
    local src="$1"
    local rel="$2"
    local dest="oss://${BUCKET}/${PREFIX}/${rel}"

    if oss_exists "${dest}"; then
        echo "  SKIP (already exists): ${dest}"
        return 0
    fi

    local size
    size="$(du -h "${src}" | cut -f1)"
    echo "  Uploading: ${src} (${size}) → ${dest}"
    ossutil cp -f \
        --parallel 16 \
        --part-size 256M \
        "${src}" "${dest}"
    echo "  DONE: ${dest}"
}

# ---------- helper: upload config.yaml with upload-only path rewrite ----------
upload_config_yaml() {
    local src="$1"
    local rel="$2"
    local tmp
    tmp="$(mktemp /tmp/upload_config.XXXXXX.yaml)"
    cp "${src}" "${tmp}"
    sed -i "s#${LOCAL_BASE_VLM}#${OSS_BASE_VLM}#g" "${tmp}"
    upload_file "${tmp}" "${rel}"
    rm -f "${tmp}"
}

echo "==============================================="
echo "Run dir   : ${RUN_DIR}"
echo "OSS dest  : oss://${BUCKET}/${PREFIX}/${RUN_DIR_NAME}/"
echo "==============================================="
echo ""

# ---------- step 1: upload sidecar metadata ----------
echo ">>> Uploading metadata files..."
for fname in config.yaml dataset_statistics.json; do
    if [[ -f "${RUN_DIR}/${fname}" ]]; then
        if [[ "${fname}" == "config.yaml" ]]; then
            upload_config_yaml "${RUN_DIR}/${fname}" "${RUN_DIR_NAME}/${fname}"
        else
            upload_file "${RUN_DIR}/${fname}" "${RUN_DIR_NAME}/${fname}"
        fi
    elif [[ -f "${CKPT_DIR}/${fname}" ]]; then
        if [[ "${fname}" == "config.yaml" ]]; then
            upload_config_yaml "${CKPT_DIR}/${fname}" "${RUN_DIR_NAME}/checkpoints/${fname}"
        else
            upload_file "${CKPT_DIR}/${fname}" "${RUN_DIR_NAME}/checkpoints/${fname}"
        fi
    else
        echo "  Warning: ${fname} not found, skipping."
    fi
done
echo ""

# ---------- step 2: build checkpoint list ----------
CKPT_FILES=()
if [[ ${#STEP_LIST[@]} -gt 0 ]]; then
    # Use specified steps
    for step in "${STEP_LIST[@]}"; do
        pt_file="${CKPT_DIR}/steps_${step}_pytorch_model.pt"
        if [[ -f "${pt_file}" ]]; then
            CKPT_FILES+=("${pt_file}")
        else
            echo "Warning: checkpoint not found, skipping: ${pt_file}" >&2
        fi
    done
else
    # Upload all checkpoints, sorted by step number
    while IFS= read -r f; do
        CKPT_FILES+=("${f}")
    done < <(ls -1 "${CKPT_DIR}"/steps_*_pytorch_model.pt 2>/dev/null | sort -t_ -k2 -n)
fi

if [[ ${#CKPT_FILES[@]} -eq 0 ]]; then
    echo "No checkpoint files to upload." >&2
    exit 1
fi

echo ">>> Uploading ${#CKPT_FILES[@]} checkpoint(s)..."
echo ""

# ---------- step 3: upload checkpoints ----------
uploaded=0
skipped=0
for ckpt in "${CKPT_FILES[@]}"; do
    basename_ckpt="$(basename "${ckpt}")"
    rel="${RUN_DIR_NAME}/checkpoints/${basename_ckpt}"
    dest="oss://${BUCKET}/${PREFIX}/${rel}"

    if oss_exists "${dest}"; then
        echo "  [$(( uploaded + skipped + 1 ))/${#CKPT_FILES[@]}] SKIP (already exists): ${basename_ckpt}"
        skipped=$(( skipped + 1 ))
    else
        size="$(du -h "${ckpt}" | cut -f1)"
        echo "  [$(( uploaded + skipped + 1 ))/${#CKPT_FILES[@]}] Uploading: ${basename_ckpt} (${size})"
        ossutil cp -f \
            --parallel 16 \
            --part-size 256M \
            "${ckpt}" "${dest}"
        echo "  DONE: ${basename_ckpt}"
        uploaded=$(( uploaded + 1 ))
    fi
done

echo ""
echo "==============================================="
echo "Summary: ${uploaded} uploaded, ${skipped} skipped (already on OSS)"
echo "OSS path: oss://${BUCKET}/${PREFIX}/${RUN_DIR_NAME}/checkpoints/"
echo "==============================================="
