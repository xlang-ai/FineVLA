#!/usr/bin/env bash
# Download checkpoint folder (or selected checkpoints) from OSS via ossutil.
#
# This script also bootstraps ~/.ossutilconfig the first time it is run on a
# new cluster (or you can run it explicitly with `config`).
#
# ---------- Usage ----------
#
# 1) One-time configure ossutil (writes ~/.ossutilconfig):
#      ./download_ckpt_from_oss.sh config
#    Credentials are read from env vars (or fall back to interactive prompt):
#      OSS_ACCESS_KEY_ID
#      OSS_ACCESS_KEY_SECRET
#      OSS_ENDPOINT   (default: oss-cn-shanghai.aliyuncs.com)
#      OSS_REGION     (default: cn-shanghai)
#
# 2) Download checkpoints:
#      ./download_ckpt_from_oss.sh <run_dir> [step1 step2 step3 ...] [--dest /path/to/local]
#
#    <run_dir>  Run directory name on OSS, e.g.
#               xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT
#    [steps]    Optional step numbers. If given, only those checkpoints are
#               downloaded. Otherwise ALL steps_*_pytorch_model.pt are downloaded.
#    [--dest]   Optional local destination root (default: /cpfs01/xintonghu/StarVLA)
#
# Behavior:
#   - Downloads config.yaml and dataset_statistics.json from the run dir first
#   - Downloads each checkpoint sequentially (prints progress)
#   - Skips files that already exist locally with the same size
#   - Uses parallel multipart download (16 threads, 256MB parts) for speed
#
# ---------- Examples ----------
#   ./scripts/download_ckpt_from_oss.sh config
#   ./scripts/download_ckpt_from_oss.sh xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT 40000 80000
#   ./scripts/download_ckpt_from_oss.sh xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT --dest /data/ckpts
#   ./scripts/download_ckpt_from_oss.sh xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT 40000 80000 --dest /data/ckpts
#
# ---------- Environment overrides ----------
#   OSS_BUCKET   default bucket          (default: ofasys-vla-shanghai)
#   OSS_PREFIX   default key prefix      (default: xintonghu/StarVLA)
#   CKPT_ROOT    local destination root  (default: /cpfs01/xintonghu/StarVLA)

set -euo pipefail
unset http_proxy https_proxy all_proxy no_proxy

# ---------- defaults ----------
DEFAULT_BUCKET="ofasys-vla-shanghai"
DEFAULT_PREFIX="xintonghu/StarVLA"
DEFAULT_ENDPOINT="oss-cn-shanghai.aliyuncs.com"
DEFAULT_REGION="cn-shanghai"

DEFAULT_CKPT_ROOT="/cpfs01/xintonghu/StarVLA"

BUCKET="${OSS_BUCKET:-${DEFAULT_BUCKET}}"
PREFIX="${OSS_PREFIX:-${DEFAULT_PREFIX}}"
PREFIX="${PREFIX%/}"
CKPT_ROOT="${CKPT_ROOT:-${DEFAULT_CKPT_ROOT}}"

OSSUTIL_CONFIG="${HOME}/.ossutilconfig"

# ---------- helpers ----------
die() { echo "Error: $*" >&2; exit 1; }

require_ossutil() {
    command -v ossutil >/dev/null 2>&1 || die "ossutil not found in PATH."
}

write_ossutil_config() {
    local ak sk endpoint region
    ak="${OSS_ACCESS_KEY_ID:-}"
    sk="${OSS_ACCESS_KEY_SECRET:-}"
    endpoint="${OSS_ENDPOINT:-${DEFAULT_ENDPOINT}}"
    region="${OSS_REGION:-${DEFAULT_REGION}}"

    if [[ -z "${ak}" ]]; then
        read -r -p "OSS Access Key ID: " ak
    fi
    if [[ -z "${sk}" ]]; then
        read -r -s -p "OSS Access Key Secret: " sk
        echo
    fi

    [[ -n "${ak}" && -n "${sk}" ]] || die "Access Key ID/Secret are required."

    if [[ -f "${OSSUTIL_CONFIG}" ]]; then
        cp "${OSSUTIL_CONFIG}" "${OSSUTIL_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
        echo "Backed up existing config to ${OSSUTIL_CONFIG}.bak.*"
    fi

    umask 077
    cat > "${OSSUTIL_CONFIG}" <<EOF
[default]
accessKeyId=${ak}
accessKeySecret=${sk}
endpoint=${endpoint}
region=${region}
EOF
    chmod 600 "${OSSUTIL_CONFIG}"
    echo "Wrote ${OSSUTIL_CONFIG}"
    echo "  endpoint=${endpoint}"
    echo "  region=${region}"
}

# ---------- subcommand: config ----------
if [[ "${1:-}" == "config" ]]; then
    require_ossutil
    write_ossutil_config
    exit 0
fi

# ---------- parse args ----------
RUN_DIR_NAME=""
STEP_LIST=()
LOCAL_DEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            LOCAL_DEST="$2"
            shift 2
            ;;
        *)
            if [[ -z "${RUN_DIR_NAME}" ]]; then
                RUN_DIR_NAME="$1"
            else
                STEP_LIST+=("$1")
            fi
            shift
            ;;
    esac
done

LOCAL_DEST="${LOCAL_DEST:-${CKPT_ROOT}}"

if [[ -z "${RUN_DIR_NAME}" ]]; then
    cat >&2 <<EOF
Usage:
  $0 config
  $0 <run_dir> [step1 step2 ...] [--dest /path/to/local]

Defaults:
  bucket    : ${BUCKET}
  prefix    : ${PREFIX}
  ckpt root : ${CKPT_ROOT}

Examples:
  $0 xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT 40000 80000
  $0 xintonghu_qwen3.5_OFT_RawRDT10W_RoboTwin_SFT --dest /data/ckpts
EOF
    exit 1
fi

require_ossutil

if [[ ! -f "${OSSUTIL_CONFIG}" ]]; then
    echo "ossutil config not found at ${OSSUTIL_CONFIG}."
    echo "Run '$0 config' first (or set OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET and rerun)."
    exit 1
fi

# ---------- helper: download a single file ----------
download_file() {
    local rel="$1"
    local src="oss://${BUCKET}/${PREFIX}/${rel}"
    local dest="${LOCAL_DEST%/}/${rel}"
    local dest_dir
    dest_dir="$(dirname "${dest}")"
    mkdir -p "${dest_dir}"

    # Skip if local file exists and has same size as OSS
    if [[ -f "${dest}" ]]; then
        local local_size oss_size
        local_size="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
        oss_size="$(ossutil stat "${src}" 2>/dev/null | grep -i 'Content-Length' | awk '{print $NF}' || echo -1)"
        if [[ "${local_size}" == "${oss_size}" && "${local_size}" != "0" ]]; then
            echo "  SKIP (already exists, same size): $(basename "${dest}")"
            return 0
        fi
    fi

    # Check if file exists on OSS
    if ! ossutil stat "${src}" >/dev/null 2>&1; then
        echo "  Warning: not found on OSS, skipping: ${src}"
        return 0
    fi

    echo "  Downloading: ${src}"
    echo "         → ${dest}"
    ossutil cp -f \
        --parallel 16 \
        --part-size 256M \
        "${src}" "${dest}"
    echo "  DONE: $(basename "${dest}") ($(du -h "${dest}" | cut -f1))"
}

echo "==============================================="
echo "Run dir    : ${RUN_DIR_NAME}"
echo "OSS source : oss://${BUCKET}/${PREFIX}/${RUN_DIR_NAME}/"
echo "Local dest : ${LOCAL_DEST%/}/${RUN_DIR_NAME}/"
echo "==============================================="
echo ""

# ---------- step 1: download sidecar metadata ----------
echo ">>> Downloading metadata files..."
download_file "${RUN_DIR_NAME}/config.yaml"
download_file "${RUN_DIR_NAME}/dataset_statistics.json"
echo ""

# ---------- step 2: build checkpoint list ----------
CKPT_FILES=()
if [[ ${#STEP_LIST[@]} -gt 0 ]]; then
    # Use specified steps
    for step in "${STEP_LIST[@]}"; do
        CKPT_FILES+=("${RUN_DIR_NAME}/checkpoints/steps_${step}_pytorch_model.pt")
    done
else
    # List all checkpoints on OSS and download all
    echo ">>> Listing checkpoints on OSS..."
    while IFS= read -r line; do
        # ossutil ls output lines look like: oss://bucket/prefix/path
        fname="$(echo "${line}" | grep -oP 'steps_\d+_pytorch_model\.pt' || true)"
        if [[ -n "${fname}" ]]; then
            CKPT_FILES+=("${RUN_DIR_NAME}/checkpoints/${fname}")
        fi
    done < <(ossutil ls "oss://${BUCKET}/${PREFIX}/${RUN_DIR_NAME}/checkpoints/" 2>/dev/null)

    # Sort by step number
    IFS=$'\n' CKPT_FILES=($(printf '%s\n' "${CKPT_FILES[@]}" | sort -t_ -k2 -n)); unset IFS
fi

if [[ ${#CKPT_FILES[@]} -eq 0 ]]; then
    echo "No checkpoint files to download." >&2
    exit 1
fi

echo ">>> Downloading ${#CKPT_FILES[@]} checkpoint(s)..."
echo ""

# ---------- step 3: download checkpoints ----------
downloaded=0
skipped=0
for ((i=0; i<${#CKPT_FILES[@]}; i++)); do
    rel="${CKPT_FILES[$i]}"
    basename_ckpt="$(basename "${rel}")"
    dest="${LOCAL_DEST%/}/${rel}"

    # Skip check (same size)
    src="oss://${BUCKET}/${PREFIX}/${rel}"
    if [[ -f "${dest}" ]]; then
        local_size="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
        oss_size="$(ossutil stat "${src}" 2>/dev/null | grep -i 'Content-Length' | awk '{print $NF}' || echo -1)"
        if [[ "${local_size}" == "${oss_size}" && "${local_size}" != "0" ]]; then
            echo "  [$((i+1))/${#CKPT_FILES[@]}] SKIP (already exists): ${basename_ckpt}"
            skipped=$(( skipped + 1 ))
            continue
        fi
    fi

    echo "  [$((i+1))/${#CKPT_FILES[@]}] Downloading: ${basename_ckpt}"
    mkdir -p "$(dirname "${dest}")"
    ossutil cp -f \
        --parallel 16 \
        --part-size 256M \
        "${src}" "${dest}"
    echo "  DONE: ${basename_ckpt} ($(du -h "${dest}" | cut -f1))"
    downloaded=$(( downloaded + 1 ))
done

echo ""
echo "==============================================="
echo "Summary: ${downloaded} downloaded, ${skipped} skipped (already local)"
echo "Local path: ${LOCAL_DEST%/}/${RUN_DIR_NAME}/"
echo "==============================================="
