#!/bin/bash
#
# Download the Dolma3 mix-6T raw data from Hugging Face.
#
# Uses huggingface_hub to download the allenai/dolma3_mix-6T dataset
# (raw JSONL.zst files) that are then tokenized by tokenize_data.sh.
#
# Usage:
#   bash download_data.sh [--out-dir DIR] [--pattern GLOB] [--hf-token TOKEN] [--dry-run]
#
# Examples:
#   bash download_data.sh                              # download all 163 dirs
#   bash download_data.sh --pattern "common_crawl-*"   # only common crawl
#   bash download_data.sh --out-dir /data/dolma3       # custom output dir
#   bash download_data.sh --dry-run                    # preview without downloading
#
# Requires:
#   pip install huggingface_hub hf_transfer
#   Optional: export HF_TOKEN=hf_... for private datasets

set -euo pipefail

REPO_ID="allenai/dolma3_mix-6T"
OUT_DIR="/home/shaoliang/dolma3_mix-6T"
PATTERN="*"
HF_TOKEN="${HF_TOKEN:-}"
DRY_RUN=false
NUM_PROC=16  # parallel download connections

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        --pattern)   PATTERN="$2"; shift 2 ;;
        --hf-token)  HF_TOKEN="$2"; shift 2 ;;
        --num-proc)  NUM_PROC="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Dolma3 Data Downloader ==="
echo "Repo:        $REPO_ID"
echo "Output:      $OUT_DIR"
echo "Pattern:     data/$PATTERN"
echo "Num procs:   $NUM_PROC"
echo ""

mkdir -p "$OUT_DIR"

if $DRY_RUN; then
    echo "[DRY RUN] Would download: $REPO_ID (data/$PATTERN) -> $OUT_DIR"
    echo "[DRY RUN] Command: huggingface-cli download $REPO_ID --repo-type dataset --local-dir $OUT_DIR --include \"data/$PATTERN\""
    exit 0
fi

# Enable fast downloads via hf_transfer if available
export HF_HUB_ENABLE_HF_TRANSFER=1

TOKEN_ARG=""
if [[ -n "$HF_TOKEN" ]]; then
    TOKEN_ARG="--token $HF_TOKEN"
fi

echo "Starting download..."
echo "(Resume-safe: already-downloaded files are skipped automatically)"
echo ""

huggingface-cli download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$OUT_DIR" \
    --include "data/$PATTERN" \
    $TOKEN_ARG

echo ""
echo "=== Download Complete ==="
echo "Output dir:  $OUT_DIR"
echo "Dir count:   $(ls "$OUT_DIR/data" 2>/dev/null | wc -l) directories"
echo "Disk usage:  $(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
