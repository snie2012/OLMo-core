#!/bin/bash
#
# Tokenize downloaded Dolma3 raw text (JSONL) into .npy files for OLMo training.
#
# Uses the `dolma tokens` CLI with the allenai/dolma2-tokenizer to match
# the OLMo-3 7B training script (TokenizerConfig.dolma2()).
#
# Usage:
#   bash tokenize_data.sh [--processes N] [--dry-run] [--source PATTERN]
#
# Examples:
#   bash tokenize_data.sh                           # tokenize all sources
#   bash tokenize_data.sh --processes 32            # use 32 parallel workers
#   bash tokenize_data.sh --source "common_crawl-science*"  # only science
#   bash tokenize_data.sh --dry-run                 # preview without running
#
# After tokenization, launch training with:
#   torchrun --nproc-per-node=8 \
#     src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py \
#     --save-folder=/path/to/checkpoints \
#     --data-root=/home/shaoliang/dolma3_tokenized

set -euo pipefail

RAW_DIR="/home/shaoliang/dolma3_mix-6T/data"
OUT_DIR="/home/shaoliang/dolma3_tokenized"
TOKENIZER="allenai/dolma2-tokenizer"
EOS_TOKEN_ID=100257
PAD_TOKEN_ID=100277
DTYPE="uint32"
PROCESSES=16
DRY_RUN=false
SOURCE_PATTERN="*"
CONDA_ENV="dolma"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --processes) PROCESSES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --source) SOURCE_PATTERN="$2"; shift 2 ;;
        --raw-dir) RAW_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== OLMo-3 Data Tokenizer ==="
echo "Raw data:    $RAW_DIR"
echo "Output:      $OUT_DIR"
echo "Tokenizer:   $TOKENIZER"
echo "EOS token:   $EOS_TOKEN_ID"
echo "PAD token:   $PAD_TOKEN_ID"
echo "dtype:       $DTYPE"
echo "Processes:   $PROCESSES"
echo "Source:       $SOURCE_PATTERN"
echo ""

SOURCES=$(ls -d "$RAW_DIR"/$SOURCE_PATTERN 2>/dev/null || true)
if [[ -z "$SOURCES" ]]; then
    echo "ERROR: No source directories found matching '$RAW_DIR/$SOURCE_PATTERN'"
    exit 1
fi

TOTAL=$(echo "$SOURCES" | wc -l)
echo "Found $TOTAL source directories to tokenize."
echo ""

DONE_DIR="$OUT_DIR/.done"
mkdir -p "$DONE_DIR"

IDX=0
SKIPPED=0
for SRC_DIR in $SOURCES; do
    SRC_NAME=$(basename "$SRC_DIR")
    IDX=$((IDX + 1))
    MARKER="$DONE_DIR/$SRC_NAME"

    if [[ -f "$MARKER" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    FILE_COUNT=$(find "$SRC_DIR" -name '*.jsonl.zst' 2>/dev/null | wc -l)
    if [[ "$FILE_COUNT" -eq 0 ]]; then
        echo "[$IDX/$TOTAL] SKIP $SRC_NAME (no .jsonl.zst files yet)"
        continue
    fi

    DEST="$OUT_DIR/preprocessed/dolma3-0625/v0.1-official/allenai/dolma2-tokenizer/$SRC_NAME"
    mkdir -p "$DEST"

    echo "[$IDX/$TOTAL] Tokenizing $SRC_NAME ($FILE_COUNT files) -> $DEST"

    if $DRY_RUN; then
        echo "  [DRY RUN] dolma tokens --documents '$SRC_DIR/*.jsonl.zst' --destination '$DEST' ..."
        continue
    fi

    /opt/miniconda3/bin/conda run -n "$CONDA_ENV" \
        dolma tokens \
            --documents "$SRC_DIR/*.jsonl.zst" \
            --destination "$DEST" \
            --tokenizer.name_or_path "$TOKENIZER" \
            --tokenizer.eos_token_id "$EOS_TOKEN_ID" \
            --tokenizer.pad_token_id "$PAD_TOKEN_ID" \
            --dtype "$DTYPE" \
            --processes "$PROCESSES" \
        2>&1 | tee -a "$OUT_DIR/tokenize_${SRC_NAME}.log"

    touch "$MARKER"
    echo "[$IDX/$TOTAL] DONE $SRC_NAME"
    echo ""
done

echo "=== Tokenization Summary ==="
echo "Total sources:   $TOTAL"
echo "Skipped (done):  $SKIPPED"
echo "Output dir:      $OUT_DIR"
