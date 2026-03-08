#!/bin/bash
# Monitors tokenization progress. When the current run finishes,
# checks for newly downloaded directories and re-runs the tokenizer.

RAW_DIR="/home/shaoliang/dolma3_mix-6T/data"
OUT_DIR="/home/shaoliang/dolma3_tokenized"
TOKENIZED_DIR="$OUT_DIR/preprocessed/dolma3-0625/v0.1-official/allenai/dolma2-tokenizer"
LOG="$OUT_DIR/tokenize_all.log"
TOKENIZE_SCRIPT="/home/shaoliang/OLMo-core/scripts/data/tokenize_data.sh"

echo "[$(date)] watch_and_tokenize.sh started" | tee -a "$LOG"

while true; do
    # Check if tokenizer is running
    if pgrep -f "dolma tokens" > /dev/null 2>&1; then
        TOKENIZED=$(ls "$TOKENIZED_DIR" 2>/dev/null | wc -l)
        DOWNLOADED=$(ls "$RAW_DIR" 2>/dev/null | wc -l)
        echo "[$(date)] Tokenizer running. Tokenized: $TOKENIZED / Downloaded: $DOWNLOADED dirs" | tee -a "$LOG"
        sleep 3600
        continue
    fi

    # Tokenizer is not running — check for new directories
    TOKENIZED=$(ls "$TOKENIZED_DIR" 2>/dev/null | sort)
    DOWNLOADED=$(ls "$RAW_DIR" 2>/dev/null | sort)
    NEW_DIRS=$(comm -23 <(echo "$DOWNLOADED") <(echo "$TOKENIZED"))

    if [ -z "$NEW_DIRS" ]; then
        echo "[$(date)] Tokenizer idle. No new directories to tokenize. Checking again in 1h..." | tee -a "$LOG"
        sleep 3600
        continue
    fi

    NEW_COUNT=$(echo "$NEW_DIRS" | wc -l)
    echo "[$(date)] Tokenizer idle. Found $NEW_COUNT new directories. Starting tokenizer..." | tee -a "$LOG"
    echo "$NEW_DIRS" | head -5 | while read d; do echo "  - $d"; done | tee -a "$LOG"

    bash "$TOKENIZE_SCRIPT" --processes 128 >> "$LOG" 2>&1

    echo "[$(date)] Tokenizer run finished." | tee -a "$LOG"
done
