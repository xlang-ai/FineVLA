#!/bin/bash
# Doubao caption hard mode - auto retry with video->image fallback
# Usage: nohup bash run_doubao_hard.sh > /tmp/doubao_hard_auto.log 2>&1 &

cd /mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-Bench/caption_eval/annotate

RESULT_FILE="results/hard/doubao_doubao-seed-2-0-pro-260215_CaptionResult.jsonl"
TARGET=500
MAX_RETRIES=10

get_success_count() {
    if [ ! -f "$RESULT_FILE" ]; then
        echo 0
    else
        python3 -c "
import json
count = 0
with open('$RESULT_FILE') as f:
    for line in f:
        d = json.loads(line)
        if d.get('call_success'):
            count += 1
print(count)
"
    fi
}

echo "$(date '+%H:%M:%S') [START] Doubao hard mode auto-retry script"

# Pass 1: Video URL mode, 8 workers
echo "$(date '+%H:%M:%S') [PASS1] Running video URL mode with 8 workers..."
python3 run_annotate.py \
    --model "doubao.doubao-seed-2-0-pro-260215" \
    --input-type video \
    --fps 4 \
    --output-dir results/hard \
    --num-workers 8 \
    --no-instruction

COUNT=$(get_success_count)
echo "$(date '+%H:%M:%S') [PASS1] Video mode done. Success: $COUNT/$TARGET"

# Pass 2: Image mode fallback for remaining samples (1 worker to avoid TPM)
RETRY=0
while [ "$COUNT" -lt "$TARGET" ] && [ "$RETRY" -lt "$MAX_RETRIES" ]; do
    RETRY=$((RETRY + 1))
    REMAINING=$((TARGET - COUNT))
    echo "$(date '+%H:%M:%S') [PASS2-R$RETRY] Image mode fallback for $REMAINING remaining samples..."

    python3 run_annotate.py \
        --model "doubao.doubao-seed-2-0-pro-260215" \
        --input-type image \
        --fps 4 \
        --output-dir results/hard \
        --num-workers 1 \
        --no-instruction

    COUNT=$(get_success_count)
    echo "$(date '+%H:%M:%S') [PASS2-R$RETRY] Done. Success: $COUNT/$TARGET"

    if [ "$COUNT" -lt "$TARGET" ]; then
        echo "$(date '+%H:%M:%S') [WAIT] Sleeping 60s before retry..."
        sleep 60
    fi
done

if [ "$COUNT" -ge "$TARGET" ]; then
    echo "$(date '+%H:%M:%S') [DONE] All $TARGET samples completed successfully!"
else
    echo "$(date '+%H:%M:%S') [WARN] Only $COUNT/$TARGET completed after $MAX_RETRIES retries"
fi
