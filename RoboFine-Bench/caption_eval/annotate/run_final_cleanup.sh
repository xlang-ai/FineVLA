#!/bin/bash
# Final cleanup: wait for hard mode, then retry easy mode last sample
# Usage: nohup bash run_final_cleanup.sh > /tmp/doubao_cleanup.log 2>&1 &

cd /mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-Bench/caption_eval/annotate

HARD_PID=1620010
EASY_RESULT="results/easy/doubao_doubao-seed-2-0-pro-260215_CaptionResult.jsonl"
HARD_RESULT="results/hard/doubao_doubao-seed-2-0-pro-260215_CaptionResult.jsonl"

get_count() {
    local file=$1
    if [ ! -f "$file" ]; then echo 0; return; fi
    python3 -c "
import json
count = 0
with open('$file') as f:
    for line in f:
        d = json.loads(line)
        if d.get('call_success'):
            count += 1
print(count)
"
}

echo "$(date '+%H:%M:%S') [CLEANUP] Waiting for hard mode (PID $HARD_PID) to finish..."

# Wait for hard mode script to complete
while kill -0 $HARD_PID 2>/dev/null; do
    sleep 30
done

HARD_COUNT=$(get_count "$HARD_RESULT")
echo "$(date '+%H:%M:%S') [CLEANUP] Hard mode done: $HARD_COUNT/500"

# Now retry easy mode last sample(s)
EASY_COUNT=$(get_count "$EASY_RESULT")
echo "$(date '+%H:%M:%S') [CLEANUP] Easy mode current: $EASY_COUNT/500"

if [ "$EASY_COUNT" -lt 500 ]; then
    echo "$(date '+%H:%M:%S') [CLEANUP] Retrying easy mode remaining $((500 - EASY_COUNT)) samples (image mode, 1 worker)..."

    # Kill any lingering easy mode process
    pkill -f "run_annotate.*results/easy" 2>/dev/null || true
    sleep 5

    for i in 1 2 3 4 5; do
        python3 run_annotate.py \
            --model "doubao.doubao-seed-2-0-pro-260215" \
            --input-type image \
            --fps 4 \
            --output-dir results/easy \
            --num-workers 1

        EASY_COUNT=$(get_count "$EASY_RESULT")
        echo "$(date '+%H:%M:%S') [CLEANUP] Easy retry $i: $EASY_COUNT/500"

        if [ "$EASY_COUNT" -ge 500 ]; then
            break
        fi
        sleep 60
    done
fi

EASY_FINAL=$(get_count "$EASY_RESULT")
HARD_FINAL=$(get_count "$HARD_RESULT")
echo "$(date '+%H:%M:%S') [FINAL] Easy: $EASY_FINAL/500, Hard: $HARD_FINAL/500"

if [ "$EASY_FINAL" -ge 500 ] && [ "$HARD_FINAL" -ge 500 ]; then
    echo "$(date '+%H:%M:%S') [SUCCESS] All done! Both easy and hard modes completed."
else
    echo "$(date '+%H:%M:%S') [INCOMPLETE] Easy: $EASY_FINAL/500, Hard: $HARD_FINAL/500"
fi
