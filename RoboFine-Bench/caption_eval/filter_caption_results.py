#!/usr/bin/env python3
"""Split caption results into successful and failed records.

DirectAlign should not score failed or empty caption records as normal captions:
an empty caption receives full anti-hallucination credit and can become 0.3333
instead of a true failure. This helper writes a clean success JSONL for
DirectAlign and a sidecar failed JSONL for retry/debugging.
"""

import argparse
import json
from pathlib import Path


def _is_success(record: dict) -> bool:
    if not record.get("call_success"):
        return False
    caption = record.get("caption_result")
    if isinstance(caption, list):
        return any(str(step).strip() for step in caption)
    return bool(str(caption or "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CaptionResult JSONL")
    parser.add_argument("--success-output", required=True, help="Successful records JSONL")
    parser.add_argument("--failed-output", required=True, help="Failed records JSONL")
    args = parser.parse_args()

    input_path = Path(args.input)
    success_path = Path(args.success_output)
    failed_path = Path(args.failed_output)
    success_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    total = success = failed = bad_json = 0
    with input_path.open("r", encoding="utf-8") as src, \
            success_path.open("w", encoding="utf-8") as ok_f, \
            failed_path.open("w", encoding="utf-8") as fail_f:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                fail_f.write(json.dumps({
                    "call_success": False,
                    "error": "invalid_json",
                    "raw_line": line,
                }, ensure_ascii=False) + "\n")
                continue

            if _is_success(record):
                success += 1
                ok_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                failed += 1
                fail_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"caption_filter input={input_path} total={total} "
        f"success={success} failed={failed} bad_json={bad_json} "
        f"success_output={success_path} failed_output={failed_path}"
    )


if __name__ == "__main__":
    main()
