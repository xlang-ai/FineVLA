#!/usr/bin/env python3
"""
VQA result report: print accuracy breakdown and update VQATest_Score.csv.

Usage:
    python vqa_report.py results/qwen3-vl-plus_vqa_result.jsonl   # print report for one model
    python vqa_report.py --update-csv                              # update CSV from all results
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "VQATest_Score.csv")

CAPABILITIES = [
    "action_sequence", "active_actor", "target_object",
    "initial_configuration", "final_configuration",
    "contact_and_approach", "trajectory_and_orientation",
    "object_interaction", "failure_and_recovery", "body_motion",
]

ANSWER_TYPES = ["yes_no", "number", "multiple_choice"]
ERROR_TYPES = ["omission", "contradiction", "hallucination"]


# =========================================================================
# Single-file report (print to terminal)
# =========================================================================

def load_results(path: str) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_type") == "summary":
                continue
            results.append(r)
    return results


def print_report(path: str):
    results = load_results(path)
    if not results:
        print("No results to report.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))

    print(f"\n{'='*70}")
    print(f"  VQA Report: {path}")
    print(f"{'='*70}")
    print(f"  Total: {total},  Correct: {correct},  Accuracy: {correct/total:.1%}")

    # By answer_type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        t = r.get("answer_type", "unknown")
        by_type[t]["total"] += 1
        if r.get("correct"):
            by_type[t]["correct"] += 1

    print(f"\n  {'Answer Type':<20s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"  {'-'*48}")
    for t in ANSWER_TYPES:
        if t in by_type:
            d = by_type[t]
            acc = d["correct"] / d["total"] if d["total"] else 0
            print(f"  {t:<20s} {d['correct']:>8d} {d['total']:>8d} {acc:>10.1%}")

    # By capability
    by_cap = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        cap = r.get("capability", "unknown")
        by_cap[cap]["total"] += 1
        if r.get("correct"):
            by_cap[cap]["correct"] += 1

    print(f"\n  {'Capability':<35s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"  {'-'*63}")
    for cap in sorted(by_cap, key=lambda k: -by_cap[k]["total"]):
        d = by_cap[cap]
        acc = d["correct"] / d["total"] if d["total"] else 0
        print(f"  {cap:<35s} {d['correct']:>8d} {d['total']:>8d} {acc:>10.1%}")

    # By error_type
    by_err = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        et = r.get("error_type", "unknown")
        by_err[et]["total"] += 1
        if r.get("correct"):
            by_err[et]["correct"] += 1

    print(f"\n  {'Error Type':<20s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"  {'-'*48}")
    for et in sorted(by_err, key=lambda k: -by_err[k]["total"]):
        d = by_err[et]
        acc = d["correct"] / d["total"] if d["total"] else 0
        print(f"  {et:<20s} {d['correct']:>8d} {d['total']:>8d} {acc:>10.1%}")

    # Sample wrong answers
    wrong = [r for r in results if not r.get("correct")]
    if wrong:
        print(f"\n  Sample wrong answers (first 5):")
        print(f"  {'-'*63}")
        for r in wrong[:5]:
            print(f"  [{r.get('question_id','')}] {r.get('answer_type','')}")
            print(f"    Q: {r.get('question','')[:80]}")
            print(f"    GT: {r.get('gt_answer','')}")
            print(f"    Model: {r.get('model_answer_parsed','')} (raw: {str(r.get('model_answer_raw',''))[:50]})")
            print()

    print(f"{'='*70}\n")


# =========================================================================
# CSV generation (all models)
# =========================================================================

def _extract_summary(filepath: str) -> dict:
    """Extract summary record from a result JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_type") == "summary":
                return r
    return None


def _extract_model_and_round(filepath: str) -> tuple:
    """Extract (model_name, round_number) from result filename.

    Examples:
        qwen3_5-plus_round1_vqa_result.jsonl  -> ("qwen3.5-plus", 1)
        qwen3_5-plus_vqa_result.jsonl         -> ("qwen3.5-plus", None)
    """
    import re
    basename = os.path.basename(filepath).replace("_vqa_result.jsonl", "")
    round_match = re.search(r'_round(\d+)$', basename)
    if round_match:
        round_num = int(round_match.group(1))
        model_tag = basename[:round_match.start()]
    else:
        round_num = None
        model_tag = basename
    return model_tag, round_num


def update_csv():
    """Scan all result files and generate/update VQATest_Score.csv.

    For multi-round results, computes mean and std per model across rounds.
    Output:
      - Per-round rows: "model (round N)"
      - Aggregated row:  "model (mean±std)"
    """
    import re
    from statistics import mean, stdev

    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_vqa_result.jsonl")))

    if not result_files:
        print("No result files found in", RESULTS_DIR)
        return

    # Value columns
    value_cols = ["overall"]
    value_cols += [f"cap_{c}" for c in CAPABILITIES]
    value_cols += [f"type_{t}" for t in ANSWER_TYPES]
    value_cols += [f"err_{e}" for e in ERROR_TYPES]

    # CSV header: model, round, then value columns
    header = ["model", "round"] + value_cols

    # Collect all per-file rows, grouped by model_tag
    from collections import OrderedDict
    model_rounds = OrderedDict()  # model_tag -> [(round, row_dict)]
    all_rows = []

    for fpath in result_files:
        summary = _extract_summary(fpath)
        if not summary:
            print(f"  SKIP {os.path.basename(fpath)}: no summary record")
            continue

        model_tag, round_num = _extract_model_and_round(fpath)
        model_name = summary.get("model", model_tag)

        row = {"model": model_name, "round": round_num or ""}
        row["overall"] = summary.get("overall_accuracy", 0)

        cap_data = summary.get("accuracy_by_capability", {})
        for c in CAPABILITIES:
            row[f"cap_{c}"] = cap_data.get(c, {}).get("accuracy", "")

        type_data = summary.get("accuracy_by_answer_type", {})
        for t in ANSWER_TYPES:
            row[f"type_{t}"] = type_data.get(t, {}).get("accuracy", "")

        err_data = summary.get("accuracy_by_error_type", {})
        for e in ERROR_TYPES:
            row[f"err_{e}"] = err_data.get(e, {}).get("accuracy", "")

        all_rows.append(row)

        if model_name not in model_rounds:
            model_rounds[model_name] = []
        model_rounds[model_name].append((round_num, row))

        round_str = f" round{round_num}" if round_num else ""
        print(f"  {model_name}{round_str}: overall={row['overall']}")

    # Build final CSV rows: per-round rows + mean±std aggregated rows
    csv_rows = []
    for model_name, rounds in model_rounds.items():
        # Sort rounds by round number
        rounds.sort(key=lambda x: (x[0] or 0))
        for round_num, row in rounds:
            csv_rows.append(row)

        # If multiple rounds, add aggregated mean±std row
        if len(rounds) > 1:
            agg_row = {"model": model_name, "round": "mean"}
            for col in value_cols:
                values = [r[col] for _, r in rounds if r.get(col, "") != ""]
                values = [float(v) for v in values if v != ""]
                if values:
                    m = mean(values)
                    s = stdev(values) if len(values) > 1 else 0.0
                    agg_row[col] = f"{m:.4f}+-{s:.4f}"
                else:
                    agg_row[col] = ""
            csv_rows.append(agg_row)
            print(f"  {model_name} mean: overall={agg_row['overall']} ({len(rounds)} rounds)")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved to {OUTPUT_CSV} ({len(csv_rows)} rows, {len(model_rounds)} models)")


# =========================================================================
# Recalculate summary from raw records
# =========================================================================

def recalc_summary(filepath: str):
    """Recalculate summary from raw records and rewrite the file."""
    results = load_results(filepath)
    if not results:
        print(f"  SKIP {os.path.basename(filepath)}: no records")
        return

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))

    by_at = defaultdict(lambda: {"correct": 0, "total": 0})
    by_cap = defaultdict(lambda: {"correct": 0, "total": 0})
    by_err = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        at = r.get("answer_type", "unknown")
        cap = r.get("capability", "unknown")
        et = r.get("error_type", "unknown")
        by_at[at]["total"] += 1
        by_cap[cap]["total"] += 1
        by_err[et]["total"] += 1
        if r.get("correct"):
            by_at[at]["correct"] += 1
            by_cap[cap]["correct"] += 1
            by_err[et]["correct"] += 1

    # Extract model name from filename
    basename = os.path.basename(filepath)
    model = basename.replace("_vqa_result.jsonl", "").replace("_", ".")

    summary = {
        "_type": "summary",
        "model": model,
        "total_questions": total,
        "total_correct": correct,
        "overall_accuracy": round(correct / total, 4) if total else 0,
        "accuracy_by_answer_type": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0}
            for k, v in sorted(by_at.items())
        },
        "accuracy_by_capability": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0}
            for k, v in sorted(by_cap.items())
        },
        "accuracy_by_error_type": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0}
            for k, v in sorted(by_err.items())
        },
    }

    # Rewrite: raw records + new summary
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_type") == "summary":
                continue
            lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"  {model}: {total} questions, accuracy={summary['overall_accuracy']}")


def recalc_all():
    """Recalculate summaries for all result files, then update CSV."""
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_vqa_result.jsonl")))
    if not result_files:
        print("No result files found in", RESULTS_DIR)
        return

    print("Recalculating summaries...")
    for fpath in result_files:
        recalc_summary(fpath)

    print("\nUpdating CSV...")
    update_csv()


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vqa_report.py <result.jsonl>     # print report for one model")
        print("  python vqa_report.py --update-csv        # update VQATest_Score.csv")
        print("  python vqa_report.py --recalc            # recalculate all summaries + update CSV")
        sys.exit(1)

    if sys.argv[1] == "--update-csv":
        update_csv()
    elif sys.argv[1] == "--recalc":
        recalc_all()
    else:
        print_report(sys.argv[1])
