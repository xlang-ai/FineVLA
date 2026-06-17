#!/usr/bin/env python3
"""Merge incremental DirectAlign records and recompute dataset summaries.

Use this after rerunning DirectAlign on a small subset of fixed caption samples.
The script replaces matching sample_ids in the base raw/scored JSONL files with
the incremental records, then recomputes dataset_summary.json/csv.
"""

import argparse
import csv
import json
from pathlib import Path


CAPABILITIES = [
    "action_sequence",
    "active_actor",
    "target_object",
    "initial_configuration",
    "final_configuration",
    "contact_and_approach",
    "trajectory_and_orientation",
    "object_interaction",
    "failure_and_recovery",
    "body_motion",
]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_by_sample_id(base: list[dict], patch: list[dict]) -> list[dict]:
    patch_by_id = {r.get("sample_id"): r for r in patch if r.get("sample_id")}
    seen = set()
    merged = []
    for record in base:
        sid = record.get("sample_id")
        if sid in patch_by_id:
            merged.append(patch_by_id[sid])
            seen.add(sid)
        else:
            merged.append(record)
    for sid, record in patch_by_id.items():
        if sid not in seen:
            merged.append(record)
    return merged


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def recompute_summary(records: list[dict], model_name: str) -> dict:
    ok_records = [r for r in records if not r.get("error")]
    summary = {
        "model_name": model_name,
        "num_total_samples": len(records),
        "num_scored_samples": len(ok_records),
        "num_errors": len(records) - len(ok_records),
        "mean_consistency_score": round(_mean([r.get("consistency_score") for r in ok_records]) or 0.0, 4),
        "mean_weighted_coverage": round(_mean([r.get("weighted_coverage") for r in ok_records]) or 0.0, 4),
        "mean_weighted_anti_hallucination": round(
            _mean([r.get("weighted_anti_hallucination") for r in ok_records]) or 0.0, 4
        ),
        "mean_caption_score": round(_mean([r.get("caption_score") for r in ok_records]) or 0.0, 4),
        "per_capability": {},
    }

    per_cap = {cap: {"G": 0, "C": 0, "M": 0, "P": 0, "X": 0, "O": 0, "H": 0} for cap in CAPABILITIES}
    for record in ok_records:
        for cap_score in record.get("capability_scores", []):
            cap = cap_score.get("capability")
            if cap not in per_cap:
                continue
            for key in ["G", "C", "M", "P", "X", "O", "H"]:
                per_cap[cap][key] += int(cap_score.get(key) or 0)

    for cap in CAPABILITIES:
        counts = per_cap[cap]
        g, c = counts["G"], counts["C"]
        m, p, x, h = counts["M"], counts["P"], counts["X"], counts["H"]
        a = m + p + x
        consistency = (m + 0.5 * p) / c if c else None
        coverage = (m + 0.5 * p) / g if g else None
        anti_hallucination = a / (a + h) if (a + h) else None
        summary["per_capability"][cap] = {
            "mean_consistency": round(consistency, 4) if consistency is not None else None,
            "mean_coverage": round(coverage, 4) if coverage is not None else None,
            "mean_anti_hallucination": round(anti_hallucination, 4) if anti_hallucination is not None else None,
            "total_G": g,
            "total_C": c,
            "total_M": m,
            "total_P": p,
            "total_X": x,
            "total_O": counts["O"],
            "total_H": h,
        }
    return summary


def write_summary_csv(path: Path, summary: dict) -> None:
    row = {
        "model_name": summary["model_name"],
        "num_scored_samples": summary["num_scored_samples"],
        "num_errors": summary["num_errors"],
        "mean_caption_score": summary["mean_caption_score"],
        "mean_consistency_score": summary["mean_consistency_score"],
        "mean_weighted_coverage": summary["mean_weighted_coverage"],
        "mean_weighted_anti_hallucination": summary["mean_weighted_anti_hallucination"],
    }
    for cap in CAPABILITIES:
        data = summary["per_capability"][cap]
        row[f"{cap}_consistency"] = data["mean_consistency"]
        row[f"{cap}_coverage"] = data["mean_coverage"]
        row[f"{cap}_anti_hallucination"] = data["mean_anti_hallucination"]
        for key in ["G", "C", "M", "P", "X", "O", "H"]:
            row[f"{cap}_{key}"] = data[f"total_{key}"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, help="Existing DirectAlign result dir")
    parser.add_argument("--patch-dir", required=True, help="Incremental DirectAlign result dir")
    parser.add_argument("--output-dir", required=True, help="Merged output dir")
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_raw = merge_by_sample_id(
        read_jsonl(base_dir / "direct_align_raw.jsonl"),
        read_jsonl(patch_dir / "direct_align_raw.jsonl"),
    )
    merged_scored = merge_by_sample_id(
        read_jsonl(base_dir / "scored_results.jsonl"),
        read_jsonl(patch_dir / "scored_results.jsonl"),
    )
    write_jsonl(output_dir / "direct_align_raw.jsonl", merged_raw)
    write_jsonl(output_dir / "scored_results.jsonl", merged_scored)

    summary = recompute_summary(merged_scored, args.model_name)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(output_dir / "dataset_summary.csv", summary)
    print(
        f"merged {len(merged_scored)} scored records; "
        f"mean_caption_score={summary['mean_caption_score']}"
    )


if __name__ == "__main__":
    main()
