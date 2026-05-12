"""Preprocessing for EvalSets/Human_Review and CaptionResult data."""

import json
import re
from typing import Optional


def _clean_step(text: str) -> str:
    """Normalize whitespace in a single step string."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_evalsets(evalsets_path: str) -> list[dict]:
    """Load EvalSets.json and return normalized GT records.

    Supports both:
      - EvalSets.json (JSON array, GT field = "GT")
      - Human_Review.jsonl (JSONL, GT field = "human_review"/"human_reveiw")

    Returns list of:
      {"sample_id": str, "status": "ok"|"skip_black_screen_gt"|"skip_malformed_gt",
       "gt_steps": list[str], "preprocess_notes": list[str]}
    """
    records = _load_records(evalsets_path)
    results = []
    for record in records:
        sid = record.get("sample_id", "")
        if not sid:
            continue

        gt_raw = (record.get("GT")
                  or record.get("human_reveiw")
                  or record.get("human_review"))
        notes: list[str] = []

        if gt_raw is None:
            results.append({
                "sample_id": sid,
                "status": "skip_malformed_gt",
                "gt_steps": [],
                "preprocess_notes": ["GT field missing"],
            })
            continue

        if isinstance(gt_raw, str) and gt_raw.strip() == "错误（黑屏）":
            results.append({
                "sample_id": sid,
                "status": "skip_black_screen_gt",
                "gt_steps": [],
                "preprocess_notes": ["Black screen GT"],
            })
            continue

        steps = _parse_steps(gt_raw, notes)
        if steps is None:
            results.append({
                "sample_id": sid,
                "status": "skip_malformed_gt",
                "gt_steps": [],
                "preprocess_notes": notes,
            })
            continue

        steps = [_clean_step(s) for s in steps if _clean_step(s)]
        if not steps:
            results.append({
                "sample_id": sid,
                "status": "skip_malformed_gt",
                "gt_steps": [],
                "preprocess_notes": notes + ["All steps empty after cleaning"],
            })
            continue

        results.append({
            "sample_id": sid,
            "status": "ok",
            "gt_steps": steps,
            "preprocess_notes": notes,
        })

    return results


# Keep old name as alias for backward compatibility
normalize_human_review = normalize_evalsets


def _load_records(path: str) -> list[dict]:
    """Load records from JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()
    if first_char == "[":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def _parse_steps(raw, notes: list[str]) -> Optional[list[str]]:
    """Try to parse raw GT value into a list of strings."""
    if isinstance(raw, list):
        # Already a list; ensure all elements are strings
        return [str(s) for s in raw if s]

    if not isinstance(raw, str):
        notes.append(f"Unexpected GT type: {type(raw).__name__}")
        return None

    text = raw.strip()

    # Normalize smart quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Replace full-width commas
    text = text.replace("\uff0c", ",")

    # Try json.loads
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            notes.append("Parsed from JSON string")
            return [str(s) for s in parsed if s]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try ast.literal_eval as fallback
    try:
        import ast
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            notes.append("Parsed from ast.literal_eval")
            return [str(s) for s in parsed if s]
    except (ValueError, SyntaxError):
        pass

    # If it looks like a plain string (no brackets), wrap it
    if not text.startswith("["):
        notes.append("Wrapped plain string as single-element list")
        return [text]

    notes.append("Could not parse GT string")
    return None


def load_gt_atomic_facts(gt_facts_path: str) -> dict[str, dict]:
    """Load pre-extracted GT atomic facts, keyed by sample_id."""
    gt_map = {}
    with open(gt_facts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("sample_id", "")
            if sid:
                gt_map[sid] = record
    return gt_map


def load_caption_results(caption_path: str) -> dict[str, list[str]]:
    """Load CaptionResult.jsonl, return {sample_id: steps}.

    Supports both old format (fineGrainedSteps) and new format (caption_result).
    """
    caption_map = {}
    with open(caption_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("sample_id", "")
            if not sid:
                continue
            steps = (record.get("caption_result")
                     or record.get("fineGrainedSteps")
                     or [])
            if isinstance(steps, str):
                steps = [steps]
            elif not isinstance(steps, list):
                steps = []
            steps = [_clean_step(str(s)) for s in steps if s]
            caption_map[sid] = steps
    return caption_map


def build_extraction_payload(sample_id: str, steps: list[str]) -> dict:
    """Build unified extraction API payload (works for both GT and caption)."""
    return {
        "sample_id": sample_id,
        "steps": steps,
    }


# Legacy aliases
def build_gt_extraction_payload(sample_id: str, steps: list[str]) -> dict:
    return build_extraction_payload(sample_id, steps)


def build_caption_extraction_payload(
    sample_id: str, caption_steps: list[str]
) -> dict:
    return build_extraction_payload(sample_id, caption_steps)


def build_alignment_payload(
    sample_id: str, gt_result: dict, caption_result: dict
) -> dict:
    """Build Phase 2b API payload with pre-extracted GT and caption facts."""
    return {
        "sample_id": sample_id,
        "gt_atomic_facts_by_capability": gt_result["capability_results"],
        "caption_atomic_facts_by_capability": caption_result["capability_results"],
    }


def _slim_fact(fact: dict) -> dict:
    """Keep only the fields needed for alignment: fact_id, slot, value, fact_text."""
    return {
        "fact_id": fact["fact_id"],
        "slot": fact.get("slot", ""),
        "value": fact.get("value", ""),
        "fact_text": fact.get("fact_text", ""),
    }


def build_per_cap_alignment_payload(
    sample_id: str, capability: str,
    gt_facts: list[dict], caption_facts: list[dict],
) -> dict:
    """Build Phase 2b API payload for a single capability."""
    return {
        "sample_id": sample_id,
        "capability": capability,
        "gt_atomic_facts": [_slim_fact(f) for f in gt_facts],
        "caption_atomic_facts": [_slim_fact(f) for f in caption_facts],
    }


def build_per_slot_alignment_payload(
    sample_id: str, capability: str, slot: str,
    gt_facts: list[dict], caption_facts: list[dict],
) -> dict:
    """Build Phase 2b API payload for a single (capability, slot) bucket."""
    return {
        "sample_id": sample_id,
        "capability": capability,
        "slot": slot,
        "gt_atomic_facts": gt_facts,
        "caption_atomic_facts": caption_facts,
    }


def group_facts_by_slot(facts: list[dict]) -> dict[str, list[dict]]:
    """Group atomic facts by their 'slot' field. Returns {slot: [facts]}."""
    by_slot: dict[str, list[dict]] = {}
    for f in facts:
        s = f.get("slot", "_unknown")
        by_slot.setdefault(s, []).append(f)
    return by_slot


def build_direct_alignment_payload(
    sample_id: str,
    gt_result: dict,
    caption_steps: list[str],
) -> dict:
    """Build direct alignment payload: GT atomic facts + raw caption steps."""
    gt_by_cap = {}
    for cr in gt_result.get("capability_results", []):
        cap = cr.get("capability", "")
        facts = cr.get("gt_atomic_facts", [])
        gt_by_cap[cap] = [_slim_fact(f) for f in facts]
    return {
        "sample_id": sample_id,
        "gt_atomic_facts": gt_by_cap,
        "caption_steps": caption_steps,
    }


def load_caption_atomic_facts(cap_facts_path: str) -> dict[str, dict]:
    """Load pre-extracted caption atomic facts, keyed by sample_id."""
    cap_map = {}
    with open(cap_facts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("sample_id", "")
            if sid:
                cap_map[sid] = record
    return cap_map
