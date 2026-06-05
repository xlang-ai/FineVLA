"""Two-phase pipeline orchestration."""

import json
import os
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import CAPABILITIES, CAPABILITY_WEIGHTS, SAVE_EVERY_N

from .normalize import (
    normalize_evalsets,
    load_gt_atomic_facts,
    load_caption_results,
    load_caption_atomic_facts,
    build_gt_extraction_payload,
    build_caption_extraction_payload,
    build_alignment_payload,
    build_per_cap_alignment_payload,
    build_direct_alignment_payload,
)
from .judge_client import (
    create_client,
    call_api,
    parse_gt_extraction,
    parse_caption_extraction,
    parse_alignment,
    parse_capability_judgment,
    parse_direct_alignment,
)
from .scoring import compute_sample_scores, compute_dataset_summary
from .models import (
    SampleScores,
    AlignmentResult,
    FullCapabilityResult,
    CapabilityJudgment,
    AtomicFact,
    Omission,
    Hallucination,
    IgnoredExtra,
    reconstruct_full_capability_result,
)


# ── Phase 1: GT Atomic Fact Extraction ──────────────────────────────

def run_gt_extraction(args):
    """Extract GT atomic facts from EvalSets.json (or Human_Review.jsonl)."""
    # Load prompt
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
    )
    prompt_path = os.path.join(prompts_dir, "extraction.txt")
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(prompts_dir, "gt_extraction.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Load and normalize GT data (supports EvalSets.json and Human_Review.jsonl)
    input_path = getattr(args, 'evalsets', None) or getattr(args, 'human_review', None)
    print(f"Loading GT data: {input_path}")
    samples = normalize_evalsets(input_path)
    ok_samples = [s for s in samples if s["status"] == "ok"]
    print(f"  Total samples: {len(samples)}, valid: {len(ok_samples)}, "
          f"skipped: {len(samples) - len(ok_samples)}")

    # Create client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please provide an API Key (--api-key or OPENAI_API_KEY env var)")
    client = create_client(api_key, args.base_url)

    # Checkpoint: load existing results
    output_path = args.output
    processed_ids = set()
    existing_results = []
    error_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id", "")
                if rec.get("status") == "error":
                    # Keep error records but allow retry
                    error_ids.add(sid)
                else:
                    existing_results.append(rec)
                    processed_ids.add(sid)
        if error_ids:
            print(f"  Succeeded: {len(processed_ids)}, pending retry: {len(error_ids)}")
        else:
            print(f"  Processed: {len(processed_ids)} samples, resuming from checkpoint")

    # Filter remaining
    remaining = [s for s in ok_samples if s["sample_id"] not in processed_ids]
    if not remaining:
        print("All samples processed!")
        return

    print(f"\nStarting GT atomic fact extraction: {len(remaining)} samples "
          f"(model={args.model}, workers={args.num_workers})")
    print("=" * 60)

    results_lock = threading.Lock()
    save_counter = 0
    success_count = 0
    error_count = 0

    def process_one(sample):
        sid = sample["sample_id"]
        payload = build_gt_extraction_payload(sid, sample["gt_steps"])
        response = call_api(
            client, system_prompt, payload,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sample_id=sid,
            enable_thinking=getattr(args, 'enable_thinking', False),
        )
        result = parse_gt_extraction(response, sid)

        if result is None:
            # Retry once with repair note
            repair_payload = payload.copy()
            repair_payload["_repair_note"] = (
                "Your previous response could not be parsed. "
                "Return ONLY a valid JSON object with exactly 10 capability_results."
            )
            response = call_api(
                client, system_prompt, repair_payload,
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sample_id=f"{sid}_retry",
                enable_thinking=getattr(args, 'enable_thinking', False),
            )
            result = parse_gt_extraction(response, sid)

        return sid, result, response

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, s): s for s in remaining}
        pbar = tqdm(total=len(remaining), desc="GT Extraction", unit="sample")

        for future in as_completed(futures):
            sid, result, raw_response = future.result()
            with results_lock:
                if result is not None:
                    rec = result.model_dump()
                    existing_results.append(rec)
                    success_count += 1
                else:
                    existing_results.append({
                        "sample_id": sid,
                        "status": "error",
                        "error": "Parse/validation failed",
                        "raw_response_preview": raw_response[:500] if raw_response else "",
                    })
                    error_count += 1

                save_counter += 1
                if save_counter % SAVE_EVERY_N == 0 or save_counter == len(remaining):
                    _save_jsonl(output_path, existing_results)
            pbar.update(1)
            pbar.set_postfix(ok=success_count, err=error_count)

        pbar.close()

    # Final save
    _save_jsonl(output_path, existing_results)

    print(f"\n{'=' * 60}")
    print(f"GT atomic fact extraction complete!")
    print(f"Succeeded: {success_count}, failed: {error_count}")
    print(f"Results saved to: {output_path}")


# ── Phase 2a: Caption Atomic Fact Extraction ──────────────────────────

def run_caption_extraction(args):
    """Extract atomic facts from AI captions (Phase 2a, with thinking)."""
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
    )
    prompt_path = os.path.join(prompts_dir, "extraction.txt")
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(prompts_dir, "caption_extraction.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Load caption results
    print(f"Loading CaptionResult: {args.caption}")
    caption_map = load_caption_results(args.caption)
    print(f"  Caption samples: {len(caption_map)}")

    # Determine which sample_ids to process (intersect with GT if provided)
    if hasattr(args, 'gt_facts') and args.gt_facts:
        gt_map = load_gt_atomic_facts(args.gt_facts)
        gt_ok_ids = {k for k, v in gt_map.items() if v.get("status") == "ok"}
        target_ids = sorted(set(caption_map.keys()) & gt_ok_ids)
        print(f"  Samples overlapping with GT: {len(target_ids)}")
    else:
        target_ids = sorted(caption_map.keys())

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please provide an API Key (--api-key or OPENAI_API_KEY env var)")
    client = create_client(api_key, args.base_url)

    # Checkpoint
    output_path = os.path.join(output_dir, "caption_atomic_facts.jsonl")
    processed_ids = set()
    existing_results = []
    error_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id", "")
                if rec.get("status") == "error":
                    error_ids.add(sid)
                else:
                    existing_results.append(rec)
                    processed_ids.add(sid)
        if error_ids:
            print(f"  Succeeded: {len(processed_ids)}, pending retry: {len(error_ids)}")
        else:
            print(f"  Processed: {len(processed_ids)} samples, resuming from checkpoint")

    remaining_ids = [sid for sid in target_ids if sid not in processed_ids]
    if not remaining_ids:
        print("Phase 2a: All samples processed!")
        return

    print(f"\nStarting caption atomic fact extraction (Phase 2a): {len(remaining_ids)} samples "
          f"(model={args.model}, workers={args.num_workers}, thinking=ON)")
    print("=" * 60)

    results_lock = threading.Lock()
    save_counter = 0
    success_count = 0
    error_count = 0

    def process_one(sid):
        caption_steps = caption_map[sid]
        payload = build_caption_extraction_payload(sid, caption_steps)
        response = call_api(
            client, system_prompt, payload,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sample_id=sid,
            enable_thinking=True,  # Phase 2a always uses thinking
        )
        result = parse_caption_extraction(response, sid)

        if result is None:
            repair_payload = payload.copy()
            repair_payload["_repair_note"] = (
                "Your previous response could not be parsed. "
                "Return ONLY a valid JSON object with exactly 10 capability_results."
            )
            response = call_api(
                client, system_prompt, repair_payload,
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sample_id=f"{sid}_retry",
                enable_thinking=True,
            )
            result = parse_caption_extraction(response, sid)

        return sid, result, response

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, sid): sid for sid in remaining_ids}
        pbar = tqdm(total=len(remaining_ids), desc="Caption Extraction", unit="sample")

        for future in as_completed(futures):
            sid, result, raw_response = future.result()
            with results_lock:
                if result is not None:
                    existing_results.append(result.model_dump())
                    success_count += 1
                else:
                    existing_results.append({
                        "sample_id": sid,
                        "status": "error",
                        "error": "Parse/validation failed",
                        "raw_response_preview": raw_response[:500] if raw_response else "",
                    })
                    error_count += 1

                save_counter += 1
                if save_counter % SAVE_EVERY_N == 0 or save_counter == len(remaining_ids):
                    _save_jsonl(output_path, existing_results)
            pbar.update(1)
            pbar.set_postfix(ok=success_count, err=error_count)

        pbar.close()

    _save_jsonl(output_path, existing_results)
    print(f"\n{'=' * 60}")
    print(f"Caption atomic fact extraction complete! Succeeded: {success_count}, failed: {error_count}")
    print(f"Results saved to: {output_path}")


# ── Phase 2b: Per-Capability Alignment ──────────────────────────────

def _get_cap_facts(result: dict, capability: str, key: str) -> list[dict]:
    """Extract facts for a specific capability from extraction result."""
    for cr in result.get("capability_results", []):
        if cr.get("capability") == capability:
            return cr.get(key, [])
    return []


def _auto_generate_cap_judgment(
    sample_id: str, capability: str,
    gt_facts: list[dict], cap_facts: list[dict],
) -> CapabilityJudgment:
    """Auto-generate capability-level judgment for trivial cases (no API call needed)."""
    omissions = [
        Omission(gt_fact_id=f["fact_id"], note="Caption has no facts for this capability.")
        for f in gt_facts
    ]
    hallucinations = []
    ignored = []
    for f in cap_facts:
        if capability == "action_sequence":
            hallucinations.append(
                Hallucination(caption_fact_id=f["fact_id"],
                              reason="extra_action_not_supported_by_gt"))
        else:
            ignored.append(
                IgnoredExtra(caption_fact_id=f["fact_id"],
                             reason="non_action_extra_or_not_counted"))
    return CapabilityJudgment(
        sample_id=sample_id,
        capability=capability,
        alignments=[],
        omissions=omissions,
        hallucinations=hallucinations,
        ignored_extras=ignored,
    )


def run_alignment(args):
    """Align caption facts against GT facts (Phase 2b, per-capability)."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts", "caption_alignment_percap.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"Loading GT atomic facts: {args.gt_facts}")
    gt_map = load_gt_atomic_facts(args.gt_facts)
    gt_ok = {k: v for k, v in gt_map.items() if v.get("status") == "ok"}
    print(f"  Valid GT: {len(gt_ok)}")

    output_dir = args.output_dir
    cap_facts_path = os.path.join(output_dir, "caption_atomic_facts.jsonl")
    print(f"Loading caption atomic facts: {cap_facts_path}")
    cap_map = load_caption_atomic_facts(cap_facts_path)
    cap_ok = {k: v for k, v in cap_map.items() if v.get("status") == "ok"}
    print(f"  Valid caption facts: {len(cap_ok)}")

    caption_basename = os.path.basename(args.caption)
    if "_CaptionResult" in caption_basename:
        model_name = caption_basename.split("_CaptionResult")[0]
    else:
        model_name = os.path.splitext(caption_basename)[0]

    os.makedirs(output_dir, exist_ok=True)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please provide an API Key (--api-key or OPENAI_API_KEY env var)")
    client = create_client(api_key, args.base_url)

    common_ids = sorted(set(gt_ok.keys()) & set(cap_ok.keys()))
    print(f"  Overlapping samples: {len(common_ids)}")

    # ── Checkpoint: load capability-level progress ──
    progress_path = os.path.join(output_dir, "alignment_progress.jsonl")
    completed_caps: dict[tuple[str, str], dict] = {}
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    key = (rec["sample_id"], rec["capability"])
                    completed_caps[key] = rec["judgment"]
        print(f"  Completed capability alignments: {len(completed_caps)}")

    rerun_all = getattr(args, "rerun_all", False)
    judge_raw_path = os.path.join(output_dir, "judge_raw.jsonl")
    fully_done_ids = set()
    existing_judge = []
    if os.path.exists(judge_raw_path) and not rerun_all:
        skipped_errors = 0
        with open(judge_raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                raw = rec.get("raw_response", {})
                if isinstance(raw, dict) and raw.get("status") == "error":
                    skipped_errors += 1
                    continue
                existing_judge.append(rec)
                fully_done_ids.add(rec.get("sample_id", ""))
        print(f"  Completed samples (judge_raw): {len(fully_done_ids)}, "
              f"discarded old errors: {skipped_errors}")
    elif rerun_all:
        print("  --rerun-all: Ignoring existing results, re-aligning all")
        if os.path.exists(progress_path):
            os.remove(progress_path)
            completed_caps.clear()

    # ── Enumerate (sid, capability) work items ──
    work_items = []  # (sample_id, capability)
    auto_generated = 0

    for sid in common_ids:
        if sid in fully_done_ids:
            continue
        for cap in CAPABILITIES:
            if (sid, cap) in completed_caps:
                continue
            gt_facts = _get_cap_facts(gt_ok[sid], cap, "gt_atomic_facts")
            cap_facts = _get_cap_facts(cap_ok[sid], cap, "caption_atomic_facts")

            if len(gt_facts) == 0 and len(cap_facts) == 0:
                completed_caps[(sid, cap)] = _auto_generate_cap_judgment(
                    sid, cap, gt_facts, cap_facts).model_dump()
                auto_generated += 1
            elif len(cap_facts) == 0:
                completed_caps[(sid, cap)] = _auto_generate_cap_judgment(
                    sid, cap, gt_facts, cap_facts).model_dump()
                auto_generated += 1
            elif len(gt_facts) == 0:
                completed_caps[(sid, cap)] = _auto_generate_cap_judgment(
                    sid, cap, gt_facts, cap_facts).model_dump()
                auto_generated += 1
            else:
                work_items.append((sid, cap))

    if auto_generated > 0:
        print(f"  Auto-generated: {auto_generated} capabilities (trivial cases)")

    remaining_samples = set(sid for sid, _ in work_items)
    if not work_items:
        _assemble_all_and_save(
            common_ids, fully_done_ids, completed_caps,
            gt_ok, cap_ok, existing_judge, judge_raw_path,
        )
        print("Phase 2b: All samples processed! Recomputing scores...")
        _finalize_outputs(existing_judge, gt_ok, output_dir, model_name)
        return

    print(f"\nStarting alignment (Phase 2b, per-capability): {len(work_items)} work items "
          f"({len(remaining_samples)} samples, model={args.model}, "
          f"workers={args.num_workers}, thinking=OFF)")
    print("=" * 60)

    errors_path = os.path.join(output_dir, "errors.jsonl")
    results_lock = threading.Lock()
    save_counter = 0
    success_count = 0
    error_count = 0
    FLUSH_EVERY = 50

    def _call_and_parse_cap(sid, cap, payload, gt_facts, cap_facts, suffix=""):
        response = call_api(
            client, system_prompt, payload,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sample_id=f"{sid}/{cap}{suffix}",
            enable_thinking=False,
        )
        result = parse_capability_judgment(response, sid, cap)
        all_errors = []
        if result is not None:
            all_errors = result.check_sanity(len(gt_facts), len(cap_facts))
            gt_ids = {f["fact_id"] for f in gt_facts}
            cap_ids = {f["fact_id"] for f in cap_facts}
            all_errors.extend(result.validate_fact_ids(gt_ids, cap_ids))
            if all_errors:
                result = None
        return result, response, all_errors

    def process_one_cap(sid, cap):
        gt_facts = _get_cap_facts(gt_ok[sid], cap, "gt_atomic_facts")
        cap_facts = _get_cap_facts(cap_ok[sid], cap, "caption_atomic_facts")
        payload = build_per_cap_alignment_payload(sid, cap, gt_facts, cap_facts)

        result, response, all_errors = _call_and_parse_cap(
            sid, cap, payload, gt_facts, cap_facts)

        if not response.strip():
            for retry_i in range(2):
                time.sleep(1.5 * (retry_i + 1))
                result, response, all_errors = _call_and_parse_cap(
                    sid, cap, payload, gt_facts, cap_facts,
                    suffix=f"_empty_retry{retry_i + 1}")
                if response.strip():
                    break

        if result is None and response.strip():
            repair_note = "Your previous response failed validation."
            if all_errors:
                repair_note += f" Errors: {'; '.join(all_errors[:5])}"
            repair_note += " Please fix and re-output valid JSON."
            repair_payload = payload.copy()
            repair_payload["_repair_note"] = repair_note
            result, response, all_errors = _call_and_parse_cap(
                sid, cap, repair_payload, gt_facts, cap_facts, suffix="_retry")

        return sid, cap, result, response, all_errors

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one_cap, sid, cap): (sid, cap)
            for sid, cap in work_items
        }
        pbar = tqdm(total=len(work_items), desc="Alignment(per-cap)", unit="cap")

        for future in as_completed(futures):
            sid, cap, result, raw_response, errors_list = future.result()
            with results_lock:
                if result is not None:
                    completed_caps[(sid, cap)] = result.model_dump()
                    success_count += 1
                else:
                    completed_caps[(sid, cap)] = {
                        "sample_id": sid, "capability": cap,
                        "status": "error",
                        "errors": errors_list or [],
                        "raw_response_preview": raw_response[:500] if raw_response else "",
                    }
                    error_count += 1
                    error_rec = {
                        "sample_id": sid, "capability": cap,
                        "status": "error",
                        "error": "Parse/validation/sanity failed",
                        "errors": errors_list or [],
                        "raw_response_preview": raw_response[:500] if raw_response else "",
                    }
                    with open(errors_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(error_rec, ensure_ascii=False) + "\n")

                # Try assembling the full sample
                if (sid not in fully_done_ids
                        and all((sid, c) in completed_caps for c in CAPABILITIES)):
                    _assemble_one_sample(
                        sid, completed_caps, gt_ok, cap_ok,
                        existing_judge, errors_path,
                    )
                    fully_done_ids.add(sid)

                save_counter += 1
                if save_counter % FLUSH_EVERY == 0:
                    _flush_cap_progress(progress_path, completed_caps)
                    _save_jsonl(judge_raw_path, existing_judge)

            pbar.update(1)
            pbar.set_postfix(ok=success_count, err=error_count)

        pbar.close()

    # Final flush
    _flush_cap_progress(progress_path, completed_caps)

    # Assemble remaining samples
    _assemble_all_and_save(
        common_ids, fully_done_ids, completed_caps,
        gt_ok, cap_ok, existing_judge, judge_raw_path,
    )

    _save_jsonl(judge_raw_path, existing_judge)
    print(f"\n{'=' * 60}")
    print(f"Alignment complete! Succeeded: {success_count} caps, failed: {error_count} caps")

    sample_errors = []
    for rec in existing_judge:
        raw = rec.get("raw_response", {})
        if isinstance(raw, dict) and raw.get("status") == "error":
            sample_errors.append(rec["sample_id"])
    if sample_errors:
        print(f"\nFailed samples ({len(sample_errors)}):")
        for sid in sample_errors[:20]:
            print(f"  - {sid}")
        if len(sample_errors) > 20:
            print(f"  ... and {len(sample_errors) - 20} more")
        print(f"\nTip: Re-run the same command to automatically retry failed capabilities")

    _finalize_outputs(existing_judge, gt_ok, output_dir, model_name)


def _assemble_one_sample(
    sid: str,
    completed_caps: dict[tuple[str, str], dict],
    gt_ok: dict, cap_ok: dict,
    existing_judge: list[dict],
    errors_path: str,
):
    """Assemble per-capability judgments into one AlignmentResult."""
    has_error = False
    cap_errors = []
    capability_results = []

    for cap in CAPABILITIES:
        judgment_data = completed_caps.get((sid, cap))
        if judgment_data is None or judgment_data.get("status") == "error":
            has_error = True
            cap_errors.append(cap)
            continue

        gt_facts_raw = _get_cap_facts(gt_ok[sid], cap, "gt_atomic_facts")
        cap_facts_raw = _get_cap_facts(cap_ok[sid], cap, "caption_atomic_facts")

        gt_facts = [AtomicFact.model_validate(f) for f in gt_facts_raw]
        cap_facts = [AtomicFact.model_validate(f) for f in cap_facts_raw]

        try:
            judgment = CapabilityJudgment.model_validate(judgment_data)
            full = reconstruct_full_capability_result(judgment, gt_facts, cap_facts)
            capability_results.append(full)
        except Exception:
            has_error = True
            cap_errors.append(cap)

    if has_error:
        error_rec = {
            "sample_id": sid,
            "status": "error",
            "error": f"Failed capabilities: {', '.join(cap_errors)}",
        }
        existing_judge.append({"sample_id": sid, "raw_response": error_rec})
    else:
        alignment = AlignmentResult(
            sample_id=sid,
            status="ok",
            capability_results=capability_results,
        )
        existing_judge.append({
            "sample_id": sid,
            "raw_response": alignment.model_dump(),
        })


def _assemble_all_and_save(
    common_ids, fully_done_ids, completed_caps,
    gt_ok, cap_ok, existing_judge, judge_raw_path,
):
    """Assemble all samples that have all completed capabilities."""
    errors_path = os.path.join(os.path.dirname(judge_raw_path), "errors.jsonl")
    for sid in common_ids:
        if sid in fully_done_ids:
            continue
        all_done = all((sid, cap) in completed_caps for cap in CAPABILITIES)
        if all_done:
            _assemble_one_sample(
                sid, completed_caps, gt_ok, cap_ok,
                existing_judge, errors_path,
            )
            fully_done_ids.add(sid)
    _save_jsonl(judge_raw_path, existing_judge)


def _flush_cap_progress(
    progress_path: str,
    completed_caps: dict[tuple[str, str], dict],
):
    """Write all completed capability results to progress file."""
    with open(progress_path, "w", encoding="utf-8") as f:
        for (sid, cap), judgment in completed_caps.items():
            status = judgment.get("status", "ok") if isinstance(judgment, dict) else "ok"
            rec = {
                "sample_id": sid,
                "capability": cap,
                "status": status,
                "judgment": judgment,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Phase 2 (combined): Caption Evaluation ─────────────────────────────

def run_evaluation(args):
    """Full evaluation: extract caption facts (2a) + align (2b)."""
    print("=" * 60)
    print("  Phase 2a: Caption atomic fact extraction (thinking=ON)")
    print("=" * 60)
    run_caption_extraction(args)

    print("\n")
    print("=" * 60)
    print("  Phase 2b: Alignment (thinking=OFF)")
    print("=" * 60)
    run_alignment(args)


def _finalize_outputs(judge_records, gt_ok, output_dir, model_name):
    """Compute scores and write final output files."""
    scored_path = os.path.join(output_dir, "scored_results.jsonl")
    summary_json_path = os.path.join(output_dir, "dataset_summary.json")
    summary_csv_path = os.path.join(output_dir, "dataset_summary.csv")

    all_scores: list[SampleScores] = []

    for rec in judge_records:
        sid = rec.get("sample_id", "")
        raw = rec.get("raw_response", {})

        if isinstance(raw, dict) and raw.get("status") == "error":
            all_scores.append(SampleScores(
                sample_id=sid, status="error",
                capability_scores=[], consistency_score=0.0,
                weighted_coverage=0.0, weighted_anti_hallucination=0.0,
                caption_score=0.0, error=raw.get("error", "unknown"),
            ))
            continue

        try:
            from .models import AlignmentResult
            alignment = AlignmentResult.model_validate(raw)
            scores = compute_sample_scores(alignment)
            all_scores.append(scores)
        except Exception as e:
            all_scores.append(SampleScores(
                sample_id=sid, status="error",
                capability_scores=[], consistency_score=0.0,
                weighted_coverage=0.0, weighted_anti_hallucination=0.0,
                caption_score=0.0, error=str(e),
            ))

    # Write scored_results.jsonl
    with open(scored_path, "w", encoding="utf-8") as f:
        for s in all_scores:
            f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

    # Compute and write dataset summary
    summary = compute_dataset_summary(all_scores, model_name)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Write CSV summary
    _write_summary_csv(summary, summary_csv_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Scored samples: {summary['num_scored_samples']}/{summary['num_total_samples']}")
    print(f"Mean CaptionScore:     {summary['mean_caption_score']:.4f}")
    print(f"Mean Consistency:      {summary['mean_consistency_score']:.4f}")
    print(f"Mean Coverage:         {summary['mean_weighted_coverage']:.4f}")
    print(f"Mean AntiHallucinate:  {summary['mean_weighted_anti_hallucination']:.4f}")

    print(f"\n{'Capability':<40} {'Consistency':>12} {'Coverage':>10} {'AntiHalluc':>12}")
    print("-" * 76)
    for cap in CAPABILITIES:
        cd = summary["per_capability"].get(cap, {})
        c = cd.get("mean_consistency")
        v = cd.get("mean_coverage")
        a = cd.get("mean_anti_hallucination")
        print(f"{cap:<40} {_fmt(c):>12} {_fmt(v):>10} {_fmt(a):>12}")
    print("-" * 76)

    print(f"\nResults directory: {os.path.abspath(os.path.dirname(scored_path))}")


def _fmt(val) -> str:
    return f"{val:.4f}" if val is not None else "N/A"


# ── Score-only mode ─────────────────────────────────────────────────

def run_score_only(args):
    """Recompute scores from existing judge_raw.jsonl."""
    judge_raw_path = args.judge_raw
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load judge results
    judge_records = []
    with open(judge_raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            judge_records.append(json.loads(line))

    # Extract model name from directory
    model_name = os.path.basename(output_dir.rstrip("/")).replace("_AtomicEval", "")

    _finalize_outputs(judge_records, {}, output_dir, model_name)


# ── Cross-model summary ────────────────────────────────────────────

def run_summary(args):
    """Generate cross-model comparison CSV."""
    import glob

    # Find all dataset_summary.json files
    summaries = []
    for d in args.results_dirs:
        pattern = os.path.join(d, "dataset_summary.json")
        for fpath in glob.glob(pattern):
            with open(fpath, "r", encoding="utf-8") as f:
                summaries.append(json.load(f))

    if not summaries:
        # Try glob expansion
        for d in args.results_dirs:
            if os.path.isfile(os.path.join(d, "dataset_summary.json")):
                with open(os.path.join(d, "dataset_summary.json"), "r") as f:
                    summaries.append(json.load(f))

    if not summaries:
        print("No dataset_summary.json files found")
        return

    # Write cross-model CSV
    output_path = args.output
    header = ["model_name", "num_scored", "caption_score",
              "consistency", "coverage", "anti_hallucination"]
    for cap in CAPABILITIES:
        header.extend([f"{cap}_cov", f"{cap}_cons", f"{cap}_ah"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for s in sorted(summaries, key=lambda x: x.get("model_name", "")):
            row = {
                "model_name": s.get("model_name", ""),
                "num_scored": s.get("num_scored_samples", 0),
                "caption_score": s.get("mean_caption_score", 0),
                "consistency": s.get("mean_consistency_score", 0),
                "coverage": s.get("mean_weighted_coverage", 0),
                "anti_hallucination": s.get("mean_weighted_anti_hallucination", 0),
            }
            for cap in CAPABILITIES:
                cd = s.get("per_capability", {}).get(cap, {})
                row[f"{cap}_cov"] = cd.get("mean_coverage", "")
                row[f"{cap}_cons"] = cd.get("mean_consistency", "")
                row[f"{cap}_ah"] = cd.get("mean_anti_hallucination", "")
            writer.writerow(row)

    print(f"Cross-model summary saved to: {output_path}")
    print(f"Contains {len(summaries)} models")


# ── Direct Alignment (Method B) ─────────────────────────────────────

def run_direct_alignment(args):
    """Run direct alignment: GT atomic facts + raw caption -> GPT judges directly."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts", "direct_alignment.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"Loading GT atomic facts: {args.gt_facts}")
    gt_map = load_gt_atomic_facts(args.gt_facts)
    gt_ok = {k: v for k, v in gt_map.items() if v.get("status") == "ok"}
    print(f"  Valid GT: {len(gt_ok)}")

    print(f"Loading CaptionResult: {args.caption}")
    caption_map = load_caption_results(args.caption)
    print(f"  Caption samples: {len(caption_map)}")

    caption_basename = os.path.basename(args.caption)
    if "_CaptionResult" in caption_basename:
        model_name = caption_basename.split("_CaptionResult")[0]
    else:
        model_name = os.path.splitext(caption_basename)[0]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please provide an API Key (--api-key or OPENAI_API_KEY env var)")
    client = create_client(api_key, args.base_url)

    common_ids = sorted(set(gt_ok.keys()) & set(caption_map.keys()))
    print(f"  Overlapping samples: {len(common_ids)}")

    # Checkpoint: load existing results
    raw_path = os.path.join(output_dir, "direct_align_raw.jsonl")
    processed_ids = set()
    existing_results = []
    error_ids = set()
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id", "")
                if rec.get("status") == "error":
                    error_ids.add(sid)
                else:
                    existing_results.append(rec)
                    processed_ids.add(sid)
        if error_ids:
            print(f"  Succeeded: {len(processed_ids)}, pending retry: {len(error_ids)}")
        else:
            print(f"  Processed: {len(processed_ids)} samples, resuming from checkpoint")

    remaining = [sid for sid in common_ids if sid not in processed_ids]
    if not remaining:
        print("All samples processed! Recomputing scores...")
        _finalize_direct_alignment(existing_results, output_dir, model_name)
        return

    print(f"\nStarting Direct Alignment: {len(remaining)} samples "
          f"(model={args.model}, workers={args.num_workers})")
    print("=" * 60)

    results_lock = threading.Lock()
    save_counter = 0
    success_count = 0
    error_count = 0

    def process_one(sid):
        gt_result = gt_ok[sid]
        caption_steps = caption_map[sid]
        payload = build_direct_alignment_payload(sid, gt_result, caption_steps)

        response = call_api(
            client, system_prompt, payload,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sample_id=sid,
            enable_thinking=True,
        )

        result = parse_direct_alignment(response, sid)

        if result is None and response.strip():
            repair_payload = payload.copy()
            repair_payload["_repair_note"] = (
                "Your previous response could not be parsed. "
                "Return ONLY a valid JSON object following the required output structure."
            )
            response = call_api(
                client, system_prompt, repair_payload,
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sample_id=f"{sid}_retry",
                enable_thinking=True,
            )
            result = parse_direct_alignment(response, sid)

        return sid, result, response

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, sid): sid for sid in remaining}
        pbar = tqdm(total=len(remaining), desc="DirectAlign", unit="sample")

        for future in as_completed(futures):
            sid, result, raw_response = future.result()
            with results_lock:
                if result is not None:
                    rec = result.model_dump()
                    rec["status"] = "ok"
                    existing_results.append(rec)
                    success_count += 1
                else:
                    existing_results.append({
                        "sample_id": sid,
                        "status": "error",
                        "error": "Parse/validation failed",
                        "raw_response_preview": raw_response[:500] if raw_response else "",
                    })
                    error_count += 1

                save_counter += 1
                if save_counter % SAVE_EVERY_N == 0 or save_counter == len(remaining):
                    _save_jsonl(raw_path, existing_results)
            pbar.update(1)
            pbar.set_postfix(ok=success_count, err=error_count)

        pbar.close()

    _save_jsonl(raw_path, existing_results)
    print(f"\n{'=' * 60}")
    print(f"Direct Alignment complete! Succeeded: {success_count}, failed: {error_count}")

    _finalize_direct_alignment(existing_results, output_dir, model_name)


def _finalize_direct_alignment(records, output_dir, model_name):
    """Convert direct alignment results to scored output."""
    from .models import DirectAlignmentResult

    judge_records = []
    for rec in records:
        sid = rec.get("sample_id", "")
        if rec.get("status") == "error":
            judge_records.append({
                "sample_id": sid,
                "raw_response": {"status": "error", "error": rec.get("error", "unknown")},
            })
            continue
        try:
            da_result = DirectAlignmentResult.model_validate(rec)
            alignment = da_result.to_alignment_result()
            judge_records.append({
                "sample_id": sid,
                "raw_response": alignment.model_dump(),
            })
        except Exception as e:
            judge_records.append({
                "sample_id": sid,
                "raw_response": {"status": "error", "error": str(e)},
            })

    _finalize_outputs(judge_records, {}, output_dir, model_name)


# ── Utilities ───────────────────────────────────────────────────────

def _save_jsonl(path: str, records: list[dict]):
    """Write list of records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_summary_csv(summary: dict, csv_path: str):
    """Write flat summary CSV."""
    header = ["model_name", "num_scored_samples", "num_errors",
              "mean_caption_score", "mean_consistency_score",
              "mean_weighted_coverage", "mean_weighted_anti_hallucination"]
    for cap in CAPABILITIES:
        header.extend([f"{cap}_consistency", f"{cap}_coverage",
                       f"{cap}_anti_hallucination",
                       f"{cap}_G", f"{cap}_C", f"{cap}_M",
                       f"{cap}_P", f"{cap}_X", f"{cap}_O", f"{cap}_H"])

    row = {
        "model_name": summary.get("model_name", ""),
        "num_scored_samples": summary.get("num_scored_samples", 0),
        "num_errors": summary.get("num_errors", 0),
        "mean_caption_score": summary.get("mean_caption_score", 0),
        "mean_consistency_score": summary.get("mean_consistency_score", 0),
        "mean_weighted_coverage": summary.get("mean_weighted_coverage", 0),
        "mean_weighted_anti_hallucination": summary.get("mean_weighted_anti_hallucination", 0),
    }
    for cap in CAPABILITIES:
        cd = summary.get("per_capability", {}).get(cap, {})
        row[f"{cap}_consistency"] = cd.get("mean_consistency", "")
        row[f"{cap}_coverage"] = cd.get("mean_coverage", "")
        row[f"{cap}_anti_hallucination"] = cd.get("mean_anti_hallucination", "")
        row[f"{cap}_G"] = cd.get("total_G", 0)
        row[f"{cap}_C"] = cd.get("total_C", 0)
        row[f"{cap}_M"] = cd.get("total_M", 0)
        row[f"{cap}_P"] = cd.get("total_P", 0)
        row[f"{cap}_X"] = cd.get("total_X", 0)
        row[f"{cap}_O"] = cd.get("total_O", 0)
        row[f"{cap}_H"] = cd.get("total_H", 0)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)
