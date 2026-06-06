"""
Pipeline stage functions with a uniform interface.

Every stage function has the same signature:

    def run_<stage>(
        client,
        video_path: str,
        instruction: str,
        prev_results: Dict[str, StageResult],
        *,
        model, target_fps, max_frames, batch_size,
        sample_ratio,
        system_prompt, user_prompt_template,
    ) -> StageResult

The runner iterates over stages defined by the DatasetConfig and calls the
corresponding function from STAGE_REGISTRY.
"""

import json
from typing import Dict

from config import (
    DEFAULT_MODEL,
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_REFINEMENT_FPS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_BATCH_SIZE,
    MIN_API_FRAMES,
)
from prompts import (
    ACTION_VOCABULARY,
    ACTION_FINE_GRAINED_GUIDANCE,
    FEW_SHOT_EXAMPLES,
    ANALYSIS_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    REFINEMENT_PROMPT_TEMPLATE,
    DETAIL_REFINEMENT_SYSTEM_PROMPT,
    DETAIL_REFINEMENT_PROMPT_TEMPLATE,
    GALAXEA_STEP_SYSTEM_PROMPT,
    GALAXEA_STEP_PROMPT_TEMPLATE,
    DEDUP_STEPS_PROMPT,
)
from data_types import StageResult
from video_utils import (
    load_video_as_image_parts,
    open_video_reader,
    read_and_encode_frames,
    compute_sample_indices,
    compute_sample_indices_by_ratio,
)
from api_client import call_vision_api, extract_json_from_response


# =============================================================================
# Helper: merge consecutive identical steps
# =============================================================================

def _merge_consecutive_same_steps(steps_raw: list) -> list:
    """
    Merge consecutive steps that have identical desc and no frame gap.

    Consecutive steps like:
      {i:0, desc:"chop@chop", start:0, end:10}
      {i:1, desc:"chop@chop", start:11, end:20}
    become:
      {i:0, desc:"chop@chop", start:0, end:20}

    Re-indexes i from 0 after merging.
    """
    if not steps_raw:
        return steps_raw
    merged = [dict(steps_raw[0])]
    for step in steps_raw[1:]:
        last = merged[-1]
        same_desc = step.get("desc", "").strip() == last.get("desc", "").strip()
        if same_desc:
            last["end"] = step.get("end", last["end"])
        else:
            merged.append(dict(step))
    for i, s in enumerate(merged):
        s["i"] = i
    return merged


# =============================================================================
# Stage: analysis
# =============================================================================

def run_analysis(
    client,
    video_path: str,
    instruction: str,
    prev_results: Dict[str, StageResult],
    *,
    model: str = DEFAULT_MODEL,
    target_fps: float = DEFAULT_ANALYSIS_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_ratio: float = None,
    system_prompt: str = "",
    user_prompt_template: str = "",
    video_reader_cache: dict = None,
    **_kwargs,
) -> StageResult:
    """High-level trajectory analysis: extract action_sequence + main_object."""
    result = StageResult(stage_name="analysis")

    _sys = system_prompt or ANALYSIS_SYSTEM_PROMPT
    _user_tpl = user_prompt_template or ANALYSIS_PROMPT_TEMPLATE

    try:
        image_parts, metadata = load_video_as_image_parts(
            video_path,
            target_fps=target_fps,
            max_frames=max_frames,
            batch_size=batch_size,
            sample_ratio=sample_ratio,
            video_reader_cache=video_reader_cache,
        )

        prompt = _user_tpl.format(
            initial_instruction=instruction,
            action_vocabulary=", ".join(ACTION_VOCABULARY),
        )

        response_text, usage = call_vision_api(
            client, image_parts, _sys, prompt, model=model,
            target_fps=target_fps,
        )
        result.raw_response = response_text
        result.token_usage = usage

        if not response_text:
            result.error = "API_EXHAUSTED: API call failed after max retries"
            return result

        parsed = extract_json_from_response(response_text)
        if parsed:
            result.output = {
                "action_sequence": parsed.get("action_sequence", []),
                "main_object": parsed.get("main_object", ""),
            }
            result.success = bool(result.output["action_sequence"])
        else:
            result.error = "PARSE_ERROR: Failed to parse analysis response as JSON"

    except FileNotFoundError as e:
        result.error = f"FILE_NOT_FOUND: {str(e)}"
    except Exception as e:
        result.error = f"ANALYSIS_ERROR: {str(e)}"

    return result


# =============================================================================
# Stage: refinement
# =============================================================================

def run_refinement(
    client,
    video_path: str,
    instruction: str,
    prev_results: Dict[str, StageResult],
    *,
    model: str = DEFAULT_MODEL,
    target_fps: float = DEFAULT_REFINEMENT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_ratio: float = None,
    system_prompt: str = "",
    user_prompt_template: str = "",
    video_reader_cache: dict = None,
    **_kwargs,
) -> StageResult:
    """Instruction refinement: produce fine_grained_steps + refined_instruction."""
    result = StageResult(stage_name="refinement")

    analysis = prev_results.get("analysis")
    if not analysis or not analysis.success:
        result.error = "Skipped: Analysis stage failed"
        return result

    _sys = system_prompt or REFINEMENT_SYSTEM_PROMPT
    _user_tpl = user_prompt_template or REFINEMENT_PROMPT_TEMPLATE

    try:
        image_parts, metadata = load_video_as_image_parts(
            video_path,
            target_fps=target_fps,
            max_frames=max_frames,
            sample_ratio=sample_ratio,
            batch_size=batch_size,
            video_reader_cache=video_reader_cache,
        )

        prompt = _user_tpl.format(
            initial_instruction=instruction,
            action_sequence=json.dumps(analysis.output.get("action_sequence", [])),
            main_object=analysis.output.get("main_object", ""),
            action_guidance=ACTION_FINE_GRAINED_GUIDANCE,
            FEW_SHOT_EXAMPLES=FEW_SHOT_EXAMPLES,
        )

        response_text, usage = call_vision_api(
            client, image_parts, _sys, prompt, model=model,
            target_fps=target_fps,
        )
        result.raw_response = response_text
        result.token_usage = usage

        if not response_text:
            result.error = "API_EXHAUSTED: API call failed after max retries"
            return result

        parsed = extract_json_from_response(response_text)
        if parsed:
            result.output = {
                "fine_grained_steps": parsed.get("fine_grained_steps", []),
                "refined_instruction": parsed.get("refined_instruction", ""),
            }
            result.success = bool(result.output["refined_instruction"])
        else:
            # Fallback: try to extract numbered steps from plain text
            import re
            steps = re.findall(r'^\s*\d+[\.\)]\s*(.+)$', response_text, re.MULTILINE)
            if steps:
                result.output = {
                    "fine_grained_steps": [s.strip() for s in steps],
                    "refined_instruction": " ".join(s.strip() for s in steps),
                }
                result.success = True
                result.error = "PARSE_WARNING: JSON parse failed, extracted steps from text"
            else:
                result.output = {
                    "refined_instruction": response_text,
                    "fine_grained_steps": [],
                }
                result.success = bool(response_text)
                if not result.success:
                    result.error = "PARSE_ERROR: Failed to parse refinement response"

    except FileNotFoundError as e:
        result.error = f"FILE_NOT_FOUND: {str(e)}"
    except Exception as e:
        result.error = f"REFINEMENT_ERROR: {str(e)}"

    return result


# =============================================================================
# Stage: detail_refinement (wrist-view polish)
# =============================================================================

def run_detail_refinement(
    client,
    video_path: str,
    instruction: str,
    prev_results: Dict[str, StageResult],
    *,
    model: str = DEFAULT_MODEL,
    target_fps: float = DEFAULT_REFINEMENT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_ratio: float = None,
    system_prompt: str = "",
    user_prompt_template: str = "",
    video_reader_cache: dict = None,
    **_kwargs,
) -> StageResult:
    """Polish fine-grained steps using a wrist / first-person camera view."""
    result = StageResult(stage_name="detail_refinement")

    refinement = prev_results.get("refinement")
    if not refinement or not refinement.success:
        result.error = "Skipped: refinement stage failed"
        return result

    _sys = system_prompt or DETAIL_REFINEMENT_SYSTEM_PROMPT
    _user_tpl = user_prompt_template or DETAIL_REFINEMENT_PROMPT_TEMPLATE

    try:
        image_parts, metadata = load_video_as_image_parts(
            video_path,
            target_fps=target_fps,
            max_frames=max_frames,
            sample_ratio=sample_ratio,
            batch_size=batch_size,
            video_reader_cache=video_reader_cache,
        )

        previous_steps = refinement.output.get("fine_grained_steps", [])
        previous_refined = refinement.output.get("refined_instruction", "")

        prompt = _user_tpl.format(
            initial_instruction=instruction,
            main_object=prev_results.get("analysis", StageResult(stage_name="")).output.get("main_object", ""),
            previous_steps=json.dumps(previous_steps, ensure_ascii=False),
            previous_refined_instruction=previous_refined,
        )

        response_text, usage = call_vision_api(
            client, image_parts, _sys, prompt, model=model,
            target_fps=target_fps,
        )
        result.raw_response = response_text
        result.token_usage = usage

        if not response_text:
            result.error = "API_EXHAUSTED: API call failed after max retries"
            return result

        parsed = extract_json_from_response(response_text)
        if parsed:
            result.output = {
                "fine_grained_steps": parsed.get("fine_grained_steps", previous_steps),
                "refined_instruction": parsed.get("refined_instruction", previous_refined),
                "changes_made": parsed.get("changes_made", []),
            }
            result.success = True
        else:
            result.output = {
                "fine_grained_steps": previous_steps,
                "refined_instruction": previous_refined,
                "changes_made": [],
            }
            result.success = True
            result.error = "PARSE_WARNING: Could not parse detail_refinement JSON, kept previous steps"

    except FileNotFoundError as e:
        result.error = f"FILE_NOT_FOUND: {str(e)}"
    except Exception as e:
        result.error = f"DETAIL_REFINEMENT_ERROR: {str(e)}"

    return result


# =============================================================================
# Stage: galaxea_step_refinement (per-step clip refinement)
# =============================================================================

def run_galaxea_step_refinement(
    client,
    video_path: str,
    instruction: str,
    prev_results: Dict[str, StageResult],
    *,
    model: str = DEFAULT_MODEL,
    target_fps: float = DEFAULT_REFINEMENT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_ratio: float = None,
    system_prompt: str = "",
    user_prompt_template: str = "",
    trajectory_dict: dict = None,
    **_kwargs,
) -> StageResult:
    """
    Galaxea-specific: iterate over steps_raw, clip video per step, refine each.

    Opens the video **once** and reuses the reader for all steps to avoid
    redundant file open/close overhead.

    Reads ``steps_raw`` from trajectory_dict.  Each entry has:
      - i: step index
      - desc: "Chinese description@English description"
      - start / end: frame indices (inclusive)

    Produces fine_grained_steps (list) + refined_instruction (paragraph).
    """
    result = StageResult(stage_name="refinement")

    steps_raw = (trajectory_dict or {}).get("steps_raw", [])
    if not steps_raw:
        result.error = "MISSING_STEPS: steps_raw is empty or not found"
        return result

    steps_raw = _merge_consecutive_same_steps(steps_raw)

    _sys = system_prompt or GALAXEA_STEP_SYSTEM_PROMPT
    _user_tpl = user_prompt_template or GALAXEA_STEP_PROMPT_TEMPLATE

    from concurrent.futures import ThreadPoolExecutor, as_completed

    errors: list = []
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    vr = None
    try:
        vr, total_frames, video_fps, use_pyav = open_video_reader(video_path)

        # Phase 1: Prepare all steps (extract frames, build prompts)
        step_tasks = []  # (step_idx, desc_en, image_parts, prompt)
        for step in steps_raw:
            step_idx = step.get("i", len(step_tasks))
            desc_raw = step.get("desc", "")
            desc_en = desc_raw.split("@")[-1].strip() if "@" in desc_raw else desc_raw
            frame_start = step.get("start", 0)
            frame_end = step.get("end", None)

            eff_start = max(0, frame_start)
            eff_end = min(total_frames, frame_end + 1) if frame_end is not None else total_frames
            eff_total = max(0, eff_end - eff_start)

            if sample_ratio is not None:
                indices = compute_sample_indices_by_ratio(eff_total, sample_ratio)
            else:
                indices = compute_sample_indices(eff_total, video_fps, target_fps, max_frames)
            indices = [i + eff_start for i in indices]

            if len(indices) < MIN_API_FRAMES:
                errors.append(f"step {step_idx}: only {len(indices)} frames in range [{eff_start}:{eff_end}), skipped")
                step_tasks.append((step_idx, desc_en, None))
                continue

            try:
                image_parts, _ = read_and_encode_frames(
                    video_path, vr, indices, use_pyav,
                    batch_size=batch_size,
                )
            except Exception as e:
                errors.append(f"step {step_idx}: video load error: {e}")
                step_tasks.append((step_idx, desc_en, None))
                continue

            # Guard against decode failures producing fewer images than the API minimum
            n_images = sum(1 for p in image_parts if isinstance(p, dict) and p.get("type") in ("image_url", "image"))
            if n_images < MIN_API_FRAMES:
                errors.append(f"step {step_idx}: only {n_images} frames encoded (< {MIN_API_FRAMES}), skipped")
                step_tasks.append((step_idx, desc_en, None))
                continue

            step_tasks.append((step_idx, desc_en, image_parts))

        # Phase 2: Call API in PARALLEL using raw descriptions as context
        # Build static context from ALL raw step descriptions (no dependency chain)
        all_raw_lines = "\n".join(
            f"  Step {t[0]}: {t[1]}" for t in step_tasks
        )
        all_steps_context = f"All steps in this task (raw descriptions):\n{all_raw_lines}\n\n"

        # Prepare prompts for all steps
        api_tasks = []  # [(task_list_idx, step_idx, desc_en, image_parts, prompt)]
        for list_idx, (step_idx, desc_en, image_parts) in enumerate(step_tasks):
            if image_parts is None:
                continue

            # Context: show other steps so model knows what NOT to repeat
            other_lines = "\n".join(
                f"  Step {t[0]}: {t[1]}" for j, t in enumerate(step_tasks) if j != list_idx
            )
            previous_steps_context = f"Other steps in this task (do NOT repeat their content):\n{other_lines}\n\n" if other_lines else ""

            prompt = _user_tpl.format(
                initial_instruction=instruction,
                step_index=step_idx,
                step_description=desc_en,
                action_guidance=ACTION_FINE_GRAINED_GUIDANCE,
                previous_steps_context=previous_steps_context,
            )
            api_tasks.append((list_idx, step_idx, desc_en, image_parts, prompt))

        # Initialize results arrays
        refined_steps = [t[1] for t in step_tasks]  # default to raw desc
        raw_responses = [""] * len(step_tasks)

        # Parallel API calls for all steps
        def _call_step(task_tuple):
            list_idx, step_idx, desc_en, image_parts, prompt = task_tuple
            response_text, usage = call_vision_api(
                client, image_parts, _sys, prompt, model=model,
                target_fps=target_fps,
            )
            return list_idx, step_idx, desc_en, response_text, usage

        if api_tasks:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(len(api_tasks), 8)) as step_executor:
                futures = {step_executor.submit(_call_step, t): t for t in api_tasks}
                for future in as_completed(futures):
                    list_idx, step_idx, desc_en, response_text, usage = future.result()
                    raw_responses[list_idx] = response_text or ""
                    for k in total_usage:
                        total_usage[k] += usage.get(k, 0)

                    if not response_text:
                        errors.append(f"step {step_idx}: API returned empty response")
                        continue

                    parsed = extract_json_from_response(response_text)
                    if parsed and parsed.get("refined_step"):
                        refined_steps[list_idx] = parsed["refined_step"]
                    else:
                        errors.append(f"step {step_idx}: JSON parse failed, kept original")

        # Phase 3: Dedup pass — remove redundancy across steps
        if len(refined_steps) > 1:
            original_steps_str = "\n".join(
                f"  Step {i}: {steps_raw[i].get('desc', '').split('@')[-1].strip()}"
                for i in range(len(steps_raw))
            )
            refined_steps_str = "\n".join(
                f"  Step {i}: {s}" for i, s in enumerate(refined_steps)
            )
            dedup_prompt = DEDUP_STEPS_PROMPT.format(
                initial_instruction=instruction,
                original_steps=original_steps_str,
                refined_steps=refined_steps_str,
            )
            try:
                dedup_resp, dedup_usage = call_vision_api(
                    client, [], _sys, dedup_prompt, model=model,
                    target_fps=target_fps,
                )
                for k in total_usage:
                    total_usage[k] += dedup_usage.get(k, 0)
                if dedup_resp:
                    dedup_parsed = extract_json_from_response(dedup_resp)
                    dedup_steps = (dedup_parsed or {}).get("deduped_steps", [])
                    if dedup_steps and len(dedup_steps) == len(refined_steps):
                        refined_steps = dedup_steps
                    elif dedup_steps:
                        errors.append(
                            f"dedup: returned {len(dedup_steps)} steps, "
                            f"expected {len(refined_steps)}, skipped"
                        )
                raw_responses.append(dedup_resp or "")
            except Exception as e:
                errors.append(f"dedup: failed ({e}), keeping original refined steps")

        refined_instruction = " ".join(
            f"({i + 1}) {s}" for i, s in enumerate(refined_steps)
        )

        result.output = {
            "fine_grained_steps": refined_steps,
            "refined_instruction": refined_instruction,
        }
        result.raw_response = "\n---\n".join(raw_responses)
        result.token_usage = total_usage
        result.success = len(refined_steps) == len(steps_raw)

        if errors:
            result.error = f"PARTIAL_ERRORS: {'; '.join(errors)}"

    except FileNotFoundError as e:
        result.error = f"FILE_NOT_FOUND: {str(e)}"
    except Exception as e:
        result.error = f"GALAXEA_STEP_ERROR: {str(e)}"
    finally:
        if vr is not None:
            del vr

    return result


# =============================================================================
# Registry: maps fn_name (used in StageDefinition) → callable
# =============================================================================

STAGE_REGISTRY = {
    "analysis": run_analysis,
    "refinement": run_refinement,
    "detail_refinement": run_detail_refinement,
    "galaxea_step_refinement": run_galaxea_step_refinement,
}
