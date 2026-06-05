#!/usr/bin/env python3
"""
LLM-as-a-Judge: Use GPT to evaluate VLM annotation results via QA

Usage:
    python run_judge.py \
        --caption ../Eval_Result/xxx_CaptionResult.jsonl \
        --qa ../GenerateQA/QA_Results_Test.json \
        --output ../Eval_Result/xxx_JudgeResult.json

    # Use official OpenAI API
    python run_judge.py \
        --caption ../Eval_Result/xxx_CaptionResult.jsonl \
        --qa ../GenerateQA/QA_Results_Test.json \
        --output ../Eval_Result/xxx_JudgeResult.json \
        --use-openai

    # Specify model
    python run_judge.py \
        --caption ../Eval_Result/xxx_CaptionResult.jsonl \
        --qa ../GenerateQA/QA_Results_Test.json \
        --output ../Eval_Result/xxx_JudgeResult.json \
        --model qwen-plus

      python CaptionEval/AtomicEval/run_judge.py \
      --caption Eval_Result/openai_gpt-5_4-2026-03-05_20260319_130854_CaptionResult.jsonl \
      --qa GenerateQA/QA_Results_Test.json \
      --output Eval_Result/openai_gpt-5_4-2026-03-05_20260319_223900_JudgeResult.json

    
     python CaptionEval/AtomicEval/run_judge.py \
      --caption Eval_Result/openai_gpt-5_4-2026-03-05_CaptionResult.jsonl \
      --qa GenerateQA/QA_Results394.json \
      --prompt CaptionEval/AtomicEval/Prompt.txt \
      --output Eval_Result/openai_gpt-5_4-2026-03-05_NewJudgeResult.json \
      --num-workers 32


for caption in Eval_Result/*_CaptionResult.jsonl; do
    model=$(basename "$caption" _CaptionResult.jsonl)
    output="Eval_Result/${model}_NewJudgeResult.json"
    echo "=== Judging: $model ==="
    python CaptionEval/AtomicEval/run_judge.py \
      --caption "$caption" \
      --qa GenerateQA/QA_Results394.json \
      --prompt CaptionEval/AtomicEval/Prompt.txt \
      --output "$output" \
      --num-workers 32
done
"""

import json
import os
import csv
import argparse
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# 13 capability dimensions (fixed order)
CAPABILITIES = [
    'action_primitive', 'actor_identity', 'object_recognition',
    'object_disambiguation', 'contact_region', 'source_state_or_location',
    'trajectory_and_orientation', 'placement_specification',
    'interaction_with_other_objects', 'success_failure_retry',
    'gripper_state', 'temporal_order_and_step_boundary', 'body_motion'
]

# APIKey="sk-e8b188af33504bc0ba158bebf2c9a110"
DEFAULT_MODEL = "openai.gpt-5.4-2026-03-05"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEMPERATURE = 0.1  # Low temperature for more consistent judgments


def _is_gpt_reasoning_model(model: str) -> bool:
    """Detect GPT reasoning models (o1, o3, o4, gpt-5, etc.)"""
    m = model.lower()
    return (m.startswith("o1") or m.startswith("o3") or m.startswith("o4") or
            "gpt-5" in m or "gpt5" in m)


def load_prompt(prompt_file):
    """Load prompt.txt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_caption_results(caption_file):
    """Load CaptionResult.jsonl and return a {sample_id: record} dict"""
    caption_map = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("sample_id", "")
            if sid:
                caption_map[sid] = record
    return caption_map


def load_qa_results(qa_file):
    """Load QA_Results_Test.json"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_judge_input(caption_record, qa_record):
    """
    Build a single sample input for the Judge LLM.
    Only keep necessary fields to save tokens.
    """
    sample_id = qa_record.get("sample_id", "unknown")

    # Extract fineGrainedSteps from CaptionResult
    fine_grained_steps = caption_record.get("fineGrainedSteps", [])

    # Extract question list from QA
    qas = qa_record.get("qas", [])

    return {
        "sample_id": sample_id,
        "fineGrainedSteps": fine_grained_steps,
        "qas": [
            {
                "question_id": qa.get("question_id", ""),
                "capability": qa.get("capability", ""),
                "error_type": qa.get("error_type", ""),
                "answer_type": qa.get("answer_type", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "reference_text": qa.get("reference_text", ""),
                "difference_summary": qa.get("difference_summary", ""),
            }
            for qa in qas
        ]
    }


def call_judge(client, prompt, judge_input, model=DEFAULT_MODEL,
               temperature=DEFAULT_TEMPERATURE, max_retries=100,
               use_openai=False, sample_id="unknown"):
    """Call the LLM Judge API (with retry logic)"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps([judge_input], ensure_ascii=False, indent=2)}
    ]

    for attempt in range(max_retries):
        try:
            if use_openai:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
            elif 'gemini' in model.lower():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "dashscope_extend_params": {"provider": "b"},
                        "stream": False,
                    }
                )
            elif _is_gpt_reasoning_model(model):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "reasoning_effort": "high",
                        "max_completion_tokens": 16384,
                        "stream": False,
                    }
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "max_completion_tokens": 16384,
                        "stream": False,
                    }
                )

            time.sleep(0.5)
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[{sample_id}] Failed after {max_retries} attempts: {e}")
                return ""
            if attempt < 3:
                print(f"\n[{sample_id}] API call failed (attempt {attempt + 1}): {e}")
            time.sleep(0.5)


def parse_judge_response(response_text, sample_id, model_name):
    """Parse the Judge's JSON response"""
    if not response_text:
        return {
            "sample_id": sample_id,
            "model": model_name,
            "qa_results": [],
            "total_correct": 0,
            "total_questions": 0,
            "accuracy": 0,
            "error": "Empty API response"
        }

    try:
        parsed = json.loads(response_text)

        # Handle possible wrapper formats
        if isinstance(parsed, dict):
            # Some models return {"results": [...]} format
            if "results" in parsed:
                parsed = parsed["results"]
            else:
                parsed = [parsed]

        if isinstance(parsed, list) and len(parsed) > 0:
            result = parsed[0]
            # Ensure the model field is present
            result["model"] = model_name
            # Ensure sample_id is consistent
            result["sample_id"] = sample_id
            return result

    except json.JSONDecodeError as e:
        pass

    return {
        "sample_id": sample_id,
        "model": model_name,
        "qa_results": [],
        "total_correct": 0,
        "total_questions": 0,
        "accuracy": 0,
        "error": f"JSON parse error",
        "raw_response": response_text[:500]
    }


def _update_capability_csv(judge_output_path, model_name, cap_correct, cap_total,
                           total_correct, total_questions):
    """Update capability_accuracy.csv; rows with the same model_name are overwritten"""
    # CSV path: capability_accuracy.csv in the same directory as JudgeResult
    csv_dir = os.path.dirname(os.path.abspath(judge_output_path))
    csv_path = os.path.join(csv_dir, "capability_accuracy.csv")

    header = ['model_name', 'overall'] + CAPABILITIES

    # Read existing data
    existing_rows = {}  # model_name -> row dict
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows[row['model_name']] = row
        except Exception:
            pass

    # Build row for current model
    new_row = {'model_name': model_name}
    for cap in CAPABILITIES:
        c, t = cap_correct[cap], cap_total[cap]
        new_row[cap] = round(c / t * 100, 2) if t else 0
    new_row['overall'] = round(total_correct / total_questions * 100, 2) if total_questions else 0

    # Update or insert
    existing_rows[model_name] = new_row

    # Write back CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for mn in sorted(existing_rows.keys()):
            writer.writerow(existing_rows[mn])

    print(f"\nCSV updated: {csv_path}")


def _finalize_and_update_csv(qa_data, all_results, output_path, model_name):
    """Compute per-capability accuracy and update CSV"""
    # Build question_id -> capability mapping
    qa_id_to_cap = {}
    for qa_sample in qa_data:
        for qa in qa_sample.get("qas", []):
            qa_id_to_cap[qa.get("question_id", "")] = qa.get("capability", "unknown")

    # Compute per-capability accuracy (compatible with both old and new Judge output formats)
    # Old format: qa_results[].correct = true/false
    # New format: qa_results[].result = "correct" | "incorrect" | "skip"
    cap_correct = defaultdict(int)
    cap_total = defaultdict(int)  # Only count judged questions (excluding skip)
    cap_skipped = defaultdict(int)
    total_correct = 0
    total_judged = 0
    total_skipped = 0
    total_questions = 0

    for rec in all_results:
        total_correct += rec.get("total_correct", 0)
        total_questions += rec.get("total_questions", 0)
        total_judged += rec.get("total_judged", rec.get("total_questions", 0))
        total_skipped += rec.get("total_skipped", 0)
        for qr in rec.get("qa_results", []):
            cap = qa_id_to_cap.get(qr.get("question_id", ""), "unknown")
            # Compatible with both old and new formats
            result_str = qr.get("result", "")
            correct_bool = qr.get("correct", False)
            if result_str == "skip":
                cap_skipped[cap] += 1
                continue
            # Count toward cap_total (judged questions)
            cap_total[cap] += 1
            if result_str == "correct" or (not result_str and correct_bool):
                cap_correct[cap] += 1

    if total_judged > 0 or total_questions > 0:
        # Prefer total_judged as denominator (excluding skip)
        denom = total_judged if total_judged > 0 else total_questions
        overall_acc = total_correct / denom if denom > 0 else 0
        print(f"\nOverall accuracy: {total_correct}/{denom} = {overall_acc:.4f}")
        if total_skipped > 0:
            print(f"({total_questions} total questions, {total_skipped} skipped, {total_judged} actually judged)")

        print(f"\n{'Capability':<45} {'Correct':>8} {'Judged':>8} {'Skipped':>8} {'Accuracy':>10}")
        print("-" * 85)
        for cap in CAPABILITIES:
            c, t, s = cap_correct[cap], cap_total[cap], cap_skipped[cap]
            acc = c / t * 100 if t else 0
            print(f"{cap:<45} {c:>8} {t:>8} {s:>8} {acc:>9.1f}%")
        print("-" * 85)
        total_cap_skipped = sum(cap_skipped[cap] for cap in CAPABILITIES)
        print(f"{'TOTAL':<45} {total_correct:>8} {total_judged:>8} {total_cap_skipped:>8} {overall_acc*100:>9.1f}%")

    # Update capability_accuracy.csv (denominator is the number of judged questions)
    _update_capability_csv(output_path, model_name, cap_correct, cap_total,
                           total_correct, total_judged if total_judged > 0 else total_questions)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge: QA evaluation of VLM annotation results")
    parser.add_argument('--caption', '-c', required=True,
                       help='Path to CaptionResult.jsonl file')
    parser.add_argument('--qa', '-q', required=True,
                       help='Path to QA_Results_Test.json file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output JudgeResult.json file path (default: auto-generated from --caption filename)')
    parser.add_argument('--prompt', '-p', default=None,
                       help='Path to prompt.txt file (default: prompt.txt in the same directory as this script)')
    parser.add_argument('--api-key', help='API Key (optional, defaults to OPENAI_API_KEY env var)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'Model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Temperature parameter (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='API Base URL')
    parser.add_argument('--use-openai', action='store_true', help='Use the official OpenAI API')
    parser.add_argument('--max-retries', type=int, default=100, help='Maximum number of retries (default: 100)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel API call threads (default: 8)')

    args = parser.parse_args()

    # Auto-generate output path: replace CaptionResult.jsonl with JudgeResult.json
    if args.output is None:
        base = args.caption
        if "CaptionResult" in base:
            args.output = base.replace("CaptionResult.jsonl", "JudgeResult.json")
        else:
            # fallback: append _JudgeResult.json in the same directory
            root, _ = os.path.splitext(base)
            args.output = root + "_JudgeResult.json"
        print(f"Auto-generated output path: {args.output}")

    # API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please provide an API Key (via --api-key or the OPENAI_API_KEY env var)")

    # Client
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    elif not args.use_openai:
        client_kwargs["base_url"] = DEFAULT_BASE_URL
    client = OpenAI(**client_kwargs)

    # Prompt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = args.prompt or os.path.join(script_dir, "prompt.txt")
    print(f"Loading Prompt: {prompt_path}")
    prompt = load_prompt(prompt_path)

    # Load data
    print(f"Loading CaptionResult: {args.caption}")
    caption_map = load_caption_results(args.caption)
    print(f"  -> {len(caption_map)} annotation results")

    print(f"Loading QA: {args.qa}")
    qa_data = load_qa_results(args.qa)
    print(f"  -> {len(qa_data)} QA samples")

    # Extract caption model name from caption filename (for CSV records)
    caption_basename = os.path.basename(args.caption)
    if "_CaptionResult" in caption_basename:
        model_name = caption_basename.split("_CaptionResult")[0]
    else:
        model_name = os.path.splitext(caption_basename)[0]

    # Resume from checkpoint
    processed_ids = set()
    all_results = []
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if isinstance(all_results, list):
                    processed_ids = {r.get("sample_id", "") for r in all_results if r.get("sample_id")}
                    print(f"{len(processed_ids)} samples already processed, resuming from checkpoint")
        except Exception:
            pass

    # Filter out already processed samples
    remaining = []
    skipped_no_caption = 0
    for qa_sample in qa_data:
        sid = qa_sample.get("sample_id", "")
        if sid in processed_ids:
            continue
        if sid not in caption_map:
            skipped_no_caption += 1
            continue
        # Skip samples with no QA or abnormal status
        if qa_sample.get("status") not in ("ok",) or not qa_sample.get("qas"):
            # For samples without QA, write empty results directly
            all_results.append({
                "sample_id": sid,
                "model": model_name,
                "qa_results": [],
                "total_correct": 0,
                "total_questions": 0,
                "accuracy": 0,
            })
            continue
        remaining.append(qa_sample)

    if skipped_no_caption:
        print(f"  -> Skipped {skipped_no_caption} samples (no matching record in CaptionResult)")

    if not remaining:
        print("All samples have been processed!")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        # Even if all samples are processed, still need to update CSV
        _finalize_and_update_csv(qa_data, all_results, args.output, model_name)
        return

    print(f"\nStarting Judge evaluation: {len(remaining)} samples (model={args.model}, temp={args.temperature}, workers={args.num_workers})")
    print("=" * 60)

    success_count = 0
    error_count = 0
    results_lock = threading.Lock()
    save_counter = 0

    def process_sample(qa_sample):
        """Process a single sample and return the result"""
        sid = qa_sample["sample_id"]
        caption_record = caption_map[sid]

        # Build Judge input
        judge_input = build_judge_input(caption_record, qa_sample)

        # Call LLM Judge
        response_text = call_judge(
            client, prompt, judge_input,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            use_openai=args.use_openai,
            sample_id=sid
        )

        # Parse result
        result = parse_judge_response(response_text, sid, model_name)
        is_success = "error" not in result
        return result, is_success

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in remaining}
        pbar = tqdm(total=len(remaining), desc="Judging", unit="sample")

        for future in as_completed(futures):
            result, is_success = future.result()
            with results_lock:
                if is_success:
                    success_count += 1
                else:
                    error_count += 1
                all_results.append(result)
                save_counter += 1
                # Save every 5 samples
                if save_counter % 5 == 0:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
            pbar.update(1)

        pbar.close()

    # Final save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Statistics
    print(f"\n{'=' * 60}")
    print(f"Judge evaluation completed!")
    print(f"Total QA samples: {len(qa_data)}")
    print(f"Processed this run: {len(remaining)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {error_count}")

    # Compute statistics and update CSV
    _finalize_and_update_csv(qa_data, all_results, args.output, model_name)

    print(f"\nResults saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
