#!/usr/bin/env python3
"""
LLM-as-a-Judge: 使用 GPT 对 VLM 打标结果进行 QA 评测

用法:
    python run_judge.py \
        --caption ../Eval_Result/xxx_CaptionResult.jsonl \
        --qa ../GenerateQA/QA_Results_Test.json \
        --output ../Eval_Result/xxx_JudgeResult.json

    # 使用官方 OpenAI API
    python run_judge.py \
        --caption ../Eval_Result/xxx_CaptionResult.jsonl \
        --qa ../GenerateQA/QA_Results_Test.json \
        --output ../Eval_Result/xxx_JudgeResult.json \
        --use-openai

    # 指定模型
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

# 13 个能力维度（固定顺序）
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
DEFAULT_TEMPERATURE = 0.1  # 低温度以获得更一致的判断


def _is_gpt_reasoning_model(model: str) -> bool:
    """检测 GPT reasoning 模型（o1, o3, o4, gpt-5 等）"""
    m = model.lower()
    return (m.startswith("o1") or m.startswith("o3") or m.startswith("o4") or
            "gpt-5" in m or "gpt5" in m)


def load_prompt(prompt_file):
    """加载 prompt.txt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_caption_results(caption_file):
    """加载 CaptionResult.jsonl，返回 {sample_id: record} 字典"""
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
    """加载 QA_Results_Test.json"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_judge_input(caption_record, qa_record):
    """
    构建发给 Judge LLM 的单个样本输入
    只保留必要字段以节省 token
    """
    sample_id = qa_record.get("sample_id", "unknown")

    # 从 CaptionResult 提取 fineGrainedSteps
    fine_grained_steps = caption_record.get("fineGrainedSteps", [])

    # 从 QA 提取问题列表
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
    """调用 LLM Judge API（带重试逻辑）"""
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
    """解析 Judge 的 JSON 响应"""
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

        # 处理可能的包装格式
        if isinstance(parsed, dict):
            # 有些模型会返回 {"results": [...]} 格式
            if "results" in parsed:
                parsed = parsed["results"]
            else:
                parsed = [parsed]

        if isinstance(parsed, list) and len(parsed) > 0:
            result = parsed[0]
            # 确保包含 model 字段
            result["model"] = model_name
            # 确保 sample_id 一致
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
    """更新 capability_accuracy.csv，同一 model_name 的行会被覆盖"""
    # CSV 路径：与 JudgeResult 同目录下的 capability_accuracy.csv
    csv_dir = os.path.dirname(os.path.abspath(judge_output_path))
    csv_path = os.path.join(csv_dir, "capability_accuracy.csv")

    header = ['model_name', 'overall'] + CAPABILITIES

    # 读取已有数据
    existing_rows = {}  # model_name -> row dict
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows[row['model_name']] = row
        except Exception:
            pass

    # 构建当前模型的行
    new_row = {'model_name': model_name}
    for cap in CAPABILITIES:
        c, t = cap_correct[cap], cap_total[cap]
        new_row[cap] = round(c / t * 100, 2) if t else 0
    new_row['overall'] = round(total_correct / total_questions * 100, 2) if total_questions else 0

    # 更新或插入
    existing_rows[model_name] = new_row

    # 写回 CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for mn in sorted(existing_rows.keys()):
            writer.writerow(existing_rows[mn])

    print(f"\n已更新 CSV: {csv_path}")


def _finalize_and_update_csv(qa_data, all_results, output_path, model_name):
    """统计各 capability 准确率并更新 CSV"""
    # 构建 question_id -> capability 映射
    qa_id_to_cap = {}
    for qa_sample in qa_data:
        for qa in qa_sample.get("qas", []):
            qa_id_to_cap[qa.get("question_id", "")] = qa.get("capability", "unknown")

    # 按 capability 统计准确率（兼容新旧两种 Judge 输出格式）
    # 旧格式: qa_results[].correct = true/false
    # 新格式: qa_results[].result = "correct" | "incorrect" | "skip"
    cap_correct = defaultdict(int)
    cap_total = defaultdict(int)  # 只计入被 judge 的问题（不含 skip）
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
            # 兼容新旧格式
            result_str = qr.get("result", "")
            correct_bool = qr.get("correct", False)
            if result_str == "skip":
                cap_skipped[cap] += 1
                continue
            # 计入 cap_total（被 judge 的问题）
            cap_total[cap] += 1
            if result_str == "correct" or (not result_str and correct_bool):
                cap_correct[cap] += 1

    if total_judged > 0 or total_questions > 0:
        # 优先用 total_judged 作为分母（排除 skip）
        denom = total_judged if total_judged > 0 else total_questions
        overall_acc = total_correct / denom if denom > 0 else 0
        print(f"\n整体准确率: {total_correct}/{denom} = {overall_acc:.4f}")
        if total_skipped > 0:
            print(f"（共 {total_questions} 题，其中 {total_skipped} 题被 skip，{total_judged} 题被实际评判）")

        print(f"\n{'Capability':<45} {'Correct':>8} {'Judged':>8} {'Skipped':>8} {'Accuracy':>10}")
        print("-" * 85)
        for cap in CAPABILITIES:
            c, t, s = cap_correct[cap], cap_total[cap], cap_skipped[cap]
            acc = c / t * 100 if t else 0
            print(f"{cap:<45} {c:>8} {t:>8} {s:>8} {acc:>9.1f}%")
        print("-" * 85)
        total_cap_skipped = sum(cap_skipped[cap] for cap in CAPABILITIES)
        print(f"{'TOTAL':<45} {total_correct:>8} {total_judged:>8} {total_cap_skipped:>8} {overall_acc*100:>9.1f}%")

    # 更新 capability_accuracy.csv（分母为被 judge 的问题数）
    _update_capability_csv(output_path, model_name, cap_correct, cap_total,
                           total_correct, total_judged if total_judged > 0 else total_questions)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge: QA 评测 VLM 打标结果")
    parser.add_argument('--caption', '-c', required=True,
                       help='CaptionResult.jsonl 文件路径')
    parser.add_argument('--qa', '-q', required=True,
                       help='QA_Results_Test.json 文件路径')
    parser.add_argument('--output', '-o', default=None,
                       help='输出 JudgeResult.json 文件路径（默认：根据 --caption 文件名自动生成）')
    parser.add_argument('--prompt', '-p', default=None,
                       help='prompt.txt 文件路径（默认：脚本同目录下的 prompt.txt）')
    parser.add_argument('--api-key', help='API Key（可选，默认从环境变量 OPENAI_API_KEY 读取）')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'使用的模型（默认：{DEFAULT_MODEL}）')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'温度参数（默认：{DEFAULT_TEMPERATURE}）')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='API Base URL')
    parser.add_argument('--use-openai', action='store_true', help='使用官方 OpenAI API')
    parser.add_argument('--max-retries', type=int, default=100, help='最大重试次数（默认：100）')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='并行调用 API 的线程数（默认：8）')

    args = parser.parse_args()

    # 自动生成 output 路径：将 CaptionResult.jsonl 替换为 JudgeResult.json
    if args.output is None:
        base = args.caption
        if "CaptionResult" in base:
            args.output = base.replace("CaptionResult.jsonl", "JudgeResult.json")
        else:
            # fallback: 同目录下加 _JudgeResult.json
            root, _ = os.path.splitext(base)
            args.output = root + "_JudgeResult.json"
        print(f"自动生成输出路径: {args.output}")

    # API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("请提供 API Key（通过 --api-key 或环境变量 OPENAI_API_KEY）")

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
    print(f"加载 Prompt: {prompt_path}")
    prompt = load_prompt(prompt_path)

    # 加载数据
    print(f"加载 CaptionResult: {args.caption}")
    caption_map = load_caption_results(args.caption)
    print(f"  -> {len(caption_map)} 条打标结果")

    print(f"加载 QA: {args.qa}")
    qa_data = load_qa_results(args.qa)
    print(f"  -> {len(qa_data)} 条 QA 样本")

    # 从 caption 文件名提取 caption model 名（用于 CSV 记录）
    caption_basename = os.path.basename(args.caption)
    if "_CaptionResult" in caption_basename:
        model_name = caption_basename.split("_CaptionResult")[0]
    else:
        model_name = os.path.splitext(caption_basename)[0]

    # 断点续传
    processed_ids = set()
    all_results = []
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if isinstance(all_results, list):
                    processed_ids = {r.get("sample_id", "") for r in all_results if r.get("sample_id")}
                    print(f"已处理 {len(processed_ids)} 个样本，从断点继续")
        except Exception:
            pass

    # 过滤已处理
    remaining = []
    skipped_no_caption = 0
    for qa_sample in qa_data:
        sid = qa_sample.get("sample_id", "")
        if sid in processed_ids:
            continue
        if sid not in caption_map:
            skipped_no_caption += 1
            continue
        # 跳过没有 QA 或状态异常的样本
        if qa_sample.get("status") not in ("ok",) or not qa_sample.get("qas"):
            # 对于无 QA 的样本，直接写入空结果
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
        print(f"  -> 跳过 {skipped_no_caption} 个样本（CaptionResult 中无对应记录）")

    if not remaining:
        print("所有样本已处理完成！")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        # 即使全部已处理，也需要更新 CSV
        _finalize_and_update_csv(qa_data, all_results, args.output, model_name)
        return

    print(f"\n开始 Judge 评测: {len(remaining)} 个样本 (model={args.model}, temp={args.temperature}, workers={args.num_workers})")
    print("=" * 60)

    success_count = 0
    error_count = 0
    results_lock = threading.Lock()
    save_counter = 0

    def process_sample(qa_sample):
        """处理单个样本并返回结果"""
        sid = qa_sample["sample_id"]
        caption_record = caption_map[sid]

        # 构建 Judge 输入
        judge_input = build_judge_input(caption_record, qa_sample)

        # 调用 LLM Judge
        response_text = call_judge(
            client, prompt, judge_input,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            use_openai=args.use_openai,
            sample_id=sid
        )

        # 解析结果
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
                # 每 5 个样本保存一次
                if save_counter % 5 == 0:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
            pbar.update(1)

        pbar.close()

    # 最终保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"\n{'=' * 60}")
    print(f"Judge 评测完成！")
    print(f"总 QA 样本数: {len(qa_data)}")
    print(f"本次处理: {len(remaining)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")

    # 统计并更新 CSV
    _finalize_and_update_csv(qa_data, all_results, args.output, model_name)

    print(f"\n结果已保存到: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
