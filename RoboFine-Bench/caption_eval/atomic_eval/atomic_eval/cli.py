"""CLI entry point for atomic fact matching evaluation."""

import argparse
import os
import sys

from .config import DEFAULT_MODEL, DEFAULT_BASE_URL, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_NUM_WORKERS


def _add_common_api_args(parser):
    """Add common API-related arguments to a subparser."""
    parser.add_argument("--api-key", default=None,
                        help="API Key (默认从环境变量 OPENAI_API_KEY 读取)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Judge 模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"温度参数 (默认: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"API Base URL (默认: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"最大重试次数 (默认: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"并行线程数 (默认: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="开启 thinking/reasoning 模式（适用于 qwen3/qwen3.5 等支持的模型）")


def main():
    parser = argparse.ArgumentParser(
        description="Atomic Fact Matching Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Phase 1: 提取 GT 原子事实（只运行一次）
  python -m caption_eval.atomic_eval.atomic_eval extract-gt \\
    --evalsets EvalData/EvalSets.json \\
    --output EvalData/GT_AtomicFacts.jsonl

  # Phase 2 (完整): 提取 caption 事实 + 对齐 + 评分
  python -m caption_eval.atomic_eval.atomic_eval evaluate \\
    --gt-facts EvalData/GT_AtomicFacts.jsonl \\
    --caption CaptionEval/CaptionResult/xxx_CaptionResult.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # Phase 2a (单独): 仅提取 caption 原子事实 (thinking=ON)
  python -m caption_eval.atomic_eval.atomic_eval extract-caption \\
    --gt-facts EvalData/GT_AtomicFacts.jsonl \\
    --caption CaptionEval/CaptionResult/xxx_CaptionResult.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # Phase 2b (单独): 仅做对齐比对 (thinking=OFF)
  python -m caption_eval.atomic_eval.atomic_eval align \\
    --gt-facts EvalData/GT_AtomicFacts.jsonl \\
    --caption CaptionEval/CaptionResult/xxx_CaptionResult.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # 仅重新计算分数
  python -m caption_eval.atomic_eval.atomic_eval score-only \\
    --judge-raw CaptionEval/AtomicResult/xxx_AtomicEval/judge_raw.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # 跨模型汇总
  python -m caption_eval.atomic_eval.atomic_eval summary \\
    --results-dirs CaptionEval/AtomicResult/*_AtomicEval/ \\
    --output CaptionEval/cross_model_summary.csv
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── extract-gt ──
    p_gt = subparsers.add_parser(
        "extract-gt",
        help="Phase 1: 从 EvalSets.json 提取 GT 原子事实（只运行一次）",
    )
    p_gt.add_argument("--evalsets", required=True,
                      help="EvalSets.json 文件路径（也兼容 Human_Review.jsonl）")
    p_gt.add_argument("--output", default="CaptionEval/AtomicResult/GT_AtomicFacts.jsonl",
                      help="输出文件路径 (默认: CaptionEval/AtomicResult/GT_AtomicFacts.jsonl)")
    _add_common_api_args(p_gt)

    # ── extract-caption ──
    p_cap = subparsers.add_parser(
        "extract-caption",
        help="Phase 2a: 从 AI Caption 提取原子事实（thinking=ON）",
    )
    p_cap.add_argument("--gt-facts", default=None,
                       help="GT 原子事实文件路径（用于过滤样本，可选）")
    p_cap.add_argument("--caption", required=True,
                       help="CaptionResult.jsonl 文件路径")
    p_cap.add_argument("--output-dir", required=True,
                       help="输出目录路径")
    _add_common_api_args(p_cap)

    # ── align ──
    p_align = subparsers.add_parser(
        "align",
        help="Phase 2b: 将 caption 原子事实与 GT 原子事实对齐比对（thinking=OFF）",
    )
    p_align.add_argument("--gt-facts", required=True,
                         help="Human_Review_AtomicFacts.jsonl 文件路径")
    p_align.add_argument("--caption", required=True,
                         help="CaptionResult.jsonl 文件路径（用于提取模型名）")
    p_align.add_argument("--output-dir", required=True,
                         help="输出目录路径（需包含 caption_atomic_facts.jsonl）")
    p_align.add_argument("--rerun-all", action="store_true", default=False,
                         help="忽略已有结果，全部重新对齐（默认只重跑失败的）")
    _add_common_api_args(p_align)

    # ── evaluate (2a + 2b combined) ──
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Phase 2: 完整评估（2a 提取 + 2b 对齐 + 评分）",
    )
    p_eval.add_argument("--gt-facts", required=True,
                        help="Human_Review_AtomicFacts.jsonl 文件路径")
    p_eval.add_argument("--caption", required=True,
                        help="CaptionResult.jsonl 文件路径")
    p_eval.add_argument("--output-dir", required=True,
                        help="输出目录路径")
    p_eval.add_argument("--rerun-all", action="store_true", default=False,
                         help="忽略已有对齐结果，全部重新对齐")
    _add_common_api_args(p_eval)

    # ── direct-align ──
    p_direct = subparsers.add_parser(
        "direct-align",
        help="Direct Alignment (Method B): GT 原子事实 + 原始 Caption → GPT 直接判断",
    )
    p_direct.add_argument("--gt-facts", required=True,
                          help="GT_AtomicFacts.jsonl 文件路径")
    p_direct.add_argument("--caption", required=True,
                          help="CaptionResult.jsonl 文件路径")
    p_direct.add_argument("--output-dir", required=True,
                          help="输出目录路径")
    _add_common_api_args(p_direct)

    # ── score-only ──
    p_score = subparsers.add_parser(
        "score-only",
        help="从已有的 judge_raw.jsonl 重新计算分数（不调用 API）",
    )
    p_score.add_argument("--judge-raw", required=True,
                         help="judge_raw.jsonl 文件路径")
    p_score.add_argument("--output-dir", required=True,
                         help="输出目录路径")

    # ── summary ──
    p_summ = subparsers.add_parser(
        "summary",
        help="跨模型汇总对比表",
    )
    p_summ.add_argument("--results-dirs", nargs="+", required=True,
                        help="各模型结果目录（包含 dataset_summary.json）")
    p_summ.add_argument("--output", default="Eval_Result/cross_model_summary.csv",
                        help="输出 CSV 路径")

    args = parser.parse_args()

    # Dispatch
    from .pipeline import (
        run_gt_extraction, run_caption_extraction, run_alignment,
        run_evaluation, run_score_only, run_summary, run_direct_alignment,
    )

    if args.command == "extract-gt":
        run_gt_extraction(args)
    elif args.command == "extract-caption":
        run_caption_extraction(args)
    elif args.command == "align":
        run_alignment(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "direct-align":
        run_direct_alignment(args)
    elif args.command == "score-only":
        run_score_only(args)
    elif args.command == "summary":
        run_summary(args)


if __name__ == "__main__":
    main()
