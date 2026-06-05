"""CLI entry point for atomic fact matching evaluation."""

import argparse
import os
import sys

from .config import DEFAULT_MODEL, DEFAULT_BASE_URL, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_NUM_WORKERS


def _add_common_api_args(parser):
    """Add common API-related arguments to a subparser."""
    parser.add_argument("--api-key", default=None,
                        help="API Key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Judge model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature parameter (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"API Base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"Max retries (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable thinking/reasoning mode (for models that support it, e.g. qwen3/qwen3.5)")


def main():
    parser = argparse.ArgumentParser(
        description="Atomic Fact Matching Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Extract GT atomic facts (run once)
  python -m caption_eval.atomic_eval.atomic_eval extract-gt \\
    --evalsets EvalData/EvalSets.json \\
    --output EvalData/GT_AtomicFacts.jsonl

  # Phase 2 (full): Extract caption facts + align + score
  python -m caption_eval.atomic_eval.atomic_eval evaluate \\
    --gt-facts EvalData/GT_AtomicFacts.jsonl \\
    --caption CaptionEval/CaptionResult/xxx_CaptionResult.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # Phase 2a (standalone): Extract caption atomic facts only (thinking=ON)
  python -m caption_eval.atomic_eval.atomic_eval extract-caption \\
    --gt-facts EvalData/GT_AtomicFacts.jsonl \\
    --caption CaptionEval/CaptionResult/xxx_CaptionResult.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # Phase 2b (standalone): Alignment only (thinking=OFF)
  python -m caption_eval.atomic_eval.atomic_eval align \\
    --gt-facts EvalData/GT_AtomicFacts.jsonl \\
    --caption CaptionEval/CaptionResult/xxx_CaptionResult.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # Recompute scores only
  python -m caption_eval.atomic_eval.atomic_eval score-only \\
    --judge-raw CaptionEval/AtomicResult/xxx_AtomicEval/judge_raw.jsonl \\
    --output-dir CaptionEval/AtomicResult/xxx_AtomicEval/

  # Cross-model summary
  python -m caption_eval.atomic_eval.atomic_eval summary \\
    --results-dirs CaptionEval/AtomicResult/*_AtomicEval/ \\
    --output CaptionEval/cross_model_summary.csv
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── extract-gt ──
    p_gt = subparsers.add_parser(
        "extract-gt",
        help="Phase 1: Extract GT atomic facts from EvalSets.json (run once)",
    )
    p_gt.add_argument("--evalsets", required=True,
                      help="Path to EvalSets.json (also compatible with Human_Review.jsonl)")
    p_gt.add_argument("--output", default="CaptionEval/AtomicResult/GT_AtomicFacts.jsonl",
                      help="Output file path (default: CaptionEval/AtomicResult/GT_AtomicFacts.jsonl)")
    _add_common_api_args(p_gt)

    # ── extract-caption ──
    p_cap = subparsers.add_parser(
        "extract-caption",
        help="Phase 2a: Extract atomic facts from AI captions (thinking=ON)",
    )
    p_cap.add_argument("--gt-facts", default=None,
                       help="Path to GT atomic facts file (for sample filtering, optional)")
    p_cap.add_argument("--caption", required=True,
                       help="Path to CaptionResult.jsonl")
    p_cap.add_argument("--output-dir", required=True,
                       help="Output directory path")
    _add_common_api_args(p_cap)

    # ── align ──
    p_align = subparsers.add_parser(
        "align",
        help="Phase 2b: Align caption atomic facts against GT atomic facts (thinking=OFF)",
    )
    p_align.add_argument("--gt-facts", required=True,
                         help="Path to Human_Review_AtomicFacts.jsonl")
    p_align.add_argument("--caption", required=True,
                         help="Path to CaptionResult.jsonl (used to extract model name)")
    p_align.add_argument("--output-dir", required=True,
                         help="Output directory path (must contain caption_atomic_facts.jsonl)")
    p_align.add_argument("--rerun-all", action="store_true", default=False,
                         help="Ignore existing results and re-align all (default: only re-run failures)")
    _add_common_api_args(p_align)

    # ── evaluate (2a + 2b combined) ──
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Phase 2: Full evaluation (2a extraction + 2b alignment + scoring)",
    )
    p_eval.add_argument("--gt-facts", required=True,
                        help="Path to Human_Review_AtomicFacts.jsonl")
    p_eval.add_argument("--caption", required=True,
                        help="Path to CaptionResult.jsonl")
    p_eval.add_argument("--output-dir", required=True,
                        help="Output directory path")
    p_eval.add_argument("--rerun-all", action="store_true", default=False,
                         help="Ignore existing alignment results and re-align all")
    _add_common_api_args(p_eval)

    # ── direct-align ──
    p_direct = subparsers.add_parser(
        "direct-align",
        help="Direct Alignment (Method B): GT atomic facts + raw caption -> GPT direct judgment",
    )
    p_direct.add_argument("--gt-facts", required=True,
                          help="Path to GT_AtomicFacts.jsonl")
    p_direct.add_argument("--caption", required=True,
                          help="Path to CaptionResult.jsonl")
    p_direct.add_argument("--output-dir", required=True,
                          help="Output directory path")
    _add_common_api_args(p_direct)

    # ── score-only ──
    p_score = subparsers.add_parser(
        "score-only",
        help="Recompute scores from existing judge_raw.jsonl (no API calls)",
    )
    p_score.add_argument("--judge-raw", required=True,
                         help="Path to judge_raw.jsonl")
    p_score.add_argument("--output-dir", required=True,
                         help="Output directory path")

    # ── summary ──
    p_summ = subparsers.add_parser(
        "summary",
        help="Cross-model comparison summary table",
    )
    p_summ.add_argument("--results-dirs", nargs="+", required=True,
                        help="Result directories for each model (containing dataset_summary.json)")
    p_summ.add_argument("--output", default="Eval_Result/cross_model_summary.csv",
                        help="Output CSV path")

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
