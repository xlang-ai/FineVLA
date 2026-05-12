"""Scoring computation — all arithmetic in Python, not in the LLM."""

from typing import Optional

from .config import (
    CAPABILITIES,
    CAPABILITY_WEIGHTS,
    PARTIAL_CREDIT_ALPHA,
    CAPTION_SCORE_W_CONSISTENCY,
    CAPTION_SCORE_W_COVERAGE,
    CAPTION_SCORE_W_ANTI_HALLUCINATION,
)
from .models import AlignmentResult, CapabilityScores, SampleScores


def compute_capability_scores(cap_result) -> CapabilityScores:
    """Compute scores for a single capability from alignment result."""
    G = len(cap_result.gt_atomic_facts)
    C = len(cap_result.caption_atomic_facts)
    M = sum(1 for a in cap_result.alignments if a.label == "match")
    P = sum(1 for a in cap_result.alignments if a.label == "partial")
    X = sum(1 for a in cap_result.alignments if a.label == "contradiction")
    O = len(cap_result.omissions)
    H = len(cap_result.hallucinations)
    A = M + P + X

    # Per-capability metrics
    alpha = PARTIAL_CREDIT_ALPHA

    consistency: Optional[float] = None
    if A > 0:
        consistency = (M + alpha * P) / A

    coverage: Optional[float] = None
    if G > 0:
        coverage = (M + alpha * P) / G

    omission_rate: Optional[float] = None
    if coverage is not None:
        omission_rate = 1.0 - coverage

    anti_hallucination: Optional[float] = None
    if C > 0:
        anti_hallucination = 1.0 - (H / C)

    hallucination_rate: Optional[float] = None
    if anti_hallucination is not None:
        hallucination_rate = 1.0 - anti_hallucination

    return CapabilityScores(
        capability=cap_result.capability,
        G=G, C=C, M=M, P=P, X=X, O=O, H=H, A=A,
        consistency=consistency,
        coverage=coverage,
        omission_rate=omission_rate,
        anti_hallucination=anti_hallucination,
        hallucination_rate=hallucination_rate,
    )


def compute_sample_scores(alignment: AlignmentResult) -> SampleScores:
    """Compute per-sample aggregate scores from alignment result."""
    cap_scores = [
        compute_capability_scores(cr) for cr in alignment.capability_results
    ]

    # Weighted aggregates for consistency and coverage
    consistency_score = _weighted_avg(
        [(cs.capability, cs.consistency) for cs in cap_scores],
        condition=lambda cs: cs.A > 0,
        cap_scores_list=cap_scores,
    )

    weighted_coverage = _weighted_avg(
        [(cs.capability, cs.coverage) for cs in cap_scores],
        condition=lambda cs: cs.G > 0,
        cap_scores_list=cap_scores,
    )

    # Anti-Hallucination: 1 - H / G_action_sequence
    # H only exists for action_sequence; denominator is GT action fact count
    action_cs = next(
        (cs for cs in cap_scores if cs.capability == "action_sequence"), None
    )
    if action_cs and action_cs.G > 0:
        weighted_anti_hallucination = 1.0 - (action_cs.H / action_cs.G)
    else:
        weighted_anti_hallucination = 1.0

    caption_score = (
        CAPTION_SCORE_W_CONSISTENCY * consistency_score
        + CAPTION_SCORE_W_COVERAGE * weighted_coverage
        + CAPTION_SCORE_W_ANTI_HALLUCINATION * weighted_anti_hallucination
    )

    return SampleScores(
        sample_id=alignment.sample_id,
        status="ok",
        capability_scores=cap_scores,
        consistency_score=round(consistency_score, 4),
        weighted_coverage=round(weighted_coverage, 4),
        weighted_anti_hallucination=round(weighted_anti_hallucination, 4),
        caption_score=round(caption_score, 4),
    )


def _weighted_avg(
    cap_values: list[tuple[str, Optional[float]]],
    condition,
    cap_scores_list: list[CapabilityScores],
) -> float:
    """Compute weighted average over active capabilities."""
    # Build lookup
    cs_map = {cs.capability: cs for cs in cap_scores_list}

    numerator = 0.0
    denominator = 0.0
    for cap_name, value in cap_values:
        cs = cs_map[cap_name]
        if condition(cs) and value is not None:
            w = CAPABILITY_WEIGHTS[cap_name]
            numerator += w * value
            denominator += w

    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_dataset_summary(
    all_scores: list[SampleScores], model_name: str
) -> dict:
    """Compute dataset-level aggregate summary."""
    scored = [s for s in all_scores if s.status == "ok"]
    n_scored = len(scored)

    if n_scored == 0:
        return {
            "model_name": model_name,
            "num_total_samples": len(all_scores),
            "num_scored_samples": 0,
            "mean_consistency_score": 0.0,
            "mean_weighted_coverage": 0.0,
            "mean_weighted_anti_hallucination": 0.0,
            "mean_caption_score": 0.0,
            "per_capability": {},
        }

    mean_consistency = round(sum(s.consistency_score for s in scored) / n_scored, 4)
    mean_coverage = round(sum(s.weighted_coverage for s in scored) / n_scored, 4)
    mean_anti_hall = round(sum(s.weighted_anti_hallucination for s in scored) / n_scored, 4)
    mean_caption = round(sum(s.caption_score for s in scored) / n_scored, 4)

    # Per-capability aggregates
    per_cap = {}
    for cap in CAPABILITIES:
        cap_data = []
        total_G = total_C = total_M = total_P = total_X = total_O = total_H = 0
        consistency_vals = []
        coverage_vals = []
        anti_hall_vals = []

        for s in scored:
            for cs in s.capability_scores:
                if cs.capability == cap:
                    total_G += cs.G
                    total_C += cs.C
                    total_M += cs.M
                    total_P += cs.P
                    total_X += cs.X
                    total_O += cs.O
                    total_H += cs.H
                    if cs.consistency is not None:
                        consistency_vals.append(cs.consistency)
                    if cs.coverage is not None:
                        coverage_vals.append(cs.coverage)
                    if cs.anti_hallucination is not None:
                        anti_hall_vals.append(cs.anti_hallucination)

        per_cap[cap] = {
            "mean_consistency": round(sum(consistency_vals) / len(consistency_vals), 4) if consistency_vals else None,
            "mean_coverage": round(sum(coverage_vals) / len(coverage_vals), 4) if coverage_vals else None,
            "mean_anti_hallucination": round(sum(anti_hall_vals) / len(anti_hall_vals), 4) if anti_hall_vals else None,
            "total_G": total_G,
            "total_C": total_C,
            "total_M": total_M,
            "total_P": total_P,
            "total_X": total_X,
            "total_O": total_O,
            "total_H": total_H,
        }

    return {
        "model_name": model_name,
        "num_total_samples": len(all_scores),
        "num_scored_samples": n_scored,
        "num_errors": len(all_scores) - n_scored,
        "mean_consistency_score": mean_consistency,
        "mean_weighted_coverage": mean_coverage,
        "mean_weighted_anti_hallucination": mean_anti_hall,
        "mean_caption_score": mean_caption,
        "per_capability": per_cap,
    }
