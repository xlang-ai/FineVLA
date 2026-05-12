"""Pydantic data models for atomic fact matching evaluation."""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator

from .config import CAPABILITIES


# ── Atomic building blocks ──────────────────────────────────────────

class AtomicFact(BaseModel):
    fact_id: str
    step_anchor: Optional[str] = None
    object: Optional[str] = None
    slot: str
    value: str
    fact_text: str

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v):
        if not isinstance(v, str):
            return str(v)
        return v


class Alignment(BaseModel):
    gt_fact_id: str
    caption_fact_id: str
    label: Literal["match", "partial", "contradiction"]
    note: str


class Omission(BaseModel):
    gt_fact_id: str
    note: str


class Hallucination(BaseModel):
    caption_fact_id: str
    reason: str


class IgnoredExtra(BaseModel):
    caption_fact_id: str
    reason: str


# ── Phase 1: GT extraction result ──────────────────────────────────

class GTCapabilityResult(BaseModel):
    capability: str
    gt_atomic_facts: list[AtomicFact]


class GTExtractionResult(BaseModel):
    sample_id: str
    status: str  # "ok"
    capability_results: list[GTCapabilityResult]

    @model_validator(mode="after")
    def validate_capabilities(self):
        caps = [c.capability for c in self.capability_results]
        expected_n = len(CAPABILITIES)
        if len(caps) != expected_n:
            raise ValueError(f"Expected {expected_n} capability_results, got {len(caps)}")
        for i, (got, expected) in enumerate(zip(caps, CAPABILITIES)):
            if got != expected:
                raise ValueError(
                    f"capability_results[{i}]: expected '{expected}', got '{got}'"
                )
        return self


# ── Phase 2a: Caption atomic fact extraction ──────────────────────

class CaptionCapabilityResult(BaseModel):
    capability: str
    caption_atomic_facts: list[AtomicFact]


class CaptionExtractionResult(BaseModel):
    sample_id: str
    status: str  # "ok"
    capability_results: list[CaptionCapabilityResult]

    @model_validator(mode="after")
    def validate_capabilities(self):
        caps = [c.capability for c in self.capability_results]
        expected_n = len(CAPABILITIES)
        if len(caps) != expected_n:
            raise ValueError(f"Expected {expected_n} capability_results, got {len(caps)}")
        for i, (got, expected) in enumerate(zip(caps, CAPABILITIES)):
            if got != expected:
                raise ValueError(
                    f"capability_results[{i}]: expected '{expected}', got '{got}'"
                )
        return self


# ── Phase 2b: Full alignment result ──────────────────────────────────

class FullCapabilityResult(BaseModel):
    capability: str
    gt_atomic_facts: list[AtomicFact]
    caption_atomic_facts: list[AtomicFact]
    alignments: list[Alignment]
    omissions: list[Omission]
    hallucinations: list[Hallucination]
    ignored_extras: list[IgnoredExtra]


class AlignmentResult(BaseModel):
    sample_id: str
    status: str  # "ok"
    capability_results: list[FullCapabilityResult]

    @model_validator(mode="after")
    def validate_capabilities(self):
        caps = [c.capability for c in self.capability_results]
        expected_n = len(CAPABILITIES)
        if len(caps) != expected_n:
            raise ValueError(f"Expected {expected_n} capability_results, got {len(caps)}")
        for i, (got, expected) in enumerate(zip(caps, CAPABILITIES)):
            if got != expected:
                raise ValueError(
                    f"capability_results[{i}]: expected '{expected}', got '{got}'"
                )
        return self

    def check_sanity(self) -> list[str]:
        """Return list of sanity-check violation messages (empty = all OK)."""
        errors = []
        for cr in self.capability_results:
            cap = cr.capability
            G = len(cr.gt_atomic_facts)
            C = len(cr.caption_atomic_facts)
            M = sum(1 for a in cr.alignments if a.label == "match")
            P = sum(1 for a in cr.alignments if a.label == "partial")
            X = sum(1 for a in cr.alignments if a.label == "contradiction")
            O = len(cr.omissions)
            H = len(cr.hallucinations)
            I = len(cr.ignored_extras)

            if G != M + P + X + O:
                errors.append(
                    f"{cap}: G={G} != M+P+X+O={M+P+X+O} "
                    f"(M={M}, P={P}, X={X}, O={O})"
                )
            if C != M + P + X + H + I:
                errors.append(
                    f"{cap}: C={C} != M+P+X+H+I={M+P+X+H+I} "
                    f"(M={M}, P={P}, X={X}, H={H}, I={I})"
                )
        return errors


# ── Phase 2b: Lightweight judgment from LLM (per-slot or per-capability) ──

class CapabilityJudgment(BaseModel):
    """LLM alignment judgment (per-slot or merged per-capability)."""
    sample_id: str
    capability: str
    slot: Optional[str] = None
    alignments: list[Alignment]
    omissions: list[Omission]
    hallucinations: list[Hallucination]
    ignored_extras: list[IgnoredExtra]

    def check_sanity(self, G: int, C: int) -> list[str]:
        """Validate alignment counts against known fact counts."""
        errors = []
        ctx = f"{self.capability}/{self.slot}" if self.slot else self.capability
        M = sum(1 for a in self.alignments if a.label == "match")
        P = sum(1 for a in self.alignments if a.label == "partial")
        X = sum(1 for a in self.alignments if a.label == "contradiction")
        O = len(self.omissions)
        H = len(self.hallucinations)
        I = len(self.ignored_extras)

        if G != M + P + X + O:
            errors.append(
                f"{ctx}: G={G} != M+P+X+O={M + P + X + O} "
                f"(M={M}, P={P}, X={X}, O={O})"
            )
        if C != M + P + X + H + I:
            errors.append(
                f"{ctx}: C={C} != M+P+X+H+I={M + P + X + H + I} "
                f"(M={M}, P={P}, X={X}, H={H}, I={I})"
            )
        return errors

    def validate_fact_ids(
        self, gt_ids: set[str], cap_ids: set[str]
    ) -> list[str]:
        """Check all referenced fact_ids exist and each appears exactly once."""
        errors = []

        align_gt = [a.gt_fact_id for a in self.alignments]
        align_cap = [a.caption_fact_id for a in self.alignments]
        omit_gt = [o.gt_fact_id for o in self.omissions]
        hall_cap = [h.caption_fact_id for h in self.hallucinations]
        ignored_cap = [ie.caption_fact_id for ie in self.ignored_extras]

        # GT side: alignments + omissions
        used_gt = set(align_gt + omit_gt)
        for fid in used_gt:
            if fid not in gt_ids:
                errors.append(f"Unknown gt_fact_id: {fid}")
        missing_gt = gt_ids - used_gt
        if missing_gt:
            errors.append(f"GT fact_ids not referenced: {missing_gt}")
        all_gt_refs = align_gt + omit_gt
        dup_gt = {fid for fid in all_gt_refs if all_gt_refs.count(fid) > 1}
        if dup_gt:
            errors.append(f"Duplicate gt_fact_ids: {dup_gt}")

        # Caption side: alignments + hallucinations + ignored_extras
        all_cap_refs = align_cap + hall_cap + ignored_cap
        used_cap = set(all_cap_refs)
        for fid in used_cap:
            if fid not in cap_ids:
                errors.append(f"Unknown caption_fact_id: {fid}")
        missing_cap = cap_ids - used_cap
        if missing_cap:
            errors.append(f"Caption fact_ids not referenced: {missing_cap}")
        dup_cap = {fid for fid in all_cap_refs if all_cap_refs.count(fid) > 1}
        if dup_cap:
            errors.append(f"Duplicate caption_fact_ids: {dup_cap}")

        return errors


def merge_slot_judgments(
    sample_id: str, capability: str,
    slot_judgments: list[CapabilityJudgment],
) -> CapabilityJudgment:
    """Merge per-slot judgments into one capability-level judgment."""
    all_alignments: list[Alignment] = []
    all_omissions: list[Omission] = []
    all_hallucinations: list[Hallucination] = []
    all_ignored: list[IgnoredExtra] = []
    for sj in slot_judgments:
        all_alignments.extend(sj.alignments)
        all_omissions.extend(sj.omissions)
        all_hallucinations.extend(sj.hallucinations)
        all_ignored.extend(sj.ignored_extras)
    return CapabilityJudgment(
        sample_id=sample_id,
        capability=capability,
        slot=None,
        alignments=all_alignments,
        omissions=all_omissions,
        hallucinations=all_hallucinations,
        ignored_extras=all_ignored,
    )


def reconstruct_full_capability_result(
    judgment: CapabilityJudgment,
    gt_facts: list[AtomicFact],
    caption_facts: list[AtomicFact],
) -> FullCapabilityResult:
    """Merge LLM judgment with original input facts into a full result."""
    return FullCapabilityResult(
        capability=judgment.capability,
        gt_atomic_facts=gt_facts,
        caption_atomic_facts=caption_facts,
        alignments=judgment.alignments,
        omissions=judgment.omissions,
        hallucinations=judgment.hallucinations,
        ignored_extras=judgment.ignored_extras,
    )


# ── Scoring output models ──────────────────────────────────────────

class CapabilityScores(BaseModel):
    capability: str
    G: int
    C: int
    M: int
    P: int
    X: int
    O: int
    H: int
    A: int
    consistency: Optional[float] = None
    coverage: Optional[float] = None
    omission_rate: Optional[float] = None
    anti_hallucination: Optional[float] = None
    hallucination_rate: Optional[float] = None


class SampleScores(BaseModel):
    sample_id: str
    status: str
    capability_scores: list[CapabilityScores]
    consistency_score: float
    weighted_coverage: float
    weighted_anti_hallucination: float
    caption_score: float
    error: Optional[str] = None


# ── Direct Alignment (Method B) models ────────────────────────────

class GTEvaluation(BaseModel):
    fact_id: str
    capability: str
    slot: str
    gt_value: str
    label: Literal["match", "partial", "contradiction", "omission"]
    caption_evidence: Optional[str] = None
    note: str

    @field_validator("gt_value", mode="before")
    @classmethod
    def coerce_gt_value_to_str(cls, v):
        if not isinstance(v, str):
            return str(v)
        return v


class HallucinatedAction(BaseModel):
    description: str
    caption_step: str
    note: str


class DirectAlignmentSummary(BaseModel):
    total_gt_facts: int
    match: int
    partial: int
    contradiction: int
    omission: int
    hallucinated_actions: int


class DirectAlignmentResult(BaseModel):
    sample_id: str
    gt_evaluations: list[GTEvaluation]
    hallucinated_actions: list[HallucinatedAction]
    summary: DirectAlignmentSummary

    def to_alignment_result(self) -> AlignmentResult:
        """Convert direct alignment output to standard AlignmentResult for scoring."""
        from .config import CAPABILITIES

        cap_evals: dict[str, list[GTEvaluation]] = {c: [] for c in CAPABILITIES}
        for ev in self.gt_evaluations:
            if ev.capability in cap_evals:
                cap_evals[ev.capability].append(ev)

        capability_results = []
        for cap in CAPABILITIES:
            evals = cap_evals[cap]

            gt_facts = [
                AtomicFact(
                    fact_id=ev.fact_id, slot=ev.slot,
                    value=ev.gt_value, fact_text=ev.note,
                )
                for ev in evals
            ]

            alignments = []
            omissions = []
            for ev in evals:
                if ev.label in ("match", "partial", "contradiction"):
                    alignments.append(Alignment(
                        gt_fact_id=ev.fact_id,
                        caption_fact_id=f"_direct_{ev.fact_id}",
                        label=ev.label,
                        note=ev.note,
                    ))
                elif ev.label == "omission":
                    omissions.append(Omission(
                        gt_fact_id=ev.fact_id,
                        note=ev.note,
                    ))

            hallucinations = []
            ignored_extras = []
            if cap == "action_sequence":
                for i, h in enumerate(self.hallucinated_actions):
                    hallucinations.append(Hallucination(
                        caption_fact_id=f"_halluc_{i+1}",
                        reason=h.note,
                    ))

            caption_facts = [
                AtomicFact(
                    fact_id=a.caption_fact_id, slot="", value="", fact_text="",
                )
                for a in alignments
            ]
            if cap == "action_sequence":
                caption_facts.extend([
                    AtomicFact(
                        fact_id=h.caption_fact_id, slot="primitive_action",
                        value=h.reason, fact_text="",
                    )
                    for h in hallucinations
                ])

            capability_results.append(FullCapabilityResult(
                capability=cap,
                gt_atomic_facts=gt_facts,
                caption_atomic_facts=caption_facts,
                alignments=alignments,
                omissions=omissions,
                hallucinations=hallucinations,
                ignored_extras=ignored_extras,
            ))

        return AlignmentResult(
            sample_id=self.sample_id,
            status="ok",
            capability_results=capability_results,
        )
