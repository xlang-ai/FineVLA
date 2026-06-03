from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrajectoryItem:
    """Represents a single trajectory item from the JSONL input."""
    dataset: str
    task: str
    initial_instruction: str
    video_path: str
    robot_type: str = ""
    original_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryItem':
        return cls(
            dataset=data.get('dataset', ''),
            task=data.get('sample_id', data.get('task', '')),
            initial_instruction=data.get('instruction_raw', data.get('initialInstruction', '')),
            video_path=data.get('video_path', data.get('videoPath', '')),
            robot_type=data.get('robot_type', ''),
            original_data=data.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return original data structure."""
        return self.original_data.copy()


@dataclass
class StageResult:
    """Generic result from any pipeline stage."""
    stage_name: str
    output: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    success: bool = False
    error: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    required: bool = True  # False for optional stages (e.g. detail_refinement)


REQUIRED_STAGES: Set[str] = {"analysis", "refinement"}


@dataclass
class ProcessingResult:
    """Complete processing result for a trajectory (supports N stages)."""
    trajectory: TrajectoryItem
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    processing_time: float = 0.0
    model: str = ""

    @property
    def success(self) -> bool:
        """Success = all required (non-optional) stages that were run succeeded."""
        if not self.stage_results:
            return False
        for name, r in self.stage_results.items():
            if r.required and not r.success:
                return False
        return True

    def to_output_dict(self) -> Dict[str, Any]:
        """Convert to output dictionary format with backward compatibility."""
        output = self.trajectory.to_dict()

        # Backward compat: "analysis" → analysisResult
        analysis = self.stage_results.get("analysis")
        if analysis:
            output['analysisResult'] = {
                'action_sequence': analysis.output.get('action_sequence', []),
                'main_object': analysis.output.get('main_object', ''),
            }

        # Backward compat: "refinement" → refinedInstruction / fineGrainedSteps
        # If detail_refinement ran successfully, its output supersedes refinement's.
        refinement = self.stage_results.get("refinement")
        detail = self.stage_results.get("detail_refinement")
        best_refine = detail if (detail and detail.success) else refinement
        if best_refine:
            output['refinedInstruction'] = best_refine.output.get('refined_instruction', '')
            output['fineGrainedSteps'] = best_refine.output.get('fine_grained_steps', [])

        # detail_refinement extras (changes_made, original refinement preserved)
        if detail and detail.success and refinement:
            output['detailRefinement'] = {
                'changes_made': detail.output.get('changes_made', []),
                'pre_detail_refinedInstruction': refinement.output.get('refined_instruction', ''),
                'pre_detail_fineGrainedSteps': refinement.output.get('fine_grained_steps', []),
            }

        # Generic: any stage not in the well-known set gets its own top-level key
        _COMPAT_STAGES = {"analysis", "refinement", "detail_refinement"}
        for name, r in self.stage_results.items():
            if name not in _COMPAT_STAGES:
                output[f'stage_{name}'] = r.output

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for r in self.stage_results.values():
            for k in total_usage:
                total_usage[k] += r.token_usage.get(k, 0)

        output['processingMetadata'] = {
            'success': self.success,
            'model': self.model,
            'processingTime': round(self.processing_time, 2),
            'tokenUsage': total_usage,
        }
        for name, r in self.stage_results.items():
            if r.error:
                output['processingMetadata'][f'{name}Error'] = r.error

        return output
