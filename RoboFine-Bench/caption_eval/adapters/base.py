"""Adapter interface for Caption benchmark model calls.

Benchmark code owns data loading, view selection, prompts, parsing, and result
formatting. Model adapters only translate a standardized CaptionRequest into a
model/API call and return raw text plus optional token usage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class CaptionView:
    """One synchronized view in a RoboFine-Bench sample."""

    name: str
    label: str
    video_url: Optional[str] = None
    frame_urls: List[str] = field(default_factory=list)


@dataclass
class CaptionRequest:
    """Standard input passed from the benchmark runner to a model adapter."""

    sample_id: str
    dataset: str
    input_type: str
    fps: float
    prompt: str
    system_prompt: str
    views: List[CaptionView]
    image_parts: List[Dict] = field(default_factory=list)
    video_urls: List[str] = field(default_factory=list)


@dataclass
class CaptionResponse:
    """Raw model output returned by an adapter."""

    text: str
    token_usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Dict = field(default_factory=dict)


class CaptionModelAdapter(Protocol):
    """Interface implemented by all caption model adapters."""

    model_name: str

    def generate_caption(self, request: CaptionRequest) -> CaptionResponse:
        """Call a model for one benchmark sample."""
        ...

