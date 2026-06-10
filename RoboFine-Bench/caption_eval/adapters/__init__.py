"""Model adapters for RoboFine-Bench Caption evaluation."""

from .base import CaptionModelAdapter, CaptionRequest, CaptionResponse, CaptionView
from .openai_compatible import OpenAICompatibleAdapter


def create_caption_adapter(
    adapter: str,
    model: str,
    api_key: str = None,
    base_url: str = None,
) -> CaptionModelAdapter:
    """Create a caption model adapter by name."""
    name = adapter.replace("_", "-").lower()
    if name in {"openai-compatible", "openai", "dashscope", "vllm"}:
        return OpenAICompatibleAdapter(model=model, api_key=api_key, base_url=base_url)
    raise ValueError(
        f"Unknown caption adapter: {adapter}. "
        "Use 'openai-compatible' or implement a new adapter under caption_eval/adapters/."
    )

