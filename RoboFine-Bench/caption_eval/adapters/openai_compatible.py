"""OpenAI-compatible Caption adapter.

This adapter covers DashScope compatible-mode, OpenAI-compatible vLLM servers,
and other endpoints that accept chat.completions multimodal messages. Model
specific dispatch remains in annotate/api_call.py for backward compatibility.
"""

from typing import Optional

from caption_eval.annotate.api_call import call_api, create_api_client

from .base import CaptionRequest, CaptionResponse


class OpenAICompatibleAdapter:
    """Caption adapter backed by an OpenAI-compatible client."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model
        self.client = create_api_client(api_key=api_key, base_url=base_url)

    def generate_caption(self, request: CaptionRequest) -> CaptionResponse:
        text, token_usage = call_api(
            self.client,
            request.image_parts,
            request.system_prompt,
            request.prompt,
            self.model_name,
            video_urls=request.video_urls,
            fps=request.fps,
        )
        return CaptionResponse(text=text or "", token_usage=token_usage or {})

