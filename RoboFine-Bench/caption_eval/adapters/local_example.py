"""Template adapter for a local or non-OpenAI-compatible model.

Copy this file, replace LocalExampleAdapter.generate_caption(), and register the
adapter in caption_eval/adapters/__init__.py.
"""

from .base import CaptionRequest, CaptionResponse


class LocalExampleAdapter:
    """Minimal template for custom model integration."""

    def __init__(self, model: str, **kwargs):
        self.model_name = model
        # Load your model/client here.

    def generate_caption(self, request: CaptionRequest) -> CaptionResponse:
        # request.views contains ordered multi-view metadata.
        # request.video_urls contains one video URL per view for input_type=video.
        # request.image_parts contains interleaved [View: ...] text and image_url
        # parts for input_type=image.
        #
        # Convert those fields into your model's expected format, call the model,
        # and return its raw text output. The benchmark runner will parse the JSON.
        raise NotImplementedError("Implement your model call here.")

