"""Library-backed Qwen3.6 frontend loaded from the artifact resources."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

from .bindings import ArtifactBinding, BoundResource
from .multimodal import MultimodalBatch, batch_from_processor_output


IMAGE_PIXELS = 1024 * 1024
VIDEO_PIXELS = 4 * 1024 * 1024

_SPECIAL_TOKEN_IDS = {
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
}
_TOKENIZER_SIZE = 248077


def _fetch_videos_opencv(_processor, video_or_videos, sample_indices_fn=None):
    """Use the Transformers sampler with its local OpenCV decoder."""

    from transformers.video_utils import load_video

    if isinstance(video_or_videos, list):
        fetched = [
            _fetch_videos_opencv(
                _processor, item, sample_indices_fn=sample_indices_fn
            )
            for item in video_or_videos
        ]
        return list(zip(*fetched))
    return load_video(
        video_or_videos,
        backend="opencv",
        sample_indices_fn=sample_indices_fn,
    )


class Frontend:
    """Processor, tokenizer, template, and generation defaults for one artifact."""

    def __init__(self, binding: ArtifactBinding):
        try:
            from transformers import AutoProcessor, GenerationConfig
            from transformers.utils import is_torchcodec_available
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.6 reference inference requires Transformers with Qwen3-VL support"
            ) from exc

        resources = binding.frontend
        files: tuple[tuple[str, BoundResource], ...] = (
            ("tokenizer.json", resources.tokenizer_json),
            ("tokenizer_config.json", resources.tokenizer_config_json),
            ("chat_template.jinja", resources.chat_template_jinja),
            ("generation_config.json", resources.generation_config_json),
            ("preprocessor_config.json", resources.preprocessor_config_json),
            (
                "video_preprocessor_config.json",
                resources.video_preprocessor_config_json,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="ninfer-frontend-") as temporary:
            directory = Path(temporary)
            for filename, resource in files:
                (directory / filename).write_bytes(binding.resource_bytes(resource))
            self.processor = AutoProcessor.from_pretrained(
                directory, local_files_only=True
            )
            self.generation_config = GenerationConfig.from_pretrained(
                directory, local_files_only=True
            )

        self.tokenizer = self.processor.tokenizer
        if not is_torchcodec_available():
            self.processor.video_processor.fetch_videos = MethodType(
                _fetch_videos_opencv, self.processor.video_processor
            )

        names = set(self.processor.model_input_names)
        required = {
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "mm_token_type_ids",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(
                "the installed Transformers processor lacks Qwen3.6 inputs: "
                f"{missing}"
            )
        if len(self.tokenizer) != _TOKENIZER_SIZE:
            raise ValueError(
                f"artifact tokenizer has {len(self.tokenizer)} addressable IDs; "
                f"expected {_TOKENIZER_SIZE}"
            )
        for token, expected in _SPECIAL_TOKEN_IDS.items():
            actual = self.tokenizer.convert_tokens_to_ids(token)
            if actual != expected:
                raise ValueError(
                    f"artifact tokenizer maps {token} to {actual}; expected {expected}"
                )

    @property
    def default_stop_token_ids(self) -> set[int]:
        values = self.generation_config.eos_token_id
        if values is None:
            return set()
        if isinstance(values, int):
            return {values}
        return {int(value) for value in values}

    def process(
        self, messages: list[dict[str, Any]], *, thinking: bool
    ) -> MultimodalBatch:
        output = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": 32 * 32,
                        "longest_edge": IMAGE_PIXELS,
                    }
                },
                "videos_kwargs": {
                    "size": {
                        "shortest_edge": 128 * 32 * 32,
                        "longest_edge": VIDEO_PIXELS,
                    }
                },
            },
        )
        return batch_from_processor_output(output)

    def process_text(self, text: str, *, thinking: bool) -> MultimodalBatch:
        if not text.strip():
            raise ValueError("prompt text must not be empty")
        return self.process(
            [{"role": "user", "content": text}],
            thinking=thinking,
        )

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = True
    ) -> str:
        return self.tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )


__all__ = ["Frontend", "IMAGE_PIXELS", "VIDEO_PIXELS"]
