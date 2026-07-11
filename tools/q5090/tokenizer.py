"""Tokenizer backed directly by the assets embedded in a q5090 artifact."""

from __future__ import annotations

import json
from collections.abc import Iterable

from tokenizers import Tokenizer as HfTokenizer

from tools.q5090_convert import format as fmt

from .reader import Reader


class Tokenizer:
    def __init__(self, reader: Reader):
        tokenizer_json = reader.tokenizer_data(fmt.TOKENIZER_JSON)
        generation_json = reader.tokenizer_data(fmt.TOKENIZER_GENERATION_CONFIG)
        try:
            self._impl = HfTokenizer.from_str(tokenizer_json.decode("utf-8"))
            generation = json.loads(generation_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid embedded q5090 tokenizer asset: {exc}") from exc
        eos = generation.get("eos_token_id")
        values = eos if isinstance(eos, list) else [eos]
        if not values or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= reader.header["vocab_size"]
            for value in values
        ):
            raise ValueError("invalid embedded generation_config.json eos_token_id")
        self.default_stop_token_ids = tuple(dict.fromkeys(values))

    def encode(self, text: str) -> list[int]:
        return self._impl.encode(text, add_special_tokens=False).ids

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._impl.decode(list(ids), skip_special_tokens=skip_special_tokens)
