"""Observable contracts for the tokenizer embedded in q5090 artifacts."""

from __future__ import annotations

import json

from tokenizers import Tokenizer as HfTokenizer
from tokenizers.models import BPE

from tools.q5090.tokenizer import Tokenizer
from tools.q5090_convert import format as fmt


class StubReader:
    header = {"vocab_size": 4}

    def __init__(self):
        impl = HfTokenizer(
            BPE(
                vocab={"a": 0, "b": 1, "ab": 2, "<eos>": 3},
                merges=[("a", "b")],
            )
        )
        self.assets = {
            fmt.TOKENIZER_JSON: impl.to_str().encode("utf-8"),
            fmt.TOKENIZER_GENERATION_CONFIG: json.dumps(
                {"eos_token_id": [3, 3]}
            ).encode("utf-8"),
        }

    def tokenizer_data(self, kind: int) -> bytes:
        return self.assets[kind]


def test_embedded_tokenizer_encodes_decodes_and_loads_stop_ids() -> None:
    tokenizer = Tokenizer(StubReader())
    assert tokenizer.encode("ab") == [2]
    assert tokenizer.decode([2]) == "ab"
    assert tokenizer.default_stop_token_ids == (3,)
