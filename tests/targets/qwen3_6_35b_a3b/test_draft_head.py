from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert.qwen3_6_35b_a3b_rtx5090 import draft_head


def _context(selected: list[int]) -> draft_head.DraftHeadContext:
    return draft_head.DraftHeadContext(
        n=len(selected),
        selected=np.asarray(selected, dtype=np.int64),
        ranking=Path("ranking.i64"),
        tokenizer=Path("tokenizer"),
        force_include=(),
    )


def test_35b_uses_the_measured_27b_ranking() -> None:
    assert draft_head.VOCAB_SIZE == 248320
    assert draft_head.TOKENIZER_VOCAB_SIZE == 248077
    assert draft_head.DRAFT_HEAD_N == 131072
    assert draft_head.DRAFT_HEAD_WIDTH == 2048
    assert draft_head.DEFAULT_RANKING == Path(
        "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    )
    assert draft_head.RANKING_SOURCE_TARGET == "qwen3_6_27b_rtx5090"


def test_shortlist_uses_stable_frequency_order_and_forces_special_ids(
    tmp_path: Path,
) -> None:
    ranking = tmp_path / "counts.i64"
    np.asarray([9, 5, 9, 5, 0, 0], dtype="<i8").tofile(ranking)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "4": {"content": "ordinary", "special": False},
                    "5": {"content": "forced", "special": True},
                }
            }
        ),
        encoding="utf-8",
    )

    context = draft_head.compute_shortlist(
        ranking,
        tokenizer,
        n=3,
        vocab=6,
        tokenizer_vocab_size=6,
    )
    assert context.selected.tolist() == [0, 2, 5]
    assert context.force_include == (5,)


def test_shortlist_excludes_padded_output_head_rows(tmp_path: Path) -> None:
    ranking = tmp_path / "counts.i64"
    np.asarray([9, 5, 4, 3, 0, 1], dtype="<i8").tofile(ranking)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="padded non-token"):
        draft_head.compute_shortlist(
            ranking,
            tokenizer,
            n=3,
            vocab=6,
            tokenizer_vocab_size=5,
        )


def test_draft_rows_and_persistent_id_map_share_one_order() -> None:
    context = _context([4, 1, 3])
    full_head = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    full_head = full_head.to(torch.bfloat16)

    token_ids = draft_head.materialize_draft_head_token_ids(context)
    rows = draft_head.materialize_draft_head(full_head, context)

    assert token_ids.dtype == torch.int32
    assert token_ids.tolist() == [4, 1, 3]
    assert rows.dtype == torch.bfloat16
    assert torch.equal(rows, full_head[[4, 1, 3]])
