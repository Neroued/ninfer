from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6_35b_a3b import draft_head


def test_35b_uses_the_measured_27b_ranking() -> None:
    assert draft_head.VOCAB_SIZE == 248320
    assert draft_head.TOKENIZER_VOCAB_SIZE == 248077
    assert draft_head.DRAFT_HEAD_N == 131072
    assert draft_head.DEFAULT_RANKING == Path(
        "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    )
    assert draft_head.RANKING_SOURCE_TARGET == "qwen3_6_27b"
