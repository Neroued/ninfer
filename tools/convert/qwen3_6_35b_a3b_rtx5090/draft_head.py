"""Fixed draft-head shortlist and materialization for Qwen3.6-35B-A3B.

The 27B and 35B checkpoints have the same semantic token-id vocabulary.  The
35B target therefore deliberately uses the existing measured 27B ranking and
gathers the selected rows from its own 2048-wide output head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.convert.qwen3_6.common.draft_head import (
    DraftHeadContext,
    compute_shortlist as _compute_shortlist,
    load_total_counts as _load_total_counts,
    materialize_draft_head,
    materialize_draft_head_token_ids,
    read_special_ids,
    select_shortlist,
)


VOCAB_SIZE = 248320
TOKENIZER_VOCAB_SIZE = 248077
DRAFT_HEAD_N = 131072
DRAFT_HEAD_WIDTH = 2048

DRAFT_HEAD_OBJECT = "text/draft_head"
DRAFT_HEAD_TOKEN_IDS_OBJECT = "text/draft_head_token_ids"
DEFAULT_RANKING = Path(
    "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
)
RANKING_SOURCE_TARGET = "qwen3_6_27b_rtx5090"


def load_total_counts(
    path: str | Path,
    vocab: int = VOCAB_SIZE,
) -> np.ndarray:
    return _load_total_counts(path, vocab)


def compute_shortlist(
    ranking_path: str | Path,
    tokenizer_dir: str | Path,
    n: int = DRAFT_HEAD_N,
    vocab: int = VOCAB_SIZE,
    tokenizer_vocab_size: int | None = None,
) -> DraftHeadContext:
    """Build the fixed shortlist over the shared 27B/35B token-id domain."""

    domain = (
        min(TOKENIZER_VOCAB_SIZE, vocab)
        if tokenizer_vocab_size is None
        else tokenizer_vocab_size
    )
    return _compute_shortlist(
        ranking_path,
        tokenizer_dir,
        n=n,
        vocab=vocab,
        tokenizer_vocab_size=domain,
    )


__all__ = [
    "DEFAULT_RANKING",
    "DRAFT_HEAD_N",
    "DRAFT_HEAD_OBJECT",
    "DRAFT_HEAD_TOKEN_IDS_OBJECT",
    "DRAFT_HEAD_WIDTH",
    "DraftHeadContext",
    "RANKING_SOURCE_TARGET",
    "TOKENIZER_VOCAB_SIZE",
    "VOCAB_SIZE",
    "compute_shortlist",
    "load_total_counts",
    "materialize_draft_head",
    "materialize_draft_head_token_ids",
    "read_special_ids",
    "select_shortlist",
]
