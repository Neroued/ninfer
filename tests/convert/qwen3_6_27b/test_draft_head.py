from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.convert.qwen3_6_27b.draft_head import (
    DraftHeadContext,
    materialize_draft_head,
    materialize_draft_head_token_ids,
    select_shortlist,
)


def _context(selected: list[int]) -> DraftHeadContext:
    return DraftHeadContext(
        n=len(selected),
        selected=np.asarray(selected, dtype=np.int64),
        ranking=Path("ranking.i64"),
        tokenizer=Path("tokenizer"),
        force_include=(),
    )


def test_shortlist_is_stably_frequency_sorted_and_forces_special_ids():
    totals = np.array([9, 5, 9, 5, 0, 0], dtype=np.int64)
    assert select_shortlist(totals, 3).tolist() == [0, 2, 1]

    selected = select_shortlist(totals, 3, force_include=[5])
    assert selected.tolist() == [0, 2, 5]

    forced_only = select_shortlist(totals, 2, force_include=[1, 2, 5])
    assert forced_only.tolist() == [2, 1]


def test_persistent_id_map_preserves_shortlist_row_order():
    context = _context([4, 1, 3])
    token_ids = materialize_draft_head_token_ids(context)
    assert token_ids.dtype == torch.int32
    assert token_ids.is_contiguous()
    assert token_ids.tolist() == [4, 1, 3]


def test_draft_head_rows_follow_the_same_id_mapping():
    full_head = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    context = _context([4, 1, 3])

    draft_head = materialize_draft_head(full_head, context)
    assert draft_head.dtype == torch.bfloat16
    assert draft_head.is_contiguous()
    assert torch.equal(draft_head, full_head[[4, 1, 3]])
