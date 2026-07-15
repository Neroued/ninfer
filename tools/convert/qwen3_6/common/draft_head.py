"""Frequency shortlist and draft-head materialization shared by Qwen3.6 targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class DraftHeadContext:
    """Cached ordered shortlist shared by both derived artifact objects."""

    n: int
    selected: np.ndarray
    ranking: Path
    tokenizer: Path
    force_include: tuple[int, ...]


def load_total_counts(path: str | Path, vocab: int) -> np.ndarray:
    """Load only the total-frequency row from a ``[rows,vocab]`` I64 ranking."""

    ranking = Path(path)
    word_bytes = np.dtype("<i8").itemsize
    file_bytes = ranking.stat().st_size
    row_bytes = vocab * word_bytes
    if file_bytes == 0 or file_bytes % row_bytes:
        raise ValueError(
            f"ranking size {file_bytes} is not a positive multiple of {row_bytes}"
        )
    total = np.fromfile(ranking, dtype="<i8", count=vocab)
    if total.size != vocab:
        raise ValueError(f"ranking total row has {total.size} entries, expected {vocab}")
    return total


def read_special_ids(tokenizer_dir: str | Path) -> tuple[int, ...]:
    """Return sorted special IDs from ``added_tokens_decoder``."""

    config_path = Path(tokenizer_dir) / "tokenizer_config.json"
    if not config_path.is_file():
        return ()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return tuple(
        sorted(
            {
                int(token_id)
                for token_id, metadata in config.get(
                    "added_tokens_decoder", {}
                ).items()
                if isinstance(metadata, dict) and metadata.get("special", False)
            }
        )
    )


def select_shortlist(
    total: np.ndarray,
    n: int,
    force_include: Iterable[int] = (),
) -> np.ndarray:
    """Select exactly ``n`` IDs in stable descending-frequency order."""

    counts = np.asarray(total, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("frequency totals must be one-dimensional")
    vocab = counts.size
    if n <= 0 or n > vocab:
        raise ValueError(f"shortlist size {n} is outside 1..{vocab}")

    order = np.argsort(-counts, kind="stable")
    forced = np.array(
        sorted({int(token_id) for token_id in force_include if 0 <= int(token_id) < vocab}),
        dtype=np.int64,
    )
    if forced.size >= n:
        forced_by_frequency = forced[np.argsort(-counts[forced], kind="stable")]
        return np.ascontiguousarray(forced_by_frequency[:n], dtype=np.int64)

    forced_set = set(forced.tolist())
    wanted = n - forced.size
    picked: list[int] = []
    for token_id in order.tolist():
        if token_id not in forced_set:
            picked.append(token_id)
            if len(picked) == wanted:
                break

    selected = np.concatenate((np.asarray(picked, dtype=np.int64), forced))
    selected = selected[np.argsort(-counts[selected], kind="stable")]
    if selected.size != n or np.unique(selected).size != n:
        raise ValueError("shortlist selection did not produce unique requested rows")
    return np.ascontiguousarray(selected, dtype=np.int64)


def compute_shortlist(
    ranking_path: str | Path,
    tokenizer_dir: str | Path,
    *,
    n: int,
    vocab: int,
    tokenizer_vocab_size: int,
) -> DraftHeadContext:
    """Compute a shortlist constrained to an explicitly supplied tokenizer domain."""

    if tokenizer_vocab_size <= 0 or tokenizer_vocab_size > vocab:
        raise ValueError(
            f"tokenizer domain {tokenizer_vocab_size} is outside 1..{vocab}"
        )
    ranking = Path(ranking_path)
    tokenizer = Path(tokenizer_dir)
    forced = read_special_ids(tokenizer)
    total = load_total_counts(ranking, vocab)
    if np.any(total < 0):
        raise ValueError("draft ranking total counts must be nonnegative")
    if np.any(total[tokenizer_vocab_size:] != 0):
        raise ValueError("draft ranking assigns counts to padded non-token lm_head rows")
    selected = select_shortlist(total[:tokenizer_vocab_size], n, forced)
    return DraftHeadContext(
        n=n,
        selected=selected,
        ranking=ranking,
        tokenizer=tokenizer,
        force_include=forced,
    )


def _selected(context: DraftHeadContext) -> np.ndarray:
    selected = np.asarray(context.selected, dtype=np.int64)
    if selected.ndim != 1 or selected.size != context.n:
        raise ValueError("draft-head context has an inconsistent shortlist")
    return np.ascontiguousarray(selected)


def materialize_draft_head_token_ids(context: DraftHeadContext) -> torch.Tensor:
    """Materialize ``text/draft_head_token_ids`` as contiguous I32 words."""

    return torch.from_numpy(_selected(context)).to(torch.int32)


def materialize_draft_head(
    lm_head: torch.Tensor,
    context: DraftHeadContext,
) -> torch.Tensor:
    """Select full-head rows in exactly the persistent ID-map order."""

    if lm_head.dim() != 2:
        raise ValueError(f"lm_head.weight must be rank two, got {tuple(lm_head.shape)}")
    selected = _selected(context)
    if selected.size and int(selected.max()) >= lm_head.shape[0]:
        raise ValueError("draft shortlist exceeds lm_head.weight rows")
    indices = torch.from_numpy(selected)
    if lm_head.device.type != "cpu":
        indices = indices.to(lm_head.device)
    return lm_head.index_select(0, indices)


__all__ = [
    "DraftHeadContext",
    "compute_shortlist",
    "load_total_counts",
    "materialize_draft_head",
    "materialize_draft_head_token_ids",
    "read_special_ids",
    "select_shortlist",
]
