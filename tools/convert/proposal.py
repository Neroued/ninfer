"""The existing frequency shortlist algorithm, using the final tokenizer resources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import Parameter
from .recipe import Recipe
from .sources.logical import array_source, gather_source

DEFAULT_RANKING = (
    Path(__file__).resolve().parents[1]
    / "freq_corpus/fixtures/ranking/ranking.train.counts.i64"
)


def select_shortlist(total: np.ndarray, n: int, force_include=()) -> np.ndarray:
    counts = np.asarray(total, dtype=np.int64)
    if counts.ndim != 1 or not 0 < n <= counts.size:
        raise ValueError("shortlist size is outside the frequency domain")
    order = np.argsort(-counts, kind="stable")
    forced = np.array(
        sorted({int(i) for i in force_include if 0 <= int(i) < counts.size}),
        dtype=np.int64,
    )
    if forced.size >= n:
        return np.ascontiguousarray(
            forced[np.argsort(-counts[forced], kind="stable")][:n]
        )
    forced_set = set(forced.tolist())
    wanted = n - forced.size
    picked = []
    for token_id in order.tolist():
        if token_id not in forced_set:
            picked.append(token_id)
            if len(picked) == wanted:
                break
    selected = np.concatenate((np.asarray(picked, dtype=np.int64), forced))
    return np.ascontiguousarray(selected[np.argsort(-counts[selected], kind="stable")])


def add_proposal(
    recipe: Recipe,
    *,
    ranking: str | Path = DEFAULT_RANKING,
    rows: int = 131072,
    source=None,
    format="q4_g64_fp16",
) -> None:
    model = recipe.model
    if "proposal/head" in model.parameters:
        raise ValueError("proposal head already configured")
    vocab = model.config["vocab_size"]
    ranking = Path(ranking)
    if ranking.stat().st_size == 0 or ranking.stat().st_size % (vocab * 8):
        raise ValueError("ranking must contain complete int64 vocabulary rows")
    total = np.fromfile(ranking, dtype="<i8", count=vocab)
    selected = select_shortlist(
        total[: model.token_count], rows, model.special_token_ids
    )
    ids = torch.from_numpy(selected).to(torch.int32)
    if source is None:
        selections = recipe.selections["text/output_head"]
        if len(selections) != 1:
            raise ValueError(
                "provide a proposal source when the main head uses split sources"
            )
        source = selections[0].source
    head = gather_source(source, ids)
    inputs = tuple(
        component + "/final_hidden"
        for component in ("mtp", "dflash", "dflash2")
        if component in model.components
    )
    model.components["text"]["proposal"] = {"domain": "indexed", "rows": rows}
    model.add(
        Parameter(
            "proposal/head", head.shape, head, inputs=inputs, residency="proposal"
        )
    )
    model.add(
        Parameter(
            "proposal/token_ids",
            (rows,),
            array_source(ids, f"shortlist({ranking})"),
            direct_format="int32",
            residency="proposal",
        )
    )
    recipe.add_parameter("proposal/head")
    recipe.add_parameter("proposal/token_ids")
    recipe.assign("proposal/head", format=format, method="grouped_absmax")


def add_official_proposal(
    recipe: Recipe, *, ranking=DEFAULT_RANKING, rows=131072
) -> None:
    add_proposal(
        recipe,
        ranking=ranking,
        rows=rows,
        source=recipe.model.parameters["text/output_head"].source,
    )
