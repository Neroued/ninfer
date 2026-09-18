from __future__ import annotations

import numpy as np
import torch

from tools.convert.model import Model, Parameter
from tools.convert.proposal import add_proposal
from tools.convert.recipe import Recipe
from tools.convert.resources import token_domain
from tools.convert.sources.logical import array_source


def test_proposal_rows_ids_and_uses_follow_final_token_domain(tmp_path):
    head = torch.arange(8 * 4).reshape(8, 4).to(torch.bfloat16)
    count, special = token_domain(
        {
            "model": {"vocab": {str(i): i for i in range(5)}},
            "added_tokens": [{"id": 5, "content": "<eos>", "special": True}],
        },
        {"added_tokens_decoder": {}},
        vocab_size=8,
    )
    model = Model(
        {
            "text": {"config": {"vocab_size": 8}},
            "mtp": {"config": {}, "target": "text"},
        },
        token_count=count,
        special_token_ids=special,
    )
    model.add(Parameter("text/output_head", (8, 4), array_source(head, "head")))
    ranking = tmp_path / "ranking.i64"
    np.array([[9, 5, 9, 5, 0, 0, 999, 999], [999] * 8], dtype="<i8").tofile(ranking)
    recipe = Recipe(model)
    add_proposal(recipe, ranking=ranking, rows=3)
    ids = model.parameters["proposal/token_ids"].source.values()
    selected = model.parameters["proposal/head"].source.values().reshape(3, 4)
    assert ids.dtype == torch.int32 and ids.tolist() == [0, 2, 5]
    assert torch.equal(selected, head[[0, 2, 5]])
    assert model.parameters["proposal/head"].inputs == ("mtp/final_hidden",)
    assert model.components["text"]["proposal"] == {"domain": "indexed", "rows": 3}
