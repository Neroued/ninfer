from pathlib import Path
from types import SimpleNamespace

import torch

from tools.convert.model import Model
from tools.convert.qwen4_exp import Builder, bank_source


class Checkpoint:
    path = Path("memory")

    def __init__(self, tensors):
        self.tensors = tensors

    def describe(self, name):
        return SimpleNamespace(shape=tuple(self.tensors[name].shape), dtype="BF16")

    def read_flat(self, name, begin=0, end=None):
        return self.tensors[name].flatten()[begin:end]


def test_qsa_q_gate_rows_and_gdn_conv_axes():
    cfg = dict(
        hidden_size=2,
        head_dim=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        indexer_n_heads=2,
        indexer_head_dim=2,
        indexer_kv_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
    )
    shapes = {
        "self_attn.q_proj.weight": (12, 2),
        "self_attn.k_proj.weight": (2, 2),
        "self_attn.v_proj.weight": (2, 2),
        "self_attn.o_proj.weight": (2, 6),
        "self_attn.indexer.index_qk_proj.weight": (6, 2),
        "self_attn.q_norm.weight": (2,),
        "self_attn.k_norm.weight": (2,),
        "self_attn.indexer.q_layernorm.weight": (2,),
        "self_attn.indexer.k_layernorm.weight": (2,),
        "linear_attn.in_proj_qkv.weight": (8, 2),
        "linear_attn.in_proj_z.weight": (4, 2),
        "linear_attn.in_proj_a.weight": (2, 2),
        "linear_attn.in_proj_b.weight": (2, 2),
        "linear_attn.out_proj.weight": (2, 4),
        "linear_attn.conv1d.weight": (8, 1, 4),
        "linear_attn.A_log": (2,),
        "linear_attn.dt_bias": (2,),
        "linear_attn.norm.weight": (2,),
    }
    from math import prod

    store = Checkpoint(
        {
            key: torch.arange(prod(shape)).bfloat16().reshape(shape)
            for key, shape in shapes.items()
        }
    )
    model = Model({"text": {"config": cfg}})
    builder = Builder(model, store, cfg)
    builder.qsa("text/", "")
    builder.qsa("mtp/", "", mtp=True)
    builder.gdn("text/", "")
    query = model.parameters["text/attention/query"].source.values().reshape(-1, 2)
    gate = model.parameters["text/attention/output_gate"].source.values().reshape(-1, 2)
    assert query[:, 0].tolist() == [0, 2, 8, 10, 16, 18]
    assert gate[:, 0].tolist() == [4, 6, 12, 14, 20, 22]
    mtp_order = ("key", "value", "index_key", "query", "output_gate", "index_query")
    assert tuple("mtp/attention/" + role for role in mtp_order) in model.packing_groups
    conv = model.parameters["text/gdn/convolution"].source.values().reshape(4, 8)
    assert torch.equal(conv, store.tensors["linear_attn.conv1d.weight"][:, 0, :].T)
    assert model.parameters["text/gdn/a_log"].direct_format == "fp32"
    assert model.parameters["text/gdn/dt_bias"].direct_format == "fp32"
    replacement = Checkpoint({name: values + 1 for name, values in store.tensors.items()})
    assert torch.equal(
        model.source("text/attention/query", replacement).values().reshape(-1, 2),
        query + 1,
    )
    assert torch.equal(
        model.source("text/gdn/convolution", replacement).values().reshape(4, 8),
        conv + 1,
    )


def test_expert_source_splits_each_expert_gate_up_before_joining_bank():
    values = torch.arange(48).bfloat16().reshape(3, 4, 4)
    store = Checkpoint({"gate_up": values})
    for role, begin in (("gate", 0), ("up", 2)):
        source = bank_source(store, "gate_up", (3, 2, 4), role)
        expected = values[:, begin : begin + 2].contiguous().flatten()
        assert torch.equal(source.values(), expected)
        assert torch.equal(source.values(5, 17), expected[5:17])
