from __future__ import annotations

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from tools.convert.sources.gguf import (
    GGUFSource,
    HFAliasSource,
    qwen35_linear_attention_name_map,
    qwen35_mtp_name_map,
    qwen35_vision_name_map,
    reorder_linear_attention_v_heads,
    standard_dense_name_map,
)


def _write_gguf(path, tensors: dict[str, np.ndarray]) -> None:
    writer = gguf.GGUFWriter(str(path), "test")
    for name, array in tensors.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_source_reads_full_and_partial_ranges(tmp_path):
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
    vector = np.arange(5, dtype=np.float32) - 2.0
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"matrix": matrix, "vector": vector})

    with GGUFSource(path) as source:
        assert source.has("matrix") and not source.has("missing")
        info = source.describe("matrix")
        assert info.shape == (6, 4)
        assert info.dtype == "F32"

        full = source.read_flat("matrix")
        assert torch.equal(full, torch.from_numpy(matrix.reshape(-1)))

        # a partial range spanning part of one row and part of the next
        partial = source.read_flat("matrix", 5, 11)
        assert torch.equal(partial, torch.from_numpy(matrix.reshape(-1)[5:11]))

        # 1D tensor, single-element range (the exact shape recipe.prepare()'s
        # preflight validation uses)
        one = source.read_flat("vector", 0, 1)
        assert torch.equal(one, torch.from_numpy(vector[:1]))

        with pytest.raises(ValueError):
            source.read_flat("matrix", 0, 100)
        with pytest.raises(ValueError):
            source.describe("missing")

    names = dict((n, s) for n, s, _ in GGUFSource(path).list_tensors())
    assert names == {"matrix": (6, 4), "vector": (5,)}


def test_gguf_source_empty_range_returns_empty_tensor(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"vector": np.arange(4, dtype=np.float32)})
    with GGUFSource(path) as source:
        empty = source.read_flat("vector", 2, 2)
        assert empty.numel() == 0


def test_hf_alias_source_name_map_and_missing_name(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"blk.0.attn_q.weight": np.ones((2, 3), dtype=np.float32)})
    with GGUFSource(path) as source:
        alias = HFAliasSource(
            source, {"model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight"}
        )
        assert alias.has("model.layers.0.self_attn.q_proj.weight")
        assert not alias.has("model.layers.0.self_attn.k_proj.weight")
        assert alias.describe("model.layers.0.self_attn.q_proj.weight").shape == (2, 3)
        values = alias.read_flat("model.layers.0.self_attn.q_proj.weight")
        assert torch.equal(values, torch.ones(6))

        with pytest.raises(ValueError, match="no GGUF tensor mapped"):
            alias.read_flat("model.layers.0.self_attn.k_proj.weight")


def test_hf_alias_source_transform_and_shape_override(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"blk.0.attn_norm.weight": np.full(4, 0.5, dtype=np.float32)})
    with GGUFSource(path) as source:

        def shift_up(values, shape):
            return values + 1.0, shape

        alias = HFAliasSource(
            source,
            {"model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight"},
            transforms={"model.layers.0.input_layernorm.weight": shift_up},
            shapes={"model.layers.0.input_layernorm.weight": lambda s: (2, s[0] // 2)},
        )
        values = alias.read_flat("model.layers.0.input_layernorm.weight")
        assert torch.equal(values, torch.full((4,), 1.5))
        # a bounded sub-range of the transformed values, not just the whole tensor
        partial = alias.read_flat("model.layers.0.input_layernorm.weight", 1, 3)
        assert torch.equal(partial, torch.full((2,), 1.5))
        assert alias.describe("model.layers.0.input_layernorm.weight").shape == (2, 2)


def test_hf_alias_source_composite_stitches_two_tensors(tmp_path):
    slice0 = np.arange(6, dtype=np.float32).reshape(2, 3)
    slice1 = (np.arange(6, dtype=np.float32) + 100).reshape(2, 3)
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"v.patch_embd.weight": slice0, "v.patch_embd.weight.1": slice1})

    def composite(source):
        a = source.read_flat("v.patch_embd.weight").reshape(2, 3)
        b = source.read_flat("v.patch_embd.weight.1").reshape(2, 3)
        stacked = torch.stack([a, b], dim=1)  # (2, 2, 3)
        return stacked.reshape(-1), (2, 2, 3)

    with GGUFSource(path) as source:
        alias = HFAliasSource(source, {}, composites={"patch_embed.weight": composite})
        assert alias.has("patch_embed.weight")
        info = alias.describe("patch_embed.weight")
        assert info.shape == (2, 2, 3)
        values = alias.read_flat("patch_embed.weight").reshape(2, 2, 3)
        assert torch.equal(values[:, 0], torch.from_numpy(slice0))
        assert torch.equal(values[:, 1], torch.from_numpy(slice1))


def test_reorder_linear_attention_v_heads_forward_matches_transpose():
    # 2 "K-heads" of 3 "V-heads" each, head_dim 1: grouped-by-K order [k0v0,k0v1,k0v2,k1v0,k1v1,k1v2]
    # should become tiled-by-V order [k0v0,k1v0,k0v1,k1v1,k0v2,k1v2].
    grouped = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    tiled = reorder_linear_attention_v_heads(grouped, (6,), 0, 2, 3, 1)
    assert torch.equal(tiled, torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0, 12.0]))


def test_reorder_linear_attention_v_heads_inverse_needs_swapped_args():
    num_k_heads, num_v_per_k = 2, 3
    original = torch.arange(6, dtype=torch.float32)
    tiled = reorder_linear_attention_v_heads(original, (6,), 0, num_k_heads, num_v_per_k, 1)

    # applying the SAME arguments again does not recover the original ...
    wrong_way = reorder_linear_attention_v_heads(
        tiled, (6,), 0, num_k_heads, num_v_per_k, 1
    )
    assert not torch.equal(wrong_way, original)

    # ... only swapping num_row_groups/group_size does.
    recovered = reorder_linear_attention_v_heads(
        tiled, (6,), 0, num_v_per_k, num_k_heads, 1
    )
    assert torch.equal(recovered, original)


def test_reorder_linear_attention_v_heads_square_case_is_self_inverse():
    # when num_row_groups == group_size, the same call IS its own inverse.
    original = torch.arange(4, dtype=torch.float32)
    once = reorder_linear_attention_v_heads(original, (4,), 0, 2, 2, 1)
    twice = reorder_linear_attention_v_heads(once, (4,), 0, 2, 2, 1)
    assert torch.equal(twice, original)


def test_standard_dense_name_map_shapes_and_norm_shift():
    name_map, transforms, shapes = standard_dense_name_map(
        2, tie_word_embeddings=False, text_prefix="model.language_model."
    )
    assert shapes == {}
    assert name_map["model.language_model.embed_tokens.weight"] == "token_embd.weight"
    assert name_map["lm_head.weight"] == "output.weight"
    assert (
        name_map["model.language_model.layers.0.self_attn.q_proj.weight"]
        == "blk.0.attn_q.weight"
    )
    assert (
        name_map["model.language_model.layers.1.post_attention_layernorm.weight"]
        == "blk.1.post_attention_norm.weight"
    )

    # norm.weight tensors get the +1 shift; projections and embeddings don't.
    for norm_name in (
        "model.language_model.norm.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        "model.language_model.layers.0.self_attn.k_norm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
    ):
        values, shape = transforms[norm_name](torch.zeros(3), (3,))
        assert torch.equal(values, torch.full((3,), -1.0))
    for plain_name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
    ):
        assert plain_name not in transforms


def test_standard_dense_name_map_tied_embeddings_drops_lm_head():
    name_map, _, _ = standard_dense_name_map(1, tie_word_embeddings=True)
    assert "lm_head.weight" not in name_map


def test_qwen35_linear_attention_name_map_names_and_a_log_transform():
    name_map, transforms, shapes = qwen35_linear_attention_name_map(
        0, num_k_heads=2, num_v_heads=6, head_k_dim=4, head_v_dim=4, text_prefix="model."
    )
    prefix = "model.layers.0.linear_attn."
    # conv1d's singleton middle dim (Conv1d's (out, 1, kernel) shape) is the
    # one rank mismatch this map needs to record.
    assert set(shapes) == {prefix + "conv1d.weight"}
    assert shapes[prefix + "conv1d.weight"]((10, 4)) == (10, 1, 4)
    assert name_map[prefix + "in_proj_qkv.weight"] == "blk.0.attn_qkv.weight"
    assert name_map[prefix + "in_proj_z.weight"] == "blk.0.attn_gate.weight"
    assert name_map[prefix + "in_proj_a.weight"] == "blk.0.ssm_alpha.weight"
    assert name_map[prefix + "in_proj_b.weight"] == "blk.0.ssm_beta.weight"
    assert name_map[prefix + "A_log"] == "blk.0.ssm_a"
    assert name_map[prefix + "dt_bias"] == "blk.0.ssm_dt.bias"
    assert name_map[prefix + "norm.weight"] == "blk.0.ssm_norm.weight"
    assert name_map[prefix + "out_proj.weight"] == "blk.0.ssm_out.weight"
    # norm.weight is the GDN's own norm: excluded from the +1 shift elsewhere.
    assert prefix + "norm.weight" not in transforms

    # -exp() undoes the exporter's transform for A_log (num_v_per_k=3 here).
    a_log = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 3.0])
    reordered_negexp = reorder_linear_attention_v_heads(
        -torch.exp(a_log), (6,), 0, 2, 3, 1
    )
    values, _ = transforms[prefix + "A_log"](reordered_negexp, (6,))
    assert torch.allclose(values, a_log, atol=1e-6)


def test_qwen35_linear_attention_name_map_no_transform_when_heads_equal():
    _, transforms, _ = qwen35_linear_attention_name_map(
        0, num_k_heads=4, num_v_heads=4, head_k_dim=4, head_v_dim=4
    )
    assert set(transforms) == {"model.language_model.layers.0.linear_attn.A_log"}


def test_qwen35_mtp_name_map_fused_and_standalone_prefixes():
    fused, _, _ = qwen35_mtp_name_map(64, hf_prefix="mtp.")
    assert fused["mtp.fc.weight"] == "blk.64.nextn.eh_proj.weight"
    assert fused["mtp.norm.weight"] == "blk.64.nextn.shared_head_norm.weight"
    assert (
        fused["mtp.layers.0.self_attn.q_proj.weight"] == "blk.64.attn_q.weight"
    )

    standalone, _, _ = qwen35_mtp_name_map(64, hf_prefix="")
    assert standalone["fc.weight"] == "blk.64.nextn.eh_proj.weight"
    assert standalone["layers.0.mlp.down_proj.weight"] == "blk.64.ffn_down.weight"


def test_qwen35_vision_name_map_has_composite_and_no_deepstack_surprises():
    name_map, transforms, shapes, composites = qwen35_vision_name_map(
        2, hidden_size=8, in_channels=3, temporal_patch_size=2, patch_size=4
    )
    assert transforms == {}
    assert shapes == {}
    assert "model.visual.patch_embed.proj.weight" in composites
    assert (
        name_map["model.visual.blocks.0.attn.qkv.weight"] == "v.blk.0.attn_qkv.weight"
    )
    assert name_map["model.visual.merger.norm.weight"] == "v.post_ln.weight"
    assert name_map["model.visual.merger.linear_fc1.weight"] == "mm.0.weight"

    with pytest.raises(ValueError, match="temporal_patch_size"):
        qwen35_vision_name_map(
            1, hidden_size=8, patch_size=4, temporal_patch_size=3
        )
