from __future__ import annotations

import torch

from .. import qtypes as qt
from .. import tensor_plan as tp
from ..convert import _prepare_source
from ..layouts import encode_tensor


class _Reader:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get(self, name: str) -> torch.Tensor:
        assert name == "model.language_model.layers.0.linear_attn.conv1d.weight"
        return self.tensor


def _canonical_layer_types():
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)]


def _block(manifest: tp.ExpectedManifest, name: str) -> tp.TensorBlockSpec:
    matches = [b for b in manifest.blocks if b.name == name]
    assert len(matches) == 1
    return matches[0]


def _segment_ranges(block: tp.TensorBlockSpec):
    return [
        (s.name, s.source_kind, s.row_begin, s.row_begin + s.row_count)
        for s in block.segments
    ]


def _conv1d_spec():
    manifest = tp.build_text_manifest(["linear_attention"])
    return _block(manifest, "layers.0.linear_attn.conv1d.weight").segments[0].source


def test_text_manifest_counts_and_canonical_order():
    manifest = tp.build_text_manifest(_canonical_layer_types())

    assert len(manifest.blocks) == 819
    assert len(manifest.segments) == 963
    assert len(manifest.fusion_groups) == 128

    assert manifest.blocks[0].name == "model.language_model.embed_tokens.weight"
    assert [b.name for b in manifest.blocks[1:15]] == [
        "layers.0.input_layernorm.weight",
        "layers.0.linear_attn.A_log",
        "layers.0.linear_attn.dt_bias",
        "layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attn.in_proj_b.weight",
        "layers.0.gdn_in.q4",
        "layers.0.gdn_in.q5",
        "layers.0.linear_attn.norm.weight",
        "layers.0.linear_attn.in_proj_z.weight",
        "layers.0.linear_attn.out_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.mlp.gateup",
        "layers.0.mlp.down_proj.weight",
    ]
    full_l3_begin = 1 + 3 * 14
    assert [b.name for b in manifest.blocks[full_l3_begin: full_l3_begin + 9]] == [
        "layers.3.input_layernorm.weight",
        "layers.3.attn_in.q4",
        "layers.3.attn_in.q5",
        "layers.3.self_attn.q_norm.weight",
        "layers.3.self_attn.k_norm.weight",
        "layers.3.self_attn.o_proj.weight",
        "layers.3.post_attention_layernorm.weight",
        "layers.3.mlp.gateup",
        "layers.3.mlp.down_proj.weight",
    ]
    assert [b.name for b in manifest.blocks[-2:]] == [
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]


def test_full_attention_template_blocks_segments_and_fusions():
    manifest = tp.build_text_manifest(_canonical_layer_types())

    q4 = _block(manifest, "layers.3.attn_in.q4")
    assert q4.qtype == qt.QT_Q4G64
    assert q4.layout == qt.LAYOUT_ROW_SPLIT
    assert q4.shape == (7168, 5120)
    assert q4.fusion_group_id == qt.FUSION_ATTN_IN
    assert q4.fusion_index == 0
    assert q4.source_kind == qt.SK_OTHER
    assert _segment_ranges(q4) == [
        ("layers.3.self_attn.q_proj.q", qt.SK_ATTN_Q, 0, 6144),
        ("layers.3.self_attn.k_proj.weight", qt.SK_ATTN_K, 6144, 7168),
    ]

    q5 = _block(manifest, "layers.3.attn_in.q5")
    assert q5.qtype == qt.QT_Q5G64
    assert q5.shape == (7168, 5120)
    assert q5.fusion_group_id == qt.FUSION_ATTN_IN
    assert q5.fusion_index == 1
    assert q5.source_kind == qt.SK_OTHER
    assert _segment_ranges(q5) == [
        ("layers.3.self_attn.q_proj.gate", qt.SK_ATTN_GATE, 0, 6144),
        ("layers.3.self_attn.v_proj.weight", qt.SK_ATTN_V, 6144, 7168),
    ]

    q_norm = _block(manifest, "layers.3.self_attn.q_norm.weight")
    assert q_norm.qtype == qt.QT_BF16
    assert q_norm.layout == qt.LAYOUT_CONTIGUOUS
    assert q_norm.shape == (256,)

    o_proj = _block(manifest, "layers.3.self_attn.o_proj.weight")
    assert o_proj.qtype == qt.QT_Q5G64
    assert o_proj.shape == (5120, 6144)
    assert o_proj.fusion_group_id == qt.FUSION_NONE
    assert _segment_ranges(o_proj) == [
        ("layers.3.self_attn.o_proj.weight", qt.SK_ATTN_O, 0, 5120),
    ]

    gateup = _block(manifest, "layers.3.mlp.gateup")
    assert gateup.qtype == qt.QT_Q4G64
    assert gateup.shape == (34816, 5120)
    assert gateup.fusion_group_id == qt.FUSION_MLP_GATEUP
    assert gateup.fusion_index == 0
    assert gateup.source_kind == qt.SK_OTHER
    assert _segment_ranges(gateup) == [
        ("layers.3.mlp.gate_proj.weight", qt.SK_MLP_GATE, 0, 17408),
        ("layers.3.mlp.up_proj.weight", qt.SK_MLP_UP, 17408, 34816),
    ]

    down = _block(manifest, "layers.3.mlp.down_proj.weight")
    assert down.qtype == qt.QT_Q5G64
    assert down.shape == (5120, 17408)


def test_gdn_template_blocks_segments_and_fusions():
    manifest = tp.build_text_manifest(_canonical_layer_types())

    conv = _block(manifest, "layers.0.linear_attn.conv1d.weight")
    assert conv.qtype == qt.QT_BF16
    assert conv.layout == qt.LAYOUT_CONTIGUOUS
    assert conv.shape == (10240, 4, 1)

    a_log = _block(manifest, "layers.0.linear_attn.A_log")
    assert a_log.qtype == qt.QT_FP32
    assert a_log.shape == (48,)

    q4 = _block(manifest, "layers.0.gdn_in.q4")
    assert q4.qtype == qt.QT_Q4G64
    assert q4.shape == (4096, 5120)
    assert q4.fusion_group_id == qt.FUSION_GDN_IN
    assert q4.fusion_index == 0
    assert q4.source_kind == qt.SK_OTHER
    assert _segment_ranges(q4) == [
        ("layers.0.linear_attn.in_proj_qkv.q", qt.SK_GDN_IN_PROJ_Q, 0, 2048),
        ("layers.0.linear_attn.in_proj_qkv.k", qt.SK_GDN_IN_PROJ_K, 2048, 4096),
    ]

    q5 = _block(manifest, "layers.0.gdn_in.q5")
    assert q5.qtype == qt.QT_Q5G64
    assert q5.shape == (6144, 5120)
    assert q5.fusion_group_id == qt.FUSION_GDN_IN
    assert q5.fusion_index == 1
    assert q5.source_kind == qt.SK_GDN_IN_PROJ_V
    assert _segment_ranges(q5) == [
        ("layers.0.linear_attn.in_proj_qkv.v", qt.SK_GDN_IN_PROJ_V, 0, 6144),
    ]

    in_z = _block(manifest, "layers.0.linear_attn.in_proj_z.weight")
    assert in_z.qtype == qt.QT_Q5G64
    assert in_z.shape == (6144, 5120)
    assert in_z.fusion_group_id == qt.FUSION_NONE

    out = _block(manifest, "layers.0.linear_attn.out_proj.weight")
    assert out.qtype == qt.QT_Q5G64
    assert out.shape == (5120, 6144)


def test_globals_are_standalone_text_blocks():
    manifest = tp.build_text_manifest(_canonical_layer_types())

    embed = manifest.blocks[0]
    assert embed.name == "model.language_model.embed_tokens.weight"
    assert embed.qtype == qt.QT_Q6G64
    assert embed.layout == qt.LAYOUT_ROW_SPLIT
    assert embed.shape == (248320, 5120)
    assert _segment_ranges(embed) == [
        ("model.language_model.embed_tokens.weight", qt.SK_EMBED, 0, 248320),
    ]

    final_norm = manifest.blocks[-2]
    assert final_norm.name == "model.language_model.norm.weight"
    assert final_norm.qtype == qt.QT_BF16
    assert final_norm.layout == qt.LAYOUT_CONTIGUOUS
    assert final_norm.shape == (5120,)

    lm_head = manifest.blocks[-1]
    assert lm_head.name == "lm_head.weight"
    assert lm_head.qtype == qt.QT_Q6G64
    assert lm_head.layout == qt.LAYOUT_ROW_SPLIT
    assert lm_head.shape == (248320, 5120)


def test_segments_partition_blocks_and_source_kind_rule():
    manifest = tp.build_text_manifest(_canonical_layer_types())

    for block in manifest.blocks:
        assert block.segment_count == len(block.segments)
        assert manifest.segments[block.segment_begin: block.segment_begin + block.segment_count] == block.segments
        row = 0
        for segment in block.segments:
            assert segment.block_index == block.block_index
            assert segment.row_begin == row
            row += segment.row_count
        assert row == block.shape[0]
        if block.segment_count == 1:
            assert block.source_kind == block.segments[0].source_kind
        else:
            assert block.source_kind == qt.SK_OTHER


def test_fusion_members_are_consecutive_with_correct_indices():
    manifest = tp.build_text_manifest(_canonical_layer_types())

    for group in manifest.fusion_groups:
        assert group.block_indices == tuple(range(group.first_block_index, group.first_block_index + group.block_count))
        assert group.total_n == sum(manifest.blocks[i].shape[0] for i in group.block_indices)
        for fusion_index, block_index in enumerate(group.block_indices):
            block = manifest.blocks[block_index]
            assert block.fusion_group_id == group.group_id
            assert block.fusion_index == fusion_index
            assert block.source_layer == group.source_layer
            assert block.shape[1] == group.shared_k


def test_source_specs_carry_value_defining_transforms():
    full = tp.build_text_manifest(["full_attention"])
    attn_q4 = _block(full, "layers.0.attn_in.q4")
    attn_q5 = _block(full, "layers.0.attn_in.q5")
    assert attn_q4.segments[0].source.src_name == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert attn_q4.segments[0].source.transform == tp.TRANSFORM_ATTN_QPROJ_QUERY
    assert attn_q4.segments[1].source.src_name == "model.language_model.layers.0.self_attn.k_proj.weight"
    assert attn_q4.segments[1].source.transform is None
    assert attn_q5.segments[0].source.src_name == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert attn_q5.segments[0].source.transform == tp.TRANSFORM_ATTN_QPROJ_GATE

    gdn = tp.build_text_manifest(["linear_attention"])
    gdn_q4 = _block(gdn, "layers.0.gdn_in.q4")
    gdn_q5 = _block(gdn, "layers.0.gdn_in.q5")
    assert gdn_q4.segments[0].source.row_slice == (0, 2048)
    assert gdn_q4.segments[1].source.row_slice == (2048, 4096)
    assert gdn_q5.segments[0].source.row_slice == (4096, 10240)

    conv = _block(gdn, "layers.0.linear_attn.conv1d.weight")
    assert conv.segments[0].source.transform == tp.TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE


def test_gdn_conv1d_spec_uses_runtime_native_transform():
    spec = _conv1d_spec()
    assert spec.qtype == qt.QT_BF16
    assert spec.layout == qt.LAYOUT_CONTIGUOUS
    assert spec.source_kind == qt.SK_GDN_CONV1D
    assert spec.source_layer == 0
    assert spec.transform == tp.TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE


def test_runtime_native_gdn_conv1d_reorders_payload_bytes():
    src = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[10.0, 11.0, 12.0, 13.0]],
            [[20.0, 21.0, 22.0, 23.0]],
        ],
        dtype=torch.float32,
    ).to(torch.bfloat16)

    got = tp.runtime_native_gdn_conv1d(src)

    assert tuple(got.shape) == (3, 4, 1)
    assert got.reshape(-1).float().tolist() == [
        0.0,
        10.0,
        20.0,
        1.0,
        11.0,
        21.0,
        2.0,
        12.0,
        22.0,
        3.0,
        13.0,
        23.0,
    ]


def test_prepare_source_applies_gdn_conv1d_transform_before_encoding():
    src = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[10.0, 11.0, 12.0, 13.0]],
            [[20.0, 21.0, 22.0, 23.0]],
        ],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    spec = _conv1d_spec()

    prepared = _prepare_source(_Reader(src), spec)
    payload, logical, padded, group, scale_dtype, code_plane_bytes, scale_plane_bytes = encode_tensor(
        prepared, spec.qtype, spec.layout, torch.device("cpu")
    )

    assert logical == [3, 4, 1]
    assert padded == [3, 4, 1]
    assert group == 0
    assert scale_dtype == qt.SCALE_NONE
    assert code_plane_bytes == len(payload)
    assert scale_plane_bytes == 0
    decoded_words = torch.frombuffer(bytearray(payload), dtype=torch.int16).to(torch.int32)
    expected_words = prepared.reshape(-1).view(torch.int16).to(torch.int32)
    assert decoded_words.tolist() == expected_words.tolist()


def test_runtime_native_gdn_conv1d_rejects_unexpected_shape():
    bad = torch.zeros((3, 4), dtype=torch.bfloat16)
    try:
        tp.runtime_native_gdn_conv1d(bad)
    except ValueError as exc:
        assert "expected [C,1,K]" in str(exc)
    else:
        raise AssertionError("runtime_native_gdn_conv1d accepted a rank-2 tensor")
