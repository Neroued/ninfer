"""Declarative tensor plan: source safetensors name -> (qtype, layout, slice/reshape).

Mirrors the policy tables in ../../docs/qwen3_6_27b_q5090_final_quant_format_v1.md
sections 3, 5, 6, 10. The converter walks these specs in order (TEXT, then MTP, then
VISION) so each module is a contiguous range of the tensor index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import qtypes as qt


@dataclass
class TensorSpec:
    name: str                      # canonical output name
    qtype: int
    layout: int
    module_kind: int
    src_name: str                  # safetensors source key
    source_kind: int
    source_layer: int = qt.NO_LAYER
    row_slice: Optional[Tuple[int, int]] = None   # take src.weight[start:end] rows
    reshape: Optional[Tuple[int, ...]] = None      # reshape source before encoding


# GDN fused in_proj_qkv row splits: q=16*128, k=16*128, v=48*128 -> [10240, 5120]
_GDN_Q = (0, 2048)
_GDN_K = (2048, 4096)
_GDN_V = (4096, 10240)
# Full-attn fused q_proj row splits: q=24*256, gate=24*256 -> [12288, 5120]
_ATTN_Q = (0, 6144)
_ATTN_GATE = (6144, 12288)


def _bf16(name, src, sk, layer=qt.NO_LAYER, module=qt.MODULE_TEXT) -> TensorSpec:
    return TensorSpec(name, qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, module, src, sk, layer)


def _fp32(name, src, sk, layer, module=qt.MODULE_TEXT) -> TensorSpec:
    return TensorSpec(name, qt.QT_FP32, qt.LAYOUT_CONTIGUOUS, module, src, sk, layer)


def _tile(name, qtype, src, sk, layer, module=qt.MODULE_TEXT, row_slice=None) -> TensorSpec:
    return TensorSpec(name, qtype, qt.LAYOUT_TILE_N64_K64, module, src, sk, layer, row_slice)


def _w8(name, src, sk, layer, module) -> TensorSpec:
    return TensorSpec(name, qt.QT_W8G128, qt.LAYOUT_TILE_N64_K128, module, src, sk, layer)


def build_text_specs(layer_types: List[str]) -> List[TensorSpec]:
    s: List[TensorSpec] = []
    # globals
    s.append(TensorSpec("model.language_model.embed_tokens.weight", qt.QT_Q6G64,
                        qt.LAYOUT_ROW_GROUPED_G64, qt.MODULE_TEXT,
                        "model.language_model.embed_tokens.weight", qt.SK_EMBED))
    for li, lt in enumerate(layer_types):
        p = f"model.language_model.layers.{li}."
        s.append(_bf16(p + "input_layernorm.weight", p + "input_layernorm.weight",
                       qt.SK_INPUT_LAYERNORM, li))
        if lt == "linear_attention":
            qkv = p + "linear_attn.in_proj_qkv.weight"
            s += [
                _fp32(p + "linear_attn.A_log", p + "linear_attn.A_log", qt.SK_GDN_A_LOG, li),
                _fp32(p + "linear_attn.dt_bias", p + "linear_attn.dt_bias", qt.SK_GDN_DT_BIAS, li),
                _bf16(p + "linear_attn.conv1d.weight", p + "linear_attn.conv1d.weight", qt.SK_GDN_CONV1D, li),
                _bf16(p + "linear_attn.in_proj_a.weight", p + "linear_attn.in_proj_a.weight", qt.SK_GDN_IN_PROJ_A, li),
                _bf16(p + "linear_attn.in_proj_b.weight", p + "linear_attn.in_proj_b.weight", qt.SK_GDN_IN_PROJ_B, li),
                _tile(p + "linear_attn.in_proj_qkv.q", qt.QT_Q4G64, qkv, qt.SK_GDN_IN_PROJ_Q, li, row_slice=_GDN_Q),
                _tile(p + "linear_attn.in_proj_qkv.k", qt.QT_Q4G64, qkv, qt.SK_GDN_IN_PROJ_K, li, row_slice=_GDN_K),
                _tile(p + "linear_attn.in_proj_qkv.v", qt.QT_Q5G64, qkv, qt.SK_GDN_IN_PROJ_V, li, row_slice=_GDN_V),
                _tile(p + "linear_attn.in_proj_z.weight", qt.QT_Q5G64, p + "linear_attn.in_proj_z.weight", qt.SK_GDN_IN_PROJ_Z, li),
                _bf16(p + "linear_attn.norm.weight", p + "linear_attn.norm.weight", qt.SK_GDN_NORM, li),
                _tile(p + "linear_attn.out_proj.weight", qt.QT_Q5G64, p + "linear_attn.out_proj.weight", qt.SK_GDN_OUT_PROJ, li),
            ]
        elif lt == "full_attention":
            qp = p + "self_attn.q_proj.weight"
            s += [
                _tile(p + "self_attn.q_proj.q", qt.QT_Q4G64, qp, qt.SK_ATTN_Q, li, row_slice=_ATTN_Q),
                _tile(p + "self_attn.q_proj.gate", qt.QT_Q5G64, qp, qt.SK_ATTN_GATE, li, row_slice=_ATTN_GATE),
                _tile(p + "self_attn.k_proj.weight", qt.QT_Q4G64, p + "self_attn.k_proj.weight", qt.SK_ATTN_K, li),
                _tile(p + "self_attn.v_proj.weight", qt.QT_Q5G64, p + "self_attn.v_proj.weight", qt.SK_ATTN_V, li),
                _bf16(p + "self_attn.q_norm.weight", p + "self_attn.q_norm.weight", qt.SK_ATTN_Q_NORM, li),
                _bf16(p + "self_attn.k_norm.weight", p + "self_attn.k_norm.weight", qt.SK_ATTN_K_NORM, li),
                _tile(p + "self_attn.o_proj.weight", qt.QT_Q5G64, p + "self_attn.o_proj.weight", qt.SK_ATTN_O, li),
            ]
        else:
            raise ValueError(f"layer {li}: unknown layer_type {lt!r}")
        s += [
            _bf16(p + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight", qt.SK_POST_ATTN_LAYERNORM, li),
            _tile(p + "mlp.gate_proj.weight", qt.QT_Q4G64, p + "mlp.gate_proj.weight", qt.SK_MLP_GATE, li),
            _tile(p + "mlp.up_proj.weight", qt.QT_Q4G64, p + "mlp.up_proj.weight", qt.SK_MLP_UP, li),
            _tile(p + "mlp.down_proj.weight", qt.QT_Q5G64, p + "mlp.down_proj.weight", qt.SK_MLP_DOWN, li),
        ]
    # final globals
    s.append(_bf16("model.language_model.norm.weight", "model.language_model.norm.weight", qt.SK_FINAL_NORM))
    s.append(_tile("lm_head.weight", qt.QT_Q6G64, "lm_head.weight", qt.SK_LM_HEAD, qt.NO_LAYER))
    return s


def build_mtp_specs() -> List[TensorSpec]:
    m = qt.MODULE_MTP
    p = "mtp.layers.0."
    return [
        _w8("mtp.fc.weight", "mtp.fc.weight", qt.SK_MTP_FC, qt.NO_LAYER, m),
        _bf16("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding.weight", qt.SK_MTP_PRE_FC_NORM_EMB, qt.NO_LAYER, m),
        _bf16("mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight", qt.SK_MTP_PRE_FC_NORM_HID, qt.NO_LAYER, m),
        _bf16(p + "input_layernorm.weight", p + "input_layernorm.weight", qt.SK_INPUT_LAYERNORM, 0, m),
        _w8(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.weight", qt.SK_ATTN_Q, 0, m),
        _w8(p + "self_attn.k_proj.weight", p + "self_attn.k_proj.weight", qt.SK_ATTN_K, 0, m),
        _w8(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.weight", qt.SK_ATTN_V, 0, m),
        _w8(p + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight", qt.SK_ATTN_O, 0, m),
        _bf16(p + "self_attn.q_norm.weight", p + "self_attn.q_norm.weight", qt.SK_ATTN_Q_NORM, 0, m),
        _bf16(p + "self_attn.k_norm.weight", p + "self_attn.k_norm.weight", qt.SK_ATTN_K_NORM, 0, m),
        _bf16(p + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight", qt.SK_POST_ATTN_LAYERNORM, 0, m),
        _w8(p + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight", qt.SK_MLP_GATE, 0, m),
        _w8(p + "mlp.up_proj.weight", p + "mlp.up_proj.weight", qt.SK_MLP_UP, 0, m),
        _w8(p + "mlp.down_proj.weight", p + "mlp.down_proj.weight", qt.SK_MLP_DOWN, 0, m),
        _bf16("mtp.norm.weight", "mtp.norm.weight", qt.SK_MTP_NORM, qt.NO_LAYER, m),
    ]


def build_vision_specs(depth: int) -> List[TensorSpec]:
    m = qt.MODULE_VISION

    def vtile(name, qtype, sk, layer):
        return TensorSpec("model.visual." + name, qtype, qt.LAYOUT_TILE_N64_K64, m,
                          "model.visual." + name, sk, layer)

    def vbf16(name, sk, layer=qt.NO_LAYER):
        return _bf16("model.visual." + name, "model.visual." + name, sk, layer, m)

    s: List[TensorSpec] = []
    # patch embed: Conv3d [1152,3,2,16,16] -> [1152, 1536]
    s.append(TensorSpec("model.visual.patch_embed.proj.weight", qt.QT_Q5G64, qt.LAYOUT_TILE_N64_K64, m,
                        "model.visual.patch_embed.proj.weight", qt.SK_VIS_PATCH_EMBED, qt.NO_LAYER,
                        reshape=(1152, 1536)))
    s.append(vbf16("patch_embed.proj.bias", qt.SK_VIS_PATCH_EMBED_BIAS))
    s.append(vbf16("pos_embed.weight", qt.SK_VIS_POS_EMBED))
    for b in range(depth):
        p = f"blocks.{b}."
        s += [
            vtile(p + "attn.qkv.weight", qt.QT_Q4G64, qt.SK_VIS_BLOCK_QKV, b),
            vbf16(p + "attn.qkv.bias", qt.SK_VIS_BLOCK_QKV_BIAS, b),
            vtile(p + "attn.proj.weight", qt.QT_Q5G64, qt.SK_VIS_BLOCK_PROJ, b),
            vbf16(p + "attn.proj.bias", qt.SK_VIS_BLOCK_PROJ_BIAS, b),
            vtile(p + "mlp.linear_fc1.weight", qt.QT_Q4G64, qt.SK_VIS_BLOCK_FC1, b),
            vbf16(p + "mlp.linear_fc1.bias", qt.SK_VIS_BLOCK_FC1_BIAS, b),
            vtile(p + "mlp.linear_fc2.weight", qt.QT_Q5G64, qt.SK_VIS_BLOCK_FC2, b),
            vbf16(p + "mlp.linear_fc2.bias", qt.SK_VIS_BLOCK_FC2_BIAS, b),
            vbf16(p + "norm1.weight", qt.SK_VIS_BLOCK_NORM1_W, b),
            vbf16(p + "norm1.bias", qt.SK_VIS_BLOCK_NORM1_B, b),
            vbf16(p + "norm2.weight", qt.SK_VIS_BLOCK_NORM2_W, b),
            vbf16(p + "norm2.bias", qt.SK_VIS_BLOCK_NORM2_B, b),
        ]
    s += [
        _w8("model.visual.merger.linear_fc1.weight", "model.visual.merger.linear_fc1.weight", qt.SK_VIS_MERGER_FC1, qt.NO_LAYER, m),
        vbf16("merger.linear_fc1.bias", qt.SK_VIS_MERGER_FC1_BIAS),
        _w8("model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.weight", qt.SK_VIS_MERGER_FC2, qt.NO_LAYER, m),
        vbf16("merger.linear_fc2.bias", qt.SK_VIS_MERGER_FC2_BIAS),
        vbf16("merger.norm.weight", qt.SK_VIS_MERGER_NORM_W),
        vbf16("merger.norm.bias", qt.SK_VIS_MERGER_NORM_B),
    ]
    return s
