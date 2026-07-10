"""Enums and per-qtype constants for the q5090_w4g64_mixed_v4_1 artifact.

Values here are the on-disk ABI; do not renumber. See
../../docs/q5090_packed_file_format_v4.md sections 7-9.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- qtype ids (TensorEntry.qtype) ---
QT_Q4G64 = 0
QT_Q5G64 = 1
QT_Q6G64 = 2
QT_W8G128 = 3
QT_BF16 = 4
QT_FP32 = 5
QT_W8G32 = 6
QT_I32 = 7

QTYPE_NAME = {
    QT_Q4G64: "Q4G64_F16S",
    QT_Q5G64: "Q5G64_F16S",
    QT_Q6G64: "Q6G64_F16S",
    QT_W8G128: "W8G128_F16S",
    QT_BF16: "BF16_CTRL",
    QT_FP32: "FP32_CTRL",
    QT_W8G32: "W8G32_F16S",
    QT_I32: "I32_CTRL",
}

# --- layout ids (TensorEntry.layout) ---
LAYOUT_ROW_SPLIT = 0
LAYOUT_CONTIGUOUS = 1

LAYOUT_NAME = {
    LAYOUT_ROW_SPLIT: "ROW_SPLIT",
    LAYOUT_CONTIGUOUS: "CONTIGUOUS",
}

# --- module kinds (TensorEntry.module_kind / ModuleRecord.module_kind) ---
MODULE_TEXT = 0
MODULE_MTP = 1
MODULE_VISION = 2
MODULE_LM_HEAD_DRAFT = 3

MODULE_NAME = {
    MODULE_TEXT: "TEXT_CORE",
    MODULE_MTP: "MTP_DRAFT",
    MODULE_VISION: "VISION_ENCODER",
    MODULE_LM_HEAD_DRAFT: "LM_HEAD_DRAFT",
}

MODULE_CANONICAL_ORDER = (
    MODULE_TEXT,
    MODULE_LM_HEAD_DRAFT,
    MODULE_MTP,
    MODULE_VISION,
)

MODULE_FORMAT = {
    MODULE_TEXT: "q5090_w4g64_mixed_v4_1",
    MODULE_MTP: "mtp_w8g32_v3",
    MODULE_VISION: "vision_q4mix_merger_w8g128_v3",
    MODULE_LM_HEAD_DRAFT: "lm_head_draft_q4g64_v1",
}

# --- scale dtype ---
SCALE_NONE = 0
SCALE_FP16 = 1

# --- fusion groups (TensorEntry.fusion_group_id / FusionGroupRecord.group_id) ---
FUSION_NONE = 0
FUSION_ATTN_IN = 1
FUSION_GDN_IN = 2
FUSION_MLP_GATEUP = 3

FUSION_GROUP_NAME = {
    FUSION_NONE: "NONE",
    FUSION_ATTN_IN: "ATTN_IN",
    FUSION_GDN_IN: "GDN_IN",
    FUSION_MLP_GATEUP: "MLP_GATEUP",
}

# --- source_kind ids (informational) ---
SK_OTHER = 0
SK_EMBED = 1
SK_LM_HEAD = 2
SK_FINAL_NORM = 3
SK_INPUT_LAYERNORM = 4
SK_POST_ATTN_LAYERNORM = 5
SK_LM_HEAD_DRAFT = 6
SK_LM_HEAD_DRAFT_IDMAP = 7
SK_GDN_A_LOG = 10
SK_GDN_DT_BIAS = 11
SK_GDN_CONV1D = 12
SK_GDN_IN_PROJ_A = 13
SK_GDN_IN_PROJ_B = 14
SK_GDN_IN_PROJ_Q = 15
SK_GDN_IN_PROJ_K = 16
SK_GDN_IN_PROJ_V = 17
SK_GDN_IN_PROJ_Z = 18
SK_GDN_NORM = 19
SK_GDN_OUT_PROJ = 20
SK_ATTN_Q = 30
SK_ATTN_GATE = 31
SK_ATTN_K = 32
SK_ATTN_V = 33
SK_ATTN_Q_NORM = 34
SK_ATTN_K_NORM = 35
SK_ATTN_O = 36
SK_MLP_GATE = 40
SK_MLP_UP = 41
SK_MLP_DOWN = 42
SK_MTP_FC = 50
SK_MTP_PRE_FC_NORM_EMB = 51
SK_MTP_PRE_FC_NORM_HID = 52
SK_MTP_NORM = 53
SK_VIS_PATCH_EMBED = 60
SK_VIS_PATCH_EMBED_BIAS = 61
SK_VIS_POS_EMBED = 62
SK_VIS_BLOCK_QKV = 63
SK_VIS_BLOCK_QKV_BIAS = 64
SK_VIS_BLOCK_PROJ = 65
SK_VIS_BLOCK_PROJ_BIAS = 66
SK_VIS_BLOCK_FC1 = 67
SK_VIS_BLOCK_FC1_BIAS = 68
SK_VIS_BLOCK_FC2 = 69
SK_VIS_BLOCK_FC2_BIAS = 70
SK_VIS_BLOCK_NORM1_W = 71
SK_VIS_BLOCK_NORM1_B = 72
SK_VIS_BLOCK_NORM2_W = 73
SK_VIS_BLOCK_NORM2_B = 74
SK_VIS_MERGER_FC1 = 75
SK_VIS_MERGER_FC1_BIAS = 76
SK_VIS_MERGER_FC2 = 77
SK_VIS_MERGER_FC2_BIAS = 78
SK_VIS_MERGER_NORM_W = 79
SK_VIS_MERGER_NORM_B = 80

NO_LAYER = 0xFFFFFFFF


@dataclass(frozen=True)
class QuantSpec:
    """Numeric parameters of a low-bit quant qtype."""

    bits: int
    group_size: int
    qmax: int
    qmin: int

    @property
    def bytes_per_group(self) -> int:
        return (self.group_size * self.bits + 7) // 8

    @property
    def nibble_bytes_per_group(self) -> int:
        if self.bits == 8:
            return self.group_size
        return self.group_size // 2

    @property
    def high_bytes_per_group(self) -> int:
        if self.bits in (4, 8):
            return 0
        return self.group_size * (self.bits - 4) // 8


# Quant parameters per qtype id (low-bit qtypes only).
QUANT_SPECS = {
    QT_Q4G64: QuantSpec(bits=4, group_size=64, qmax=7, qmin=-8),
    QT_Q5G64: QuantSpec(bits=5, group_size=64, qmax=15, qmin=-16),
    QT_Q6G64: QuantSpec(bits=6, group_size=64, qmax=31, qmin=-32),
    QT_W8G128: QuantSpec(bits=8, group_size=128, qmax=127, qmin=-127),
    QT_W8G32: QuantSpec(bits=8, group_size=32, qmax=127, qmin=-127),
}


def is_quant(qtype: int) -> bool:
    return qtype in QUANT_SPECS


def nibble_plane_bytes_per_group(qtype: int) -> int:
    return QUANT_SPECS[qtype].nibble_bytes_per_group


def high_plane_bytes_per_group(qtype: int) -> int:
    return QUANT_SPECS[qtype].high_bytes_per_group
