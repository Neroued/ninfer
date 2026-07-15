"""Fixed compute configuration for the Qwen3.6-35B-A3B reference path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 2048
    layers: int = 40
    experts: int = 256
    experts_per_token: int = 8
    expert_intermediate: int = 512
    shared_intermediate: int = 512
    vocab: int = 248320
    token_domain: int = 248077
    q_heads: int = 16
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    full_interval: int = 4
    gdn_k_heads: int = 16
    gdn_v_heads: int = 32
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    conv_width: int = 4
    conv_state_bytes: int = 2
    rms_eps: float = 1.0e-6
    rope_theta: float = 1.0e7
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    prefill_chunk: int = 1024
    max_position_embeddings: int = 262144

    @property
    def q_size(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def key_dim(self) -> int:
        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def value_dim(self) -> int:
        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def full_layers(self) -> int:
        return self.layers // self.full_interval

    @property
    def gdn_layers(self) -> int:
        return self.layers - self.full_layers

    def is_full(self, layer: int) -> bool:
        return (layer + 1) % self.full_interval == 0

    def full_index(self, layer: int) -> int:
        return (layer + 1) // self.full_interval - 1

    def gdn_index(self, layer: int) -> int:
        return layer - (layer + 1) // self.full_interval


CFG = ModelConfig()


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 27
    hidden: int = 1152
    intermediate: int = 4304
    out_hidden: int = 2048
    heads: int = 16
    in_channels: int = 3
    patch: int = 16
    temporal_patch: int = 2
    spatial_merge: int = 2
    position_embeddings: int = 2304
    rope_theta: float = 10000.0
    norm_eps: float = 1.0e-6

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch * self.patch * self.patch

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge * self.spatial_merge

    @property
    def merger_hidden(self) -> int:
        return self.hidden * self.merge_unit

    @property
    def position_side(self) -> int:
        return int(self.position_embeddings**0.5)


VISION_CFG = VisionConfig()
ATTN_SCALE = CFG.head_dim ** -0.5
GDN_SCALE = CFG.gdn_k_dim ** -0.5


__all__ = ["ATTN_SCALE", "CFG", "GDN_SCALE", "ModelConfig", "VISION_CFG", "VisionConfig"]
