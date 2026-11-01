# Copied from ninfer-ext @9c933f01, tests/models/qwen4_exp/reference.py.
# Copyright the NInfer / ninfer-ext authors (giveen), Apache License 2.0.
# capy-flash changes (marked "capy-flash"): a dense SwiGLU FFN
# (``intermediate_size``, ``*.mlp.{gate,up,down}_proj.weight``) in place of the MoE when the config
# has no ``num_experts``, and PLE-less configs (no or empty ``ple_layer_ids``: the n-gram fields are
# then optional). The MoE, PLE and every other formula are unchanged, so a Flash-Next checkpoint
# gives the original oracle's results bit for bit.
"""Independent FP64 NumPy oracle for the Qwen4Exp (Qwen3.8-Flash-Next) Text model and its MTP head.

The oracle evaluates the logical formulas of docs/maintainer/qwen4-exp-model.md on one sequence.
It starts from logical weights named as in the official checkpoint without the
``model.language_model.`` prefix (``layers.N.*``, ``embed_tokens.weight``,
``hyper_connection_mixer.*``, ``lm_head.weight``) and ``mtp.*`` for the MTP head.

Every floating-point quantity is FP64. Source casts, staging dtypes and reduction order are private
execution choices, not semantic boundaries. QSA block selection is discontinuous: when two block
scores tie exactly, the lower block index wins; production routes are compared against this oracle
with an explicit near-tie allowance rather than bit-equal selection.

A sequence is processed incrementally through ``TextState``: calling ``forward`` once with the
whole prompt or repeatedly with its pieces produces the same outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    vocab_size: int
    rms_norm_eps: float
    # Gated full attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]
    # QSA indexer
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    # GDN
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str
    # Hyper-connections
    hc_count: int
    hc_lowrank: int
    # MoE (num_experts == 0: the dense FFN below, capy-flash)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    # Dense SwiGLU FFN (capy-flash)
    intermediate_size: int = 0
    # PLE (empty ple_layer_ids: none, capy-flash)
    ple_layer_ids: tuple[int, ...] = ()
    ple_embed_dim: int = 0
    ple_conv_kernel_size: int = 0
    ngram_size: int = 0
    heads_per_ngram: int = 0
    ngram_vocab_size_base: int = 0
    make_ngram_vocab_size_divisible_by: int = 1
    seed: int = 1234
    eos_token_id: int = 0

    @staticmethod
    def from_hf(text_config: dict) -> "TextConfig":
        rope = text_config.get("rope_parameters") or {}
        eos = text_config.get("eos_token_id", 0)
        eos = eos[0] if isinstance(eos, list) else eos
        moe = bool(text_config.get("num_experts"))  # capy-flash: dense when absent or 0
        ple = bool(text_config.get("ple_layer_ids"))  # capy-flash: no PLE when absent or empty
        return TextConfig(
            hidden_size=text_config["hidden_size"],
            num_hidden_layers=text_config["num_hidden_layers"],
            layer_types=tuple(text_config["layer_types"]),
            vocab_size=text_config["vocab_size"],
            rms_norm_eps=text_config["rms_norm_eps"],
            num_attention_heads=text_config["num_attention_heads"],
            num_key_value_heads=text_config["num_key_value_heads"],
            head_dim=text_config["head_dim"],
            rope_theta=rope.get("rope_theta", text_config.get("rope_theta")),
            partial_rotary_factor=rope.get("partial_rotary_factor", text_config.get("partial_rotary_factor", 1.0)),
            mrope_section=tuple(rope.get("mrope_section", (11, 11, 10))),
            indexer_n_heads=text_config["indexer_n_heads"],
            indexer_kv_heads=text_config["indexer_kv_heads"],
            indexer_head_dim=text_config["indexer_head_dim"],
            indexer_budget=text_config["indexer_budget"],
            indexer_compress_ratio=text_config["indexer_compress_ratio"],
            linear_num_key_heads=text_config["linear_num_key_heads"],
            linear_num_value_heads=text_config["linear_num_value_heads"],
            linear_key_head_dim=text_config["linear_key_head_dim"],
            linear_value_head_dim=text_config["linear_value_head_dim"],
            linear_conv_kernel_dim=text_config["linear_conv_kernel_dim"],
            output_gate_type=text_config.get("output_gate_type") or text_config["hidden_act"],
            num_experts=text_config["num_experts"] if moe else 0,
            num_experts_per_tok=text_config["num_experts_per_tok"] if moe else 0,
            moe_intermediate_size=text_config["moe_intermediate_size"] if moe else 0,
            shared_expert_intermediate_size=text_config["shared_expert_intermediate_size"] if moe else 0,
            norm_topk_prob=text_config.get("norm_topk_prob", True),
            intermediate_size=0 if moe else text_config["intermediate_size"],
            hc_count=text_config["hc_count"],
            hc_lowrank=text_config["hc_lowrank"],
            ple_layer_ids=tuple(sorted(set(text_config.get("ple_layer_ids") or ()))),
            ple_embed_dim=text_config["ple_embed_dim"] if ple else 0,
            ple_conv_kernel_size=text_config["ple_conv_kernel_size"] if ple else 0,
            ngram_size=text_config["ngram_size"] if ple else 0,
            heads_per_ngram=text_config["heads_per_ngram"] if ple else 0,
            ngram_vocab_size_base=text_config["ngram_vocab_size_base"] if ple else 0,
            make_ngram_vocab_size_divisible_by=text_config["make_ngram_vocab_size_divisible_by"] if ple else 1,
            seed=text_config.get("seed", 1234),
            eos_token_id=eos,
        )

    @property
    def hc_dim(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


# ---------------------------------------------------------------------------------------------
# Elementary functions


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "silu":
        return _silu(x)
    if name == "sigmoid":
        return _sigmoid(x)
    raise ValueError(f"unsupported gate activation {name}")


def offset_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float, group: int | None = None) -> np.ndarray:
    """Zero-centred RMSNorm, ``x / rms(x) * (1 + weight)``; ``group`` normalizes each contiguous group."""
    shape = x.shape
    if group is not None:
        x = x.reshape(*shape[:-1], -1, group)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y.reshape(shape) * (1.0 + weight)


def _gated_rms_norm(x: np.ndarray, gate: np.ndarray, weight: np.ndarray, eps: float, act: str) -> np.ndarray:
    """GDN output norm: plain-weight RMSNorm followed by the configured gate activation."""
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return weight * y * _act(act, gate)


def _l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


# ---------------------------------------------------------------------------------------------
# Interleaved multimodal RoPE


def rope_tables(config: TextConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin ``[T, rotary_dim]`` for axis-major ``[3, T]`` (temporal, height, width) positions.

    Frequency pair i uses the height axis when ``i % 3 == 1`` and ``i < 3 * section[1]``, the
    width axis when ``i % 3 == 2`` and ``i < 3 * section[2]``, and the temporal axis otherwise.
    The pair table is duplicated for the rotate-half convention.
    """
    dim = config.rotary_dim
    inv_freq = 1.0 / (config.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    positions = np.asarray(positions, dtype=np.float64).reshape(3, -1)
    axis = np.zeros(dim // 2, dtype=np.int64)
    for a, offset in ((1, 1), (2, 2)):
        axis[offset : 3 * config.mrope_section[a] : 3] = a
    freqs = positions[axis, :].T * inv_freq  # [T, dim/2]
    freqs = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(freqs), np.sin(freqs)


def apply_partial_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate the leading ``cos.shape[-1]`` features of ``x [..., D]`` with the rotate-half rule."""
    r = cos.shape[-1]
    rot, rest = x[..., :r], x[..., r:]
    half = r // 2
    rotated = np.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
    return np.concatenate([rot * cos + rotated * sin, rest], axis=-1)


# ---------------------------------------------------------------------------------------------
# Hyper-connections


def hyper_connection_mix(config: TextConfig, weights: dict, prefix: str, x: np.ndarray, combine: bool = True):
    """Collapse wide ``x [T, hc*H]`` to the block input ``[T, H]``.

    Returns ``(mixed, inject)`` where ``inject [T, hc]`` scales the block output into each stream,
    or only ``mixed`` for the output mixer that has no inject weight.
    """
    hc, h = config.hc_count, config.hidden_size
    xn = offset_rms_norm(x, weights[f"{prefix}.hc_norm.weight"], config.rms_norm_eps, group=h)
    a = _silu(xn @ weights[f"{prefix}.input_mix_weight_down.weight"].T / hc)
    m = _sigmoid(a @ weights[f"{prefix}.input_mix_weight_up.weight"].T)
    mixed = np.mean(m.reshape(-1, hc, h) * xn.reshape(-1, hc, h), axis=1)
    if not combine:
        return mixed
    inject = 2.0 * _sigmoid(xn @ weights[f"{prefix}.block_inject_weight.weight"].T / hc)
    return mixed, inject


def hyper_connection_combine(x: np.ndarray, y: np.ndarray, inject: np.ndarray) -> np.ndarray:
    """``x_s + inject_s * y`` for every stream s of wide ``x``."""
    t, hc = inject.shape
    return x + (inject[:, :, None] * y[:, None, :]).reshape(t, -1)


# ---------------------------------------------------------------------------------------------
# N-gram hash embedding


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def ngram_layer_multipliers(config: TextConfig, ple_layer_index: int) -> list[int]:
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(config.vocab_size, 1)) // 2)
    base_seed = config.seed + _PRIME_1 * ple_layer_index
    return [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(config.ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def ngram_head_tables(config: TextConfig, ple_layer_index: int) -> tuple[list[int], list[int], int]:
    """Per-head prime modulus and row offset, and the padded table row count."""
    heads = (config.ngram_size - 1) * config.heads_per_ngram
    sizes, offsets, total = [], [], 0
    prime = config.ngram_vocab_size_base - 1
    for _ in range(ple_layer_index * heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    for _ in range(heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
        offsets.append(total)
        total += prime
    divisor = config.make_ngram_vocab_size_divisible_by
    return sizes, offsets, -(-total // divisor) * divisor


def ngram_row_ids(config: TextConfig, ple_layer_index: int, history: list[int], count: int) -> np.ndarray:
    """Table rows ``[count, heads]`` for the last ``count`` tokens of ``history``.

    ``history`` carries at least ``ngram_size - 1`` tokens before those positions (EOS-padded at
    sequence start). The context token at distance s of position p is ``history[p - s]`` when none
    of ``history[p - s .. p - 1]`` is EOS, and EOS otherwise, so an EOS closes every n-gram window.
    """
    multipliers = ngram_layer_multipliers(config, ple_layer_index)
    sizes, offsets, _ = ngram_head_tables(config, ple_layer_index)
    eos = config.eos_token_id
    rows = []
    for p in range(len(history) - count, len(history)):
        shifted = []
        for s in range(config.ngram_size):
            window = history[p - s : p]
            shifted.append(history[p - s] if eos not in window else eos)
        row = []
        for n in range(2, config.ngram_size + 1):
            mixed = 0
            for k in range(n):
                mixed ^= shifted[k] * multipliers[k]
            base = (n - 2) * config.heads_per_ngram
            for j in range(base, base + config.heads_per_ngram):
                row.append(mixed % sizes[j] + offsets[j])
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# Per-sequence state


@dataclass
class _GdnState:
    conv: np.ndarray  # [K-1, conv_dim] previous projected inputs, oldest first
    recurrent: np.ndarray  # [Hv, Dk, Dv]


@dataclass
class _AttentionState:
    keys: list = field(default_factory=list)  # [KVH, D] after norm and RoPE
    values: list = field(default_factory=list)  # [KVH, D]
    index_keys: list = field(default_factory=list)  # [Di] raw indexer keys
    positions: list = field(default_factory=list)  # [3] axis positions


@dataclass
class _PleState:
    tokens: list  # previous ngram_size-1 tokens
    conv: np.ndarray  # [(kernel-1)*dilation, hc*H] previous normalized gated values


class TextState:
    """Complete recurrent and context state of one Text sequence (and optional MTP KV)."""

    def __init__(self, config: TextConfig):
        self.config = config
        c = config
        conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim + (
            c.linear_num_value_heads * c.linear_value_head_dim
        )
        self.layers: list = []
        for i, kind in enumerate(c.layer_types[: c.num_hidden_layers]):
            if kind == "linear_attention":
                self.layers.append(
                    _GdnState(
                        conv=np.zeros((c.linear_conv_kernel_dim - 1, conv_dim)),
                        recurrent=np.zeros((c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim)),
                    )
                )
            else:
                self.layers.append(_AttentionState())
        self.ple = {
            layer_id - 1: _PleState(
                tokens=[c.eos_token_id] * (c.ngram_size - 1),
                conv=np.zeros(((c.ple_conv_kernel_size - 1) * c.ngram_size, c.hc_dim)),
            )
            for layer_id in c.ple_layer_ids
        }
        self.mtp = _AttentionState()


# ---------------------------------------------------------------------------------------------
# Blocks


def _gdn(config: TextConfig, w: dict, p: str, x: np.ndarray, state: _GdnState) -> np.ndarray:
    c = config
    hk, hv, dk, dv = (
        c.linear_num_key_heads,
        c.linear_num_value_heads,
        c.linear_key_head_dim,
        c.linear_value_head_dim,
    )
    t = x.shape[0]
    mixed = x @ w[f"{p}.in_proj_qkv.weight"].T  # [T, conv_dim]
    z = (x @ w[f"{p}.in_proj_z.weight"].T).reshape(t, hv, dv)
    beta = _sigmoid(x @ w[f"{p}.in_proj_b.weight"].T)  # [T, Hv]
    g = -np.exp(w[f"{p}.A_log"]) * _softplus(x @ w[f"{p}.in_proj_a.weight"].T + w[f"{p}.dt_bias"])

    kernel = w[f"{p}.conv1d.weight"].reshape(mixed.shape[1], -1)  # [conv_dim, K]
    k_width = kernel.shape[1]
    padded = np.concatenate([state.conv, mixed], axis=0)
    conv = np.stack([np.sum(padded[i : i + k_width] * kernel.T, axis=0) for i in range(t)])
    conv = _silu(conv)
    state.conv = padded[-(k_width - 1) :].copy()

    q = conv[:, : hk * dk].reshape(t, hk, dk)
    k = conv[:, hk * dk : 2 * hk * dk].reshape(t, hk, dk)
    v = conv[:, 2 * hk * dk :].reshape(t, hv, dv)
    repeat = hv // hk
    q = np.repeat(_l2norm(q), repeat, axis=1) / math.sqrt(dk)
    k = np.repeat(_l2norm(k), repeat, axis=1)

    s = state.recurrent
    out = np.empty((t, hv, dv))
    for i in range(t):
        s = s * np.exp(g[i])[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", s, k[i])
        delta = (v[i] - kv_mem) * beta[i][:, None]
        s = s + k[i][:, :, None] * delta[:, None, :]
        out[i] = np.einsum("hkv,hk->hv", s, q[i])
    state.recurrent = s

    out = _gated_rms_norm(out, z, w[f"{p}.norm.weight"], c.rms_norm_eps, c.output_gate_type)
    return out.reshape(t, hv * dv) @ w[f"{p}.out_proj.weight"].T


def qsa_select(
    config: TextConfig,
    w: dict,
    p: str,
    query_rows: np.ndarray,
    cos_q: np.ndarray,
    sin_q: np.ndarray,
    visible: int,
    index_keys: np.ndarray,
    key_positions: np.ndarray,
) -> np.ndarray:
    """Selected cache indices for one query that sees cache indices ``[0, visible)``.

    ``query_rows`` is the indexer query projection ``[heads * Di]`` of that query. The result is
    ``min(budget/ratio, blocks)`` whole blocks of ``ratio`` tokens by descending indexer score,
    followed by the incomplete tail; order is irrelevant to attention.
    """
    c = config
    ratio, di = c.indexer_compress_ratio, c.indexer_head_dim
    blocks = visible // ratio
    tail = list(range(blocks * ratio, visible))
    if blocks == 0:
        return np.asarray(tail, dtype=np.int64)
    q = offset_rms_norm(query_rows.reshape(c.indexer_n_heads, di), w[f"{p}.q_layernorm.weight"], c.rms_norm_eps)
    q = apply_partial_rope(q, cos_q, sin_q)
    pooled = index_keys[: blocks * ratio].reshape(blocks, ratio, di).mean(axis=1)
    pooled = offset_rms_norm(pooled, w[f"{p}.k_layernorm.weight"], c.rms_norm_eps)
    cos_k, sin_k = rope_tables(c, key_positions[:, 0 : blocks * ratio : ratio])
    pooled = apply_partial_rope(pooled, cos_k, sin_k)
    scores = np.sum(np.maximum(q @ pooled.T, 0.0), axis=0) / math.sqrt(di)  # [blocks]
    keep = min(c.indexer_budget // ratio, blocks)
    order = sorted(range(blocks), key=lambda b: (-scores[b], b))[:keep]
    selected = [b * ratio + r for b in sorted(order) for r in range(ratio)]
    return np.asarray(selected + tail, dtype=np.int64)


def _attention(
    config: TextConfig,
    w: dict,
    p: str,
    x: np.ndarray,
    positions: np.ndarray,
    state: _AttentionState,
    sparse: bool = True,
) -> np.ndarray:
    c = config
    t = x.shape[0]
    nh, kvh, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
    qg = (x @ w[f"{p}.q_proj.weight"].T).reshape(t, nh, 2 * d)
    q, gate = qg[..., :d], qg[..., d:].reshape(t, nh * d)
    q = offset_rms_norm(q, w[f"{p}.q_norm.weight"], c.rms_norm_eps)
    k = offset_rms_norm((x @ w[f"{p}.k_proj.weight"].T).reshape(t, kvh, d), w[f"{p}.k_norm.weight"], c.rms_norm_eps)
    v = (x @ w[f"{p}.v_proj.weight"].T).reshape(t, kvh, d)
    cos, sin = rope_tables(c, positions)
    q = apply_partial_rope(q, cos[:, None, :], sin[:, None, :])
    k = apply_partial_rope(k, cos[:, None, :], sin[:, None, :])

    iq_rows = x @ w[f"{p}.indexer.index_qk_proj.weight"].T
    nq = c.indexer_n_heads * c.indexer_head_dim
    base = len(state.keys)
    for i in range(t):
        state.keys.append(k[i])
        state.values.append(v[i])
        state.index_keys.append(iq_rows[i, nq : nq + c.indexer_head_dim])
        state.positions.append(np.asarray(positions).reshape(3, -1)[:, i])
    keys, values = np.stack(state.keys), np.stack(state.values)
    index_keys, key_positions = np.stack(state.index_keys), np.stack(state.positions, axis=1)

    group = nh // kvh
    out = np.empty((t, nh, d))
    for i in range(t):
        visible = base + i + 1
        if sparse:
            idx = qsa_select(c, w, f"{p}.indexer", iq_rows[i, :nq], cos[i], sin[i], visible, index_keys, key_positions)
        else:
            idx = np.arange(visible)
        for h in range(nh):
            logits = keys[idx, h // group] @ q[i, h] / math.sqrt(d)
            prob = np.exp(logits - logits.max())
            prob /= prob.sum()
            out[i, h] = prob @ values[idx, h // group]
    out = out.reshape(t, nh * d) * _sigmoid(gate)
    return out @ w[f"{p}.o_proj.weight"].T


def _moe(config: TextConfig, w: dict, p: str, x: np.ndarray) -> np.ndarray:
    c = config
    logits = x @ w[f"{p}.gate.weight"].T
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    gate_up, down = w[f"{p}.experts.gate_up_proj"], w[f"{p}.experts.down_proj"]
    inter = c.moe_intermediate_size
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        ids = sorted(range(c.num_experts), key=lambda e: (-probs[t, e], e))[: c.num_experts_per_tok]
        weights = probs[t, ids]
        if c.norm_topk_prob:
            weights = weights / weights.sum()
        for e, r in zip(ids, weights):
            gu = gate_up[e] @ x[t]
            out[t] += r * (down[e] @ (_silu(gu[:inter]) * gu[inter:]))
    shared = (
        _silu(x @ w[f"{p}.shared_expert.gate_proj.weight"].T) * (x @ w[f"{p}.shared_expert.up_proj.weight"].T)
    ) @ w[f"{p}.shared_expert.down_proj.weight"].T
    return out + _sigmoid(x @ w[f"{p}.shared_expert_gate.weight"].T) * shared


def _dense_mlp(config: TextConfig, w: dict, p: str, x: np.ndarray) -> np.ndarray:
    """capy-flash: dense SwiGLU FFN ``down . (silu(gate . x) * (up . x))``."""
    return (_silu(x @ w[f"{p}.gate_proj.weight"].T) * (x @ w[f"{p}.up_proj.weight"].T)) @ w[f"{p}.down_proj.weight"].T


def _ffn(config: TextConfig, w: dict, p: str, x: np.ndarray) -> np.ndarray:
    """The block FFN: the MoE, or (capy-flash) the dense FFN of a config without experts."""
    return _moe(config, w, p, x) if config.num_experts else _dense_mlp(config, w, p, x)


def _ple(config: TextConfig, w: dict, p: str, x: np.ndarray, tokens: list[int], state: _PleState, ple_index: int):
    c = config
    hc, h = c.hc_count, c.hidden_size
    t = len(tokens)
    history = state.tokens + list(tokens)
    rows = ngram_row_ids(c, ple_index, history, t)
    table = w[f"{p}.ple_embedding.ngram_embedding.weight"]
    embeddings = table[rows].reshape(t, -1)  # [T, ple_embed_dim]
    state.tokens = history[-(c.ngram_size - 1) :]

    key = offset_rms_norm(embeddings @ w[f"{p}.key_proj.weight"].T, w[f"{p}.norm_key.weight"], c.rms_norm_eps, group=h)
    value = embeddings @ w[f"{p}.value_proj.weight"].T  # [T, H]
    query = offset_rms_norm(x, w[f"{p}.norm_query.weight"], c.rms_norm_eps, group=h)
    gate = np.sum(key.reshape(t, hc, h) * query.reshape(t, hc, h), axis=-1) / math.sqrt(h)
    gate = np.sign(gate) * np.sqrt(np.maximum(np.abs(gate), 1e-6))
    gated = (_sigmoid(gate)[:, :, None] * value[:, None, :]).reshape(t, hc * h)
    normed = offset_rms_norm(gated, w[f"{p}.norm_conv.weight"], c.rms_norm_eps, group=h)

    kernel = w[f"{p}.conv1d.weight"].reshape(hc * h, -1)  # [C, K]
    k_width, dilation = kernel.shape[1], c.ngram_size
    padded = np.concatenate([state.conv, normed], axis=0)
    span = (k_width - 1) * dilation
    conv = np.stack([np.sum(padded[i : i + span + 1 : dilation] * kernel.T, axis=0) for i in range(t)])
    state.conv = padded[-span:].copy()
    return gated + _silu(conv)


def decoder_layer(
    config: TextConfig,
    w: dict,
    p: str,
    kind: str,
    x: np.ndarray,
    positions: np.ndarray,
    state,
    ple: tuple | None = None,
    sparse: bool = True,
) -> np.ndarray:
    """One wide-residual block. ``ple`` is ``(tokens, _PleState, ple_index)`` at an injection layer."""
    if ple is not None:
        x = x + _ple(config, w, f"{p}.ple", x, ple[0], ple[1], ple[2])
    mixed, inject = hyper_connection_mix(config, w, f"{p}.attn_hyper_connection", x)
    if kind == "linear_attention":
        y = _gdn(config, w, f"{p}.linear_attn", mixed, state)
    else:
        y = _attention(config, w, f"{p}.self_attn", mixed, positions, state, sparse)
    x = hyper_connection_combine(x, y, inject)
    mixed, inject = hyper_connection_mix(config, w, f"{p}.mlp_hyper_connection", x)
    return hyper_connection_combine(x, _ffn(config, w, f"{p}.mlp", mixed), inject)


def text_positions(start: int, count: int) -> np.ndarray:
    return np.tile(np.arange(start, start + count, dtype=np.int64), (3, 1))


def forward(
    config: TextConfig,
    weights: dict,
    state: TextState,
    tokens: list[int],
    positions: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    sparse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance ``state`` by ``tokens``; returns ``(logits [T, V], wide_hidden [T, hc*H])``.

    ``embeddings`` optionally replaces the embedding rows (media placeholders); PLE always hashes
    ``tokens``. ``sparse=False`` evaluates dense causal attention for the QSA-equivalence check.
    """
    c = config
    t = len(tokens)
    if positions is None:
        start = len(next((s.keys for s in state.layers if isinstance(s, _AttentionState)), []))
        positions = text_positions(start, t)
    x = weights["embed_tokens.weight"][np.asarray(tokens)] if embeddings is None else embeddings
    x = np.tile(x, (1, c.hc_count))
    for i, kind in enumerate(c.layer_types[: c.num_hidden_layers]):
        ple = None
        if i in state.ple:
            ple = (tokens, state.ple[i], c.ple_layer_ids.index(i + 1))
        x = decoder_layer(c, weights, f"layers.{i}", kind, x, positions, state.layers[i], ple, sparse)
    out = hyper_connection_mix(c, weights, "hyper_connection_mixer", x, combine=False)
    return out @ weights["lm_head.weight"].T, x


def mtp_forward(
    config: TextConfig,
    weights: dict,
    state: TextState,
    next_tokens: list[int],
    wide_hidden: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One MTP step over aligned inputs; returns ``(draft_logits, draft_wide_hidden)``.

    ``wide_hidden [T, hc*H]`` is the pre-mixer residual the target (or the previous MTP step)
    produced at the same positions, and ``next_tokens`` are the tokens one position later.
    """
    c = config
    e = offset_rms_norm(
        weights["embed_tokens.weight"][np.asarray(next_tokens)],
        weights["mtp.pre_fc_norm_embedding.weight"],
        c.rms_norm_eps,
    )
    e = e @ weights["mtp.fc_embedding.weight"].T
    h = offset_rms_norm(wide_hidden, weights["mtp.pre_fc_norm_hidden.weight"], c.rms_norm_eps)
    t = h.shape[0]
    x = (h.reshape(t, c.hc_count, c.hidden_size) @ weights["mtp.fc_hidden.weight"].T + e[:, None, :]).reshape(t, -1)
    x = decoder_layer(c, weights, "mtp.layers.0", "full_attention", x, positions, state.mtp)
    out = hyper_connection_mix(c, weights, "mtp.hyper_connection_mixer", x, combine=False)
    return out @ weights["lm_head.weight"].T, x
