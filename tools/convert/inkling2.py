"""Inkling-2 checkpoint axes, logical parameters, and component resources.

Supports both non-looped (`Inkling2For*`) and looped (`Inkling2LoopFor*`) checkpoints
under a single unified `Inkling2ForCausalLM` / `inkling2_text` artifact representation,
with optional `vision` (PE-Lang ViT) and `dspark` (`mtp.safetensors`) components.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite, prod

import torch

from .config import expect_value, positive_float, positive_integer
from .model import Model, Parameter
from .resources import TEXT_RESOURCES, load_resources
from .sources.inkling2 import InklingExpertBankSource
from .sources.logical import LogicalSource, array_source, transpose_source
from .sources.safetensors import SafetensorsSource, tensor_source

_SUPPORTED_ARCHITECTURES = {
    "Inkling2ForConditionalGeneration",
    "Inkling2ForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2LoopForCausalLM",
}

_TEXT_RESOURCES = TEXT_RESOURCES + ("special_tokens_map.json",)


def _align_vocab(unpadded: int, raw_vocab: int) -> int:
    padded = ((unpadded + 63) // 64) * 64
    if padded > raw_vocab:
        raise ValueError(
            f"inkling2: 64-aligned vocab {padded} exceeds checkpoint vocab_size {raw_vocab}"
        )
    return padded


def _num_logical_layers(cfg: dict) -> int:
    n = cfg["num_hidden_layers"]
    loop = cfg.get("loop_config")
    if loop is None:
        return n
    core_len = loop["core_end"] - loop["core_start"]
    return n + core_len * (loop["num_passes"] - 1)


def text_config(source: dict) -> tuple[dict, int]:
    """Return canonical trimmed `inkling2_text` config and source raw `vocab_size`."""
    architectures = source.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in _SUPPORTED_ARCHITECTURES
    ):
        raise ValueError(f"expected Inkling2 architecture, got {architectures!r}")
    raw = deepcopy(source.get("text_config", source))

    for key, expected in (
        ("q_bias", False),
        ("o_bias", False),
        ("use_value_norm_no_gamma", True),
        ("attn_res_normalize_values", True),
        ("n_shared_experts", 0),
        ("use_gate_bias", True),
    ):
        expect_value(raw, key, expected, "inkling2")

    raw_vocab = positive_integer(raw.get("vocab_size"), "inkling2.vocab_size")
    unpadded_vocab = positive_integer(
        raw.get("unpadded_vocab_size", raw_vocab), "inkling2.unpadded_vocab_size"
    )
    if unpadded_vocab > raw_vocab:
        raise ValueError("inkling2: unpadded_vocab_size exceeds vocab_size")
    padded_vocab = _align_vocab(unpadded_vocab, raw_vocab)

    hidden = positive_integer(raw.get("hidden_size"), "inkling2.hidden_size")
    num_layers = positive_integer(
        raw.get("num_hidden_layers"), "inkling2.num_hidden_layers"
    )
    max_length = positive_integer(
        raw.get("model_max_length", raw.get("max_position_embeddings")),
        "inkling2.model_max_length",
    )
    num_heads = positive_integer(
        raw.get("num_attention_heads"), "inkling2.num_attention_heads"
    )
    num_kv_heads = positive_integer(
        raw.get("num_key_value_heads"), "inkling2.num_key_value_heads"
    )
    head_dim = positive_integer(raw.get("head_dim"), "inkling2.head_dim")
    swa_num_heads = positive_integer(
        raw.get("swa_num_attention_heads", num_heads),
        "inkling2.swa_num_attention_heads",
    )
    swa_num_kv_heads = positive_integer(
        raw.get("swa_num_key_value_heads", num_kv_heads),
        "inkling2.swa_num_key_value_heads",
    )
    swa_head_dim = positive_integer(
        raw.get("swa_head_dim", head_dim), "inkling2.swa_head_dim"
    )
    if num_heads % num_kv_heads or swa_num_heads % swa_num_kv_heads:
        raise ValueError("inkling2: num_attention_heads must be divisible by kv heads")

    d_rel = positive_integer(raw.get("d_rel"), "inkling2.d_rel")
    rel_extent = positive_integer(raw.get("rel_extent"), "inkling2.rel_extent")
    sliding_window = positive_integer(
        raw.get("sliding_window_size"), "inkling2.sliding_window_size"
    )
    sconv_kernel = positive_integer(
        raw.get("sconv_kernel_size"), "inkling2.sconv_kernel_size"
    )
    attn_res_block = positive_integer(
        raw.get("n_layers_per_attn_res_block"), "inkling2.n_layers_per_attn_res_block"
    )
    experts = positive_integer(raw.get("n_routed_experts"), "inkling2.n_routed_experts")
    top_k = positive_integer(
        raw.get("num_experts_per_tok"), "inkling2.num_experts_per_tok"
    )
    if top_k > experts:
        raise ValueError("inkling2: num_experts_per_tok exceeds n_routed_experts")
    intermediate = positive_integer(
        raw.get("intermediate_size", raw.get("moe_intermediate_size")),
        "inkling2.intermediate_size",
    )

    local_ids = raw.get("local_layer_ids")
    if (
        not isinstance(local_ids, list)
        or any(type(i) is not int or not 0 <= i < num_layers for i in local_ids)
        or len(set(local_ids)) != len(local_ids)
    ):
        raise ValueError("inkling2.local_layer_ids: invalid physical layer index list")
    local_ids = sorted(local_ids)

    log_floor = raw.get("log_scaling_n_floor")
    if log_floor is not None:
        log_floor = positive_integer(log_floor, "inkling2.log_scaling_n_floor")
    log_alpha = raw.get("log_scaling_alpha", 0.0)
    if type(log_alpha) not in (int, float) or not isfinite(log_alpha) or log_alpha < 0:
        raise ValueError("inkling2.log_scaling_alpha: expected nonnegative finite float")

    eos_token_id = source.get("eos_token_id", raw.get("eos_token_id"))
    if type(eos_token_id) is not int or not 0 <= eos_token_id < unpadded_vocab:
        raise ValueError(f"inkling2.eos_token_id: invalid {eos_token_id!r}")

    cfg = {
        "architectures": ["Inkling2ForCausalLM"],
        "model_type": "inkling2_text",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "vocab_size": padded_vocab,
        "unpadded_vocab_size": unpadded_vocab,
        "max_position_embeddings": max_length,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "swa_num_attention_heads": swa_num_heads,
        "swa_num_key_value_heads": swa_num_kv_heads,
        "swa_head_dim": swa_head_dim,
        "d_rel": d_rel,
        "rel_extent": rel_extent,
        "sliding_window_size": sliding_window,
        "local_layer_ids": local_ids,
        "sconv_kernel_size": sconv_kernel,
        "n_layers_per_attn_res_block": attn_res_block,
        "attn_res_normalize_values": True,
        "use_value_norm_no_gamma": True,
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": positive_float(raw.get("rms_norm_eps", 1e-6), "inkling2.rms_norm_eps"),
        "qk_l2_clip_threshold": positive_float(
            raw.get("qk_l2_clip_threshold"), "inkling2.qk_l2_clip_threshold"
        ),
        "rel_l2_clip_threshold": positive_float(
            raw.get("rel_l2_clip_threshold"), "inkling2.rel_l2_clip_threshold"
        ),
        "log_scaling_n_floor": log_floor,
        "log_scaling_alpha": float(log_alpha),
        "final_logit_softcapping": positive_float(
            raw.get("final_logit_softcapping"), "inkling2.final_logit_softcapping"
        ),
        "n_routed_experts": experts,
        "num_experts_per_tok": top_k,
        "n_shared_experts": 0,
        "intermediate_size": intermediate,
        "mlp_gate_softcap": positive_float(
            raw.get("mlp_gate_softcap"), "inkling2.mlp_gate_softcap"
        ),
        "mlp_up_softcap": positive_float(
            raw.get("mlp_up_softcap"), "inkling2.mlp_up_softcap"
        ),
        "route_scale": positive_float(raw.get("route_scale"), "inkling2.route_scale"),
        "use_gate_bias": True,
        "eos_token_id": eos_token_id,
    }

    loop = raw.get("loop_config")
    if loop is not None:
        if not isinstance(loop, dict):
            raise ValueError("inkling2.loop_config: expected object")
        expect_value(loop, "is_varloop", False, "inkling2.loop_config")
        core_start = positive_integer(
            loop.get("core_start"), "inkling2.loop_config.core_start"
        )
        core_end = positive_integer(
            loop.get("core_end"), "inkling2.loop_config.core_end"
        )
        num_passes = positive_integer(
            loop.get("num_passes"), "inkling2.loop_config.num_passes"
        )
        gate_rank = positive_integer(
            loop.get("gate_rank"), "inkling2.loop_config.gate_rank"
        )
        if (
            not 0 < core_start < core_end <= num_layers
            or core_start % attn_res_block
            or core_end % attn_res_block
            or num_passes < 2
        ):
            raise ValueError(f"inkling2.loop_config: invalid schedule {loop!r}")
        cfg["loop_config"] = {
            "core_start": core_start,
            "core_end": core_end,
            "num_passes": num_passes,
            "gate_rank": gate_rank,
            "is_varloop": False,
        }

    return cfg, raw_vocab


def vision_config(source: dict, text: dict) -> dict:
    raw = source.get("vision_config")
    if not isinstance(raw, dict):
        raise ValueError("inkling2: vision component requested without vision_config")
    hidden = positive_integer(raw.get("hidden_size"), "inkling2.vision.hidden_size")
    num_layers = positive_integer(
        raw.get("num_hidden_layers"), "inkling2.vision.num_hidden_layers"
    )
    num_heads = positive_integer(
        raw.get("num_attention_heads"), "inkling2.vision.num_attention_heads"
    )
    if hidden % num_heads:
        raise ValueError("inkling2.vision: hidden_size not divisible by num_attention_heads")
    patch_size = positive_integer(
        raw.get("native_patch_size"), "inkling2.vision.native_patch_size"
    )
    merge_size = positive_integer(
        raw.get("adapter_merge_size"), "inkling2.vision.adapter_merge_size"
    )
    grid_size = positive_integer(
        raw.get("posemb_grid_size"), "inkling2.vision.posemb_grid_size"
    )
    proj_dim = positive_integer(
        raw.get("projection_hidden_dim"), "inkling2.vision.projection_hidden_dim"
    )
    text_hidden = positive_integer(
        raw.get("text_hidden_size", text["hidden_size"]),
        "inkling2.vision.text_hidden_size",
    )
    if text_hidden != text["hidden_size"]:
        raise ValueError("inkling2.vision.text_hidden_size disagrees with text.hidden_size")
    mlp_ratio = positive_float(raw.get("mlp_ratio", 4.0), "inkling2.vision.mlp_ratio")
    intermediate = int(round(hidden * mlp_ratio))
    if abs(intermediate - hidden * mlp_ratio) > 1e-6 or intermediate <= 0:
        raise ValueError("inkling2.vision: hidden_size * mlp_ratio must be a positive integer")
    projector_bias = raw.get("projector_bias", False)
    if type(projector_bias) is not bool:
        raise ValueError("inkling2.vision.projector_bias must be boolean")
    return {
        "model_type": "inkling2_vision",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": hidden // num_heads,
        "intermediate_size": intermediate,
        "mlp_ratio": mlp_ratio,
        "native_patch_size": patch_size,
        "adapter_merge_size": merge_size,
        "temporal_patch_size": positive_integer(
            raw.get("temporal_patch_size", 2), "inkling2.vision.temporal_patch_size"
        ),
        "posemb_grid_size": grid_size,
        "projection_hidden_dim": proj_dim,
        "projector_bias": projector_bias,
        "text_hidden_size": text_hidden,
        "layer_norm_eps": positive_float(
            raw.get("layer_norm_eps", 1e-5), "inkling2.vision.layer_norm_eps"
        ),
    }


def dspark_config(source: dict, text: dict) -> dict:
    raw = source.get("mtp_config")
    if not isinstance(raw, dict):
        raise ValueError("inkling2: dspark component requested without mtp_config")
    expect_value(raw, "q_bias", False, "dspark")
    expect_value(raw, "o_bias", False, "dspark")
    expect_value(raw, "markov_head_type", "rnn_hidden", "dspark")
    hidden = positive_integer(raw.get("hidden_size"), "dspark.hidden_size")
    if hidden != text["hidden_size"]:
        raise ValueError("dspark.hidden_size disagrees with text.hidden_size")
    num_layers = positive_integer(
        raw.get("num_hidden_layers"), "dspark.num_hidden_layers"
    )
    num_heads = positive_integer(
        raw.get("num_attention_heads"), "dspark.num_attention_heads"
    )
    num_kv_heads = positive_integer(
        raw.get("num_key_value_heads"), "dspark.num_key_value_heads"
    )
    head_dim = positive_integer(raw.get("head_dim"), "dspark.head_dim")
    if num_heads % num_kv_heads or head_dim % 2:
        raise ValueError("dspark: invalid attention geometry")
    intermediate = positive_integer(
        raw.get("intermediate_size"), "dspark.intermediate_size"
    )
    max_pos = positive_integer(
        raw.get("max_position_embeddings"), "dspark.max_position_embeddings"
    )
    block_size = positive_integer(raw.get("block_size"), "dspark.block_size")
    if block_size < 2:
        raise ValueError("dspark.block_size must be at least 2")
    markov_rank = positive_integer(raw.get("markov_rank"), "dspark.markov_rank")
    sliding_window = positive_integer(
        raw.get("sliding_window_size"), "dspark.sliding_window_size"
    )
    local_ids = raw.get("local_layer_ids")
    if (
        not isinstance(local_ids, list)
        or any(type(i) is not int or not 0 <= i < num_layers for i in local_ids)
        or len(set(local_ids)) != len(local_ids)
    ):
        raise ValueError("dspark.local_layer_ids: invalid layer index list")
    logical_layers = _num_logical_layers(text)
    taps = raw.get("target_layer_ids")
    if (
        not isinstance(taps, list)
        or not taps
        or any(type(i) is not int or not 0 <= i <= logical_layers for i in taps)
        or len(set(taps)) != len(taps)
    ):
        raise ValueError(
            f"dspark.target_layer_ids: invalid tap indices for {logical_layers} logical layers"
        )
    return {
        "architectures": ["Inkling2DSparkDraftModel"],
        "model_type": "inkling2_dspark",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "max_position_embeddings": max_pos,
        "vocab_size": text["vocab_size"],
        "unpadded_vocab_size": text["unpadded_vocab_size"],
        "rms_norm_eps": positive_float(raw.get("rms_norm_eps", 1e-6), "dspark.rms_norm_eps"),
        "rope_theta": positive_float(raw.get("rope_theta", 1e6), "dspark.rope_theta"),
        "local_layer_ids": sorted(local_ids),
        "sliding_window_size": sliding_window,
        "block_size": block_size,
        "markov_rank": markov_rank,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": list(taps),
        "final_logit_softcapping": text["final_logit_softcapping"],
    }


def _padded_vocab_rows_source(
    store: SafetensorsSource,
    name: str,
    raw_vocab: int,
    unpadded_vocab: int,
    padded_vocab: int,
    width: int,
) -> LogicalSource:
    """Slice rows [0, unpadded_vocab) from [raw_vocab, width] BF16 and zero-pad to padded_vocab."""
    info = store.describe(name)
    if info.shape != (raw_vocab, width) or info.dtype != "BF16":
        raise ValueError(
            f"{name}: expected BF16 {(raw_vocab, width)}, got {info.dtype} {info.shape}"
        )
    valid_elems = unpadded_vocab * width
    shape = (padded_vocab, width)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.bfloat16)
        if end <= valid_elems:
            return store.read_flat(name, begin, end)
        if begin >= valid_elems:
            return torch.zeros(end - begin, dtype=torch.bfloat16)
        head = store.read_flat(name, begin, valid_elems)
        tail = torch.zeros(end - valid_elems, dtype=torch.bfloat16)
        return torch.cat((head, tail))

    return LogicalSource(
        shape, f"{store.path}:{name}[:{unpadded_vocab}->{padded_vocab}]", read
    )


def _column_slice_source(
    store: SafetensorsSource,
    name: str,
    source_shape: tuple[int, int],
    col_begin: int,
    col_end: int,
) -> LogicalSource:
    rows, total_cols = source_shape
    info = store.describe(name)
    if info.shape != source_shape or info.dtype != "BF16":
        raise ValueError(
            f"{name}: expected BF16 {source_shape}, got {info.dtype} {info.shape}"
        )
    width = col_end - col_begin
    shape = (rows, width)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.bfloat16)
        first, last = begin // width, (end + width - 1) // width
        chunk = store.read_flat(name, first * total_cols, last * total_cols).reshape(
            last - first, total_cols
        )
        sliced = chunk[:, col_begin:col_end].contiguous().reshape(-1)
        return sliced[begin - first * width : end - first * width]

    return LogicalSource(shape, f"{store.path}:{name}[:,{col_begin}:{col_end}]", read)


def _stacked_bf16_source(
    store: SafetensorsSource, names: tuple[str, ...], expected_shape: tuple[int, ...]
) -> LogicalSource:
    for name in names:
        info = store.describe(name)
        if info.shape != expected_shape or info.dtype != "BF16":
            raise ValueError(
                f"{name}: expected BF16 {expected_shape}, got {info.dtype} {info.shape}"
            )
    width = prod(expected_shape)
    out_shape = (len(names), *expected_shape)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.bfloat16)
        parts = []
        while begin < end:
            idx, local = divmod(begin, width)
            stop = min(end, (idx + 1) * width)
            parts.append(store.read_flat(names[idx], local, local + stop - begin))
            begin = stop
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    return LogicalSource(out_shape, f"stack({names[0]}, {len(names)})", read)


def _precompute_dspark_fold(
    base: SafetensorsSource,
    mtp: SafetensorsSource,
    raw_vocab: int,
    unpadded_vocab: int,
    padded_vocab: int,
    hidden: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact FP64 fold_weight [V_pad, r] and fold_bias [V_pad] in BF16."""
    unembed_info = base.describe("model.llm.unembed.weight")
    if unembed_info.shape != (raw_vocab, hidden) or unembed_info.dtype != "BF16":
        raise ValueError(
            f"model.llm.unembed.weight: expected BF16 {(raw_vocab, hidden)}, "
            f"got {unembed_info.dtype} {unembed_info.shape}"
        )
    w_info = mtp.describe("markov_head.hidden_out.weight")
    b_info = mtp.describe("markov_head.hidden_out.bias")
    if w_info.shape != (hidden, rank) or w_info.dtype != "BF16":
        raise ValueError(
            f"markov_head.hidden_out.weight: expected BF16 {(hidden, rank)}, "
            f"got {w_info.dtype} {w_info.shape}"
        )
    if b_info.shape != (hidden,) or b_info.dtype != "BF16":
        raise ValueError(
            f"markov_head.hidden_out.bias: expected BF16 {(hidden,)}, "
            f"got {b_info.dtype} {b_info.shape}"
        )
    proj = mtp.read_flat("markov_head.hidden_out.weight").reshape(hidden, rank).double()
    bias = mtp.read_flat("markov_head.hidden_out.bias").reshape(hidden).double()

    fold_weight = torch.zeros((padded_vocab, rank), dtype=torch.bfloat16)
    fold_bias = torch.zeros((padded_vocab,), dtype=torch.bfloat16)
    chunk_rows = 4096
    for start in range(0, unpadded_vocab, chunk_rows):
        stop = min(unpadded_vocab, start + chunk_rows)
        head = (
            base.read_flat("model.llm.unembed.weight", start * hidden, stop * hidden)
            .reshape(stop - start, hidden)
            .double()
        )
        fold_weight[start:stop] = (head @ proj).to(torch.bfloat16)
        fold_bias[start:stop] = (head @ bias).to(torch.bfloat16)
    return fold_weight, fold_bias


class Builder:
    def __init__(self, model: Model, base: SafetensorsSource, cfg: dict, raw_vocab: int):
        self.model = model
        self.base = base
        self.cfg = cfg
        self.raw_vocab = raw_vocab
        self.consumed: set[str] = set()
        self.expert_banks: dict[int, InklingExpertBankSource] = {}

    def group(self, *names: str) -> None:
        self.model.packing_groups.append(tuple(names))

    def add(
        self,
        name: str,
        source: str,
        shape: tuple[int, ...],
        *,
        inputs: tuple[str, ...] = (),
        direct: str | None = None,
        source_shape: tuple[int, ...] | None = None,
        transpose: tuple[int, ...] | None = None,
        store: SafetensorsSource | None = None,
        consumed: set[str] | None = None,
    ) -> None:
        target_store = self.base if store is None else store
        track = self.consumed if consumed is None else consumed
        shape = tuple(shape)
        original = tuple(source_shape or shape)
        info = target_store.describe(source)
        inferred = {"BF16": "bf16", "F32": "fp32"}.get(info.dtype)
        if inferred is None:
            raise ValueError(f"{source}: unsupported direct dtype {info.dtype}")
        out_format = direct or inferred
        expected_dtype = info.dtype

        def factory(selected, format=None):
            if format not in (None, out_format):
                raise ValueError(f"{name}: source representation is {out_format}")
            desc = selected.describe(source)
            if desc.shape != original or desc.dtype != expected_dtype:
                raise ValueError(
                    f"{source}: expected {expected_dtype} {original}, got {desc.dtype} {desc.shape}"
                )
            ref = tensor_source(selected, source, original)
            if transpose is not None:
                return transpose_source(ref, transpose, shape)
            if original != shape:
                return ref.reshape(shape)
            return ref

        track.add(source)
        self.model.add(
            Parameter(
                name,
                shape,
                factory(target_store),
                source_factory=factory,
                inputs=tuple(inputs),
                direct_format=out_format,
                residency=name.split("/")[0],
            )
        )

    def add_text(self) -> None:
        cfg = self.cfg
        h = cfg["hidden_size"]
        v_pad = cfg["vocab_size"]
        v_unpad = cfg["unpadded_vocab_size"]
        num_layers = cfg["num_hidden_layers"]
        local_set = set(cfg["local_layer_ids"])
        loop = cfg.get("loop_config")
        core_range = (
            range(loop["core_start"], loop["core_end"]) if loop is not None else range(0)
        )
        num_passes = loop["num_passes"] if loop is not None else 1

        # Embeddings, norms, output query, LM head
        self.consumed.add("model.llm.embed.weight")
        self.model.add(
            Parameter(
                "text/token_embedding",
                (v_pad, h),
                _padded_vocab_rows_source(
                    self.base, "model.llm.embed.weight", self.raw_vocab, v_unpad, v_pad, h
                ),
                direct_format="bf16",
                residency="text",
            )
        )
        self.add("text/embed_norm", "model.llm.embed_norm.weight", (h,), direct="bf16")
        self.add("text/final_norm", "model.llm.norm.weight", (h,), direct="bf16")
        self.add(
            "text/output_query",
            "model.llm.attn_res.output_query",
            (h,),
            direct="fp32",
        )
        self.consumed.add("model.llm.unembed.weight")
        self.model.add(
            Parameter(
                "text/output_head",
                (v_pad, h),
                _padded_vocab_rows_source(
                    self.base,
                    "model.llm.unembed.weight",
                    self.raw_vocab,
                    v_unpad,
                    v_pad,
                    h,
                ),
                inputs=("text/final_hidden",),
                direct_format="bf16",
                residency="text",
            )
        )

        # Cross-pass message queries (regrouped per core layer as [4, H] = (attn, k, v, mlp))
        cross_q_tensor = cross_kv_tensor = None
        if loop is not None:
            core_len = loop["core_end"] - loop["core_start"]
            for field in (
                "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries",
            ):
                info = self.base.describe(field)
                if info.shape != (2 * core_len, h) or info.dtype != "F32":
                    raise ValueError(
                        f"{field}: expected F32 {(2 * core_len, h)}, got {info.dtype} {info.shape}"
                    )
                self.consumed.add(field)
            cross_q_tensor = self.base.read_flat(
                "model.llm.middle_cycle_cross_pass_queries"
            ).reshape(core_len, 2, h)
            cross_kv_tensor = self.base.read_flat(
                "model.llm.middle_cycle_cross_pass_kv_queries"
            ).reshape(core_len, 2, h)

        d_rel = cfg["d_rel"]
        rel_extent = cfg["rel_extent"]
        taps = cfg["sconv_kernel_size"]
        experts = cfg["n_routed_experts"]
        inter = cfg["intermediate_size"]

        for layer in range(num_layers):
            p = f"text/layers/{layer}/"
            s = f"model.llm.layers.{layer}."
            is_swa = layer in local_set
            in_core = layer in core_range
            passes = range(num_passes) if in_core else (0,)

            hq = (
                cfg["swa_num_attention_heads"]
                if is_swa
                else cfg["num_attention_heads"]
            )
            hkv = (
                cfg["swa_num_key_value_heads"]
                if is_swa
                else cfg["num_key_value_heads"]
            )
            d = cfg["swa_head_dim"] if is_swa else cfg["head_dim"]
            q_dim = hq * d
            r_dim = hq * d_rel
            kv_dim = hkv * d

            # AttnRes queries & layer input norms
            if layer > 0:
                q_shape = (h,) if is_swa else (3, h)
                self.add(
                    p + "attention/attn_res_query",
                    f"model.llm.attn_res.attn_queries.{layer - 1}",
                    q_shape,
                    direct="fp32",
                )
            self.add(
                p + "mlp/attn_res_query",
                f"model.llm.attn_res.mlp_queries.{layer}",
                (h,),
                direct="fp32",
            )
            if is_swa:
                self.add(p + "attention/input_norm", s + "attn_norm.weight", (h,), direct="bf16")
            else:
                norm_names = (
                    s + "q_input_norm.weight",
                    s + "k_input_norm.weight",
                    s + "v_input_norm.weight",
                )
                self.consumed.update(norm_names)
                self.model.add(
                    Parameter(
                        p + "attention/input_norm",
                        (3, h),
                        _stacked_bf16_source(self.base, norm_names, (h,)),
                        direct_format="bf16",
                        residency="text",
                    )
                )
            self.add(p + "mlp/input_norm", s + "mlp_norm.weight", (h,), direct="bf16")

            # Attention projections
            q_inputs = tuple(f"{p}attention_q_input_p{k}" for k in passes)
            k_inputs = tuple(f"{p}attention_k_input_p{k}" for k in passes)
            v_inputs = tuple(f"{p}attention_v_input_p{k}" for k in passes)
            o_inputs = tuple(f"{p}attention_output_p{k}" for k in passes)

            self.add(
                p + "attention/query",
                s + "attn.wq_du.weight",
                (q_dim, h),
                inputs=q_inputs,
                direct="bf16",
            )
            self.add(
                p + "attention/relative",
                s + "attn.wr_du.weight",
                (r_dim, h),
                inputs=q_inputs,
                direct="bf16",
            )
            self.add(
                p + "attention/output_gate",
                s + "attn.gate_attn.weight",
                (q_dim, h),
                inputs=q_inputs,
                direct="bf16",
            )
            self.add(
                p + "attention/key",
                s + "attn.wk_dv.weight",
                (kv_dim, h),
                inputs=k_inputs,
                direct="bf16",
            )
            self.add(
                p + "attention/value",
                s + "attn.wv_dv.weight",
                (kv_dim, h),
                inputs=v_inputs,
                direct="bf16",
            )
            if is_swa:
                self.group(
                    p + "attention/query",
                    p + "attention/relative",
                    p + "attention/output_gate",
                    p + "attention/key",
                    p + "attention/value",
                )
            else:
                self.group(
                    p + "attention/query",
                    p + "attention/relative",
                    p + "attention/output_gate",
                )
                self.add(
                    p + "attention/k_conv_proj",
                    s + "attn.k_sconv.weight_proj",
                    (taps * hkv, h),
                    inputs=k_inputs,
                    direct="bf16",
                )
                self.add(
                    p + "attention/v_conv_proj",
                    s + "attn.v_sconv.weight_proj",
                    (taps * hkv, h),
                    inputs=v_inputs,
                    direct="bf16",
                )

            self.add(
                p + "attention/sink",
                s + "attn.attn_sink_proj.weight",
                (hq, h),
                direct="bf16",
            )
            self.add(
                p + "attention/output",
                s + "attn.wo_ud.weight",
                (h, q_dim),
                inputs=o_inputs,
                direct="bf16",
            )
            self.add(
                p + "attention/q_scale",
                s + "attn.q_l2_clip.scale",
                (hq, d),
                direct="bf16",
            )
            self.add(
                p + "attention/r_scale",
                s + "attn.rel_l2_clip.scale",
                (hq, d_rel),
                direct="bf16",
            )
            self.add(
                p + "attention/rel_logits_proj",
                s + "attn.rel_logits_proj.proj",
                (rel_extent, d_rel),
                source_shape=(d_rel, rel_extent),
                transpose=(1, 0),
                direct="bf16",
            )
            if is_swa:
                for stream in ("k", "v"):
                    self.add(
                        p + f"attention/{stream}_conv",
                        s + f"attn.{stream}_sconv.weight",
                        (taps, kv_dim),
                        source_shape=(kv_dim, 1, taps),
                        transpose=(2, 0, 1),
                        direct="bf16",
                    )
            else:
                for stream in ("k", "v"):
                    self.add(
                        p + f"attention/{stream}_conv",
                        s + f"attn.{stream}_sconv.static_w",
                        (taps, kv_dim),
                        direct="bf16",
                    )

            self.add(
                p + "attention/short_conv",
                s + "attn_sconv.weight",
                (taps, h),
                source_shape=(h, 1, taps),
                transpose=(2, 0, 1),
                direct="bf16",
            )
            self.add(
                p + "mlp/short_conv",
                s + "mlp_sconv.weight",
                (taps, h),
                source_shape=(h, 1, taps),
                transpose=(2, 0, 1),
                direct="bf16",
            )

            # Loop core controllers & message queries
            if in_core:
                assert cross_q_tensor is not None and cross_kv_tensor is not None
                core_idx = layer - loop["core_start"]
                rank = loop["gate_rank"]
                msg_q = torch.stack(
                    (
                        cross_q_tensor[core_idx, 0],
                        cross_kv_tensor[core_idx, 0],
                        cross_kv_tensor[core_idx, 1],
                        cross_q_tensor[core_idx, 1],
                    ),
                    dim=0,
                ).contiguous()
                self.model.add(
                    Parameter(
                        p + "loop/message_queries",
                        (4, h),
                        array_source(msg_q, f"loop_message_queries(layer={layer})"),
                        direct_format="fp32",
                        residency="text",
                    )
                )
                for group_name, streams in (
                    ("attn_kv", ("attn", "k", "v")),
                    ("mlp", ("mlp",)),
                ):
                    down_names = tuple(
                        s + f"controllers.{st}.down_weight" for st in streams
                    )
                    self.consumed.update(down_names)
                    self.model.add(
                        Parameter(
                            p + f"loop/{group_name}_down",
                            (len(streams), rank, 2 * h),
                            _stacked_bf16_source(self.base, down_names, (rank, 2 * h)),
                            direct_format="bf16",
                            residency="text",
                        )
                    )
                    up_parts = []
                    for st in streams:
                        for kind in ("gain_weight", "forget_weight"):
                            field = s + f"controllers.{st}.{kind}"
                            info = self.base.describe(field)
                            if info.shape != (h, rank) or info.dtype != "BF16":
                                raise ValueError(
                                    f"{field}: expected BF16 {(h, rank)}, got {info.dtype} {info.shape}"
                                )
                            self.consumed.add(field)
                            up_parts.append(field)
                    # [S, 2*H, rank]: gain rows then forget rows per stream
                    self.model.add(
                        Parameter(
                            p + f"loop/{group_name}_up",
                            (len(streams), 2 * h, rank),
                            _stacked_bf16_source(
                                self.base, tuple(up_parts), (h, rank)
                            ).reshape((len(streams), 2 * h, rank)),
                            direct_format="bf16",
                            residency="text",
                        )
                    )

            # MoE router + experts
            mlp_in_inputs = tuple(f"{p}mlp_in_p{k}" for k in passes)
            mlp_act_inputs = tuple(f"{p}mlp_act_p{k}" for k in passes)
            self.add(
                p + "moe/router",
                s + "mlp.gate.weight",
                (experts, h),
                inputs=mlp_in_inputs,
                direct="bf16",
            )
            self.add(
                p + "moe/correction_bias",
                s + "mlp.gate.bias",
                (experts,),
                direct="fp32",
            )
            bank = InklingExpertBankSource(
                self.base, s + "mlp.experts.", experts, h, inter
            )
            self.expert_banks[layer] = bank
            self.consumed.update(bank.names())
            for role, role_inputs in (
                ("gate", mlp_in_inputs),
                ("up", mlp_in_inputs),
                ("down", mlp_act_inputs),
            ):
                src = bank.role_bank(role)
                self.model.add(
                    Parameter(
                        p + "moe/experts/" + role,
                        src.shape,
                        src,
                        inputs=role_inputs,
                        direct_format=bank.format,
                        residency="text",
                    )
                )

    def add_vision(self, vcfg: dict) -> None:
        h = vcfg["hidden_size"]
        inter = vcfg["intermediate_size"]
        p_size = vcfg["native_patch_size"]
        merge = vcfg["adapter_merge_size"]
        grid = vcfg["posemb_grid_size"]
        proj_dim = vcfg["projection_hidden_dim"]
        out_h = vcfg["text_hidden_size"]

        e = "model.visual.backbone.embeddings."
        self.add(
            "vision/patch_embedding",
            e + "patch_embedding.weight",
            (h, 3 * p_size * p_size),
            inputs=("vision/patch_input",),
            direct="bf16",
        )
        self.add(
            "vision/position_embedding",
            e + "position_embedding",
            (grid * grid, h),
            direct="bf16",
        )
        self.add(
            "vision/class_embedding", e + "class_embedding", (h,), direct="bf16"
        )
        self.add(
            "vision/cls_position_embedding",
            e + "cls_position_embedding",
            (h,),
            direct="bf16",
        )
        self.add(
            "vision/pre_norm_weight",
            "model.visual.backbone.ln_pre.weight",
            (h,),
            direct="bf16",
        )
        self.add(
            "vision/pre_norm_bias",
            "model.visual.backbone.ln_pre.bias",
            (h,),
            direct="bf16",
        )
        for layer in range(vcfg["num_hidden_layers"]):
            p = f"vision/layers/{layer}/"
            s = f"model.visual.backbone.layers.{layer}."
            for norm in ("ln_1", "ln_2"):
                for kind in ("weight", "bias"):
                    self.add(p + norm + "_" + kind, s + norm + "." + kind, (h,), direct="bf16")
            self.add(p + "ls_1_gamma", s + "ls_1.gamma", (h,), direct="bf16")
            self.add(p + "ls_2_gamma", s + "ls_2.gamma", (h,), direct="bf16")
            for role, proj in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
                self.add(
                    p + "attention/" + role,
                    s + "self_attn." + proj + ".weight",
                    (h, h),
                    inputs=(p + "attention_input",),
                    direct="bf16",
                )
                self.add(
                    p + "attention/" + role + "_bias",
                    s + "self_attn." + proj + ".bias",
                    (h,),
                    direct="bf16",
                )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            self.group(*(p + "attention/" + r + "_bias" for r in ("query", "key", "value")))
            for role, field, dims, inp in (
                ("attention/output", "self_attn.out_proj", (h, h), "attention_output"),
                ("mlp/fc1", "mlp.c_fc", (inter, h), "mlp_input"),
                ("mlp/fc2", "mlp.c_proj", (h, inter), "mlp_activation"),
            ):
                self.add(p + role, s + field + ".weight", dims, inputs=(p + inp,), direct="bf16")
                self.add(p + role + "_bias", s + field + ".bias", (dims[0],), direct="bf16")
        self.add(
            "vision/projector/in",
            "model.visual.proj_in.weight",
            (proj_dim, merge * merge * h),
            inputs=("vision/projector/input",),
            direct="bf16",
        )
        self.add(
            "vision/projector/out",
            "model.visual.proj_out.weight",
            (out_h, proj_dim),
            inputs=("vision/projector/activation",),
            direct="bf16",
        )
        if vcfg["projector_bias"]:
            self.add(
                "vision/projector/in_bias",
                "model.visual.proj_in.bias",
                (proj_dim,),
                direct="bf16",
            )
            self.add(
                "vision/projector/out_bias",
                "model.visual.proj_out.bias",
                (out_h,),
                direct="bf16",
            )

    def add_dspark(self, mtp: SafetensorsSource, dcfg: dict) -> None:
        consumed: set[str] = set()
        h = dcfg["hidden_size"]
        num_layers = dcfg["num_hidden_layers"]
        q_dim = dcfg["num_attention_heads"] * dcfg["head_dim"]
        kv_dim = dcfg["num_key_value_heads"] * dcfg["head_dim"]
        d = dcfg["head_dim"]
        inter = dcfg["intermediate_size"]
        rank = dcfg["markov_rank"]
        v_pad = dcfg["vocab_size"]
        v_unpad = dcfg["unpadded_vocab_size"]
        num_taps = len(dcfg["target_layer_ids"])

        # Shared text parameters (aliased via recipe.share in official_recipes.py)
        for role, inputs in (
            ("token_embedding", ()),
            ("embed_norm", ()),
            ("output_head", ("dspark/final_hidden",)),
        ):
            target = self.model.parameters["text/" + role]
            self.model.add(
                Parameter(
                    "dspark/" + role,
                    target.shape,
                    target.source,
                    inputs=inputs,
                    direct_format=target.direct_format,
                    residency="dspark",
                )
            )

        # fuse_weight [L+1, num_taps] BF16 -> fuse_alpha = softmax(fuse_weight.float(), dim=-1) FP32
        fuse_info = mtp.describe("fuse_weight")
        if fuse_info.shape != (num_layers + 1, num_taps) or fuse_info.dtype != "BF16":
            raise ValueError(
                f"dspark.fuse_weight: expected BF16 {(num_layers + 1, num_taps)}, "
                f"got {fuse_info.dtype} {fuse_info.shape}"
            )
        consumed.add("fuse_weight")
        fuse_alpha = torch.softmax(
            mtp.read_flat("fuse_weight").reshape(num_layers + 1, num_taps).float(), dim=-1
        ).contiguous()
        self.model.add(
            Parameter(
                "dspark/fuse_alpha",
                (num_layers + 1, num_taps),
                array_source(fuse_alpha, f"{mtp.path}:softmax(fuse_weight)"),
                direct_format="fp32",
                residency="dspark",
            )
        )
        self.add(
            "dspark/mask_embed", "mask_embed", (h,), direct="bf16", store=mtp, consumed=consumed
        )
        self.add(
            "dspark/final_norm", "norm.weight", (h,), direct="bf16", store=mtp, consumed=consumed
        )

        for layer in range(num_layers):
            p = f"dspark/layers/{layer}/"
            s = f"layers.{layer}."
            for role, field in (
                ("dflare_norm", "dflare_norm.weight"),
                ("input_norm", "input_layernorm.weight"),
                ("post_attention_norm", "post_attention_layernorm.weight"),
            ):
                self.add(p + role, s + field, (h,), direct="bf16", store=mtp, consumed=consumed)
            for role, field, rows in (
                ("query", "q_proj", q_dim),
                ("key", "k_proj", kv_dim),
                ("value", "v_proj", kv_dim),
            ):
                self.add(
                    p + "attention/" + role,
                    s + "self_attn." + field + ".weight",
                    (rows, h),
                    inputs=(p + "attention_input",),
                    direct="bf16",
                    store=mtp,
                    consumed=consumed,
                )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            for role, field in (("context_key", "k_proj_ctx"), ("context_value", "v_proj_ctx")):
                self.add(
                    p + "attention/" + role,
                    s + "self_attn." + field + ".weight",
                    (kv_dim, h),
                    inputs=(p + "context_input",),
                    direct="bf16",
                    store=mtp,
                    consumed=consumed,
                )
            self.group(p + "attention/context_key", p + "attention/context_value")
            for role, field in (
                ("query_norm", "q_norm"),
                ("key_norm", "k_norm"),
                ("context_key_norm", "k_norm_ctx"),
            ):
                self.add(
                    p + "attention/" + role,
                    s + "self_attn." + field + ".weight",
                    (d,),
                    direct="bf16",
                    store=mtp,
                    consumed=consumed,
                )
            self.add(
                p + "attention/output",
                s + "self_attn.o_proj.weight",
                (h, q_dim),
                inputs=(p + "attention_output",),
                direct="bf16",
                store=mtp,
                consumed=consumed,
            )
            for role in ("gate", "up"):
                self.add(
                    p + "mlp/" + role,
                    s + "mlp." + role + "_proj.weight",
                    (inter, h),
                    inputs=(p + "mlp_input",),
                    direct="bf16",
                    store=mtp,
                    consumed=consumed,
                )
            self.group(p + "mlp/gate", p + "mlp/up")
            self.add(
                p + "mlp/down",
                s + "mlp.down_proj.weight",
                (h, inter),
                inputs=(p + "mlp_product",),
                direct="bf16",
                store=mtp,
                consumed=consumed,
            )

        # Markov head
        mp, ms = "dspark/markov/", "markov_head."
        self.add(
            mp + "dflare_norm",
            ms + "dflare_norm.weight",
            (h,),
            direct="bf16",
            store=mtp,
            consumed=consumed,
        )
        consumed.add(ms + "markov_w1.weight")
        self.model.add(
            Parameter(
                mp + "w1_embed",
                (v_pad, rank),
                _padded_vocab_rows_source(
                    mtp, ms + "markov_w1.weight", self.raw_vocab, v_unpad, v_pad, rank
                ),
                direct_format="bf16",
                residency="dspark",
            )
        )
        rnn_shape = (3 * rank, 2 * rank + 2 * h)
        consumed.add(ms + "rnn_proj.weight")
        for role, c0, c1, inp in (
            ("w_recurrent", 0, 2 * rank, mp + "recurrent_input"),
            ("w_hidden", 2 * rank, 2 * rank + h, mp + "hidden_input"),
            ("w_frnn", 2 * rank + h, 2 * rank + 2 * h, mp + "frnn_input"),
        ):
            self.model.add(
                Parameter(
                    mp + role,
                    (3 * rank, c1 - c0),
                    _column_slice_source(mtp, ms + "rnn_proj.weight", rnn_shape, c0, c1),
                    inputs=(inp,),
                    direct_format="bf16",
                    residency="dspark",
                )
            )
        self.add(
            mp + "rnn_bias",
            ms + "rnn_proj.bias",
            (3 * rank,),
            direct="fp32",
            store=mtp,
            consumed=consumed,
        )

        # Offline FP64 pre-folded LM head projection
        consumed.add(ms + "hidden_out.weight")
        consumed.add(ms + "hidden_out.bias")
        fold_weight, fold_bias = _precompute_dspark_fold(
            self.base, mtp, self.raw_vocab, v_unpad, v_pad, h, rank
        )
        self.model.add(
            Parameter(
                mp + "fold_weight",
                (v_pad, rank),
                array_source(fold_weight, f"fp64_fold({mtp.path}:hidden_out.weight)"),
                inputs=(mp + "latent",),
                direct_format="bf16",
                residency="dspark",
            )
        )
        self.model.add(
            Parameter(
                mp + "fold_bias",
                (v_pad,),
                array_source(fold_bias, f"fp64_fold({mtp.path}:hidden_out.bias)"),
                direct_format="bf16",
                residency="dspark",
            )
        )

        if consumed != set(mtp.weight_map):
            raise ValueError(
                f"Inkling2 DSpark source coverage differs: "
                f"missing {sorted(set(mtp.weight_map) - consumed)}, "
                f"extra {sorted(consumed - set(mtp.weight_map))}"
            )


def build_model(
    base: SafetensorsSource,
    *,
    components=("text",),
    companions=None,
    resource_overrides=None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "dspark"}:
        raise ValueError("Inkling2 supports text, vision and dspark components")
    companions = {} if companions is None else companions
    cfg, raw_vocab = text_config(base.config)
    records: dict[str, dict] = {"text": {"config": cfg}}
    roles: dict[str, tuple[str, ...]] = {"text": _TEXT_RESOURCES}

    if "vision" in selected:
        records["vision"] = {"config": vision_config(base.config, cfg), "target": "text"}
        roles["vision"] = ("processor_config.json",)
    if "dspark" in selected:
        if "dspark" not in companions:
            raise ValueError("selected dspark requires mtp.safetensors source")
        records["dspark"] = {"config": dspark_config(base.config, cfg), "target": "text"}

    refs, payloads, count, special = load_resources(
        base.root,
        vocab_size=cfg["vocab_size"],
        roles=roles,
        overrides=resource_overrides,
    )
    if count != cfg["unpadded_vocab_size"]:
        raise ValueError(
            f"inkling2: tokenizer public token count {count} != unpadded_vocab_size {cfg['unpadded_vocab_size']}"
        )
    for component, values in refs.items():
        records[component]["resources"] = values

    model = Model(
        records,
        resources=payloads,
        token_count=count,
        special_token_ids=special,
        proposal_domain="full",
    )
    builder = Builder(model, base, cfg, raw_vocab)
    builder.add_text()
    if "vision" in selected:
        builder.add_vision(records["vision"]["config"])
    if "dspark" in selected:
        builder.add_dspark(companions["dspark"], records["dspark"]["config"])

    expected = {
        name
        for name in base.weight_map
        if not (name.startswith("model.visual.") and "vision" not in selected)
    }
    if builder.consumed != expected:
        raise ValueError(
            f"Inkling2 source coverage differs: "
            f"missing {sorted(expected - builder.consumed)}, "
            f"extra {sorted(builder.consumed - expected)}"
        )
    return model
