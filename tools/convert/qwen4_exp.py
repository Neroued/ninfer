"""Qwen4Exp architecture adapter (Qwen3.8-Flash-Next): config, logical parameters, groups.

The mathematics is defined in docs/maintainer/qwen4-exp-model.md. Shared pieces (gated
attention, GDN, a dense SwiGLU FFN, Vision) reuse the Qwen3.5 source mapping; hyper-connections,
the QSA indexer, the n-gram PLE embedding and the expert banks are mapped here.

The FFN is the routed-expert MoE (``num_experts`` ...) or, for a dense Qwen4 checkpoint, the
Qwen3.5 dense MLP (``intermediate_size``; ``mlp.{gate,up,down}_proj``, MTP included). The PLE is
optional: no (or an empty) ``ple_layer_ids`` converts no n-gram embedding and writes no n-gram
field.

Routed expert sources are resolved per expert. A checkpoint that stores fused BF16
``experts.gate_up_proj``/``down_proj`` banks exposes them as row ranges; a ModelOpt NVFP4
checkpoint exposes one encoded matrix per expert and role (its block scales possibly CSF-compressed);
a block-FP8 MTP bank is read as values. The n-gram table is BF16, row-FP8 shards under one multiplier,
or (FP4-Local) row-NVFP4 shards under one ``weight_scale_2``, which then also becomes the one-element
FP32 parameter ``ple/table_global_scale``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .model import Model, Parameter
from .qwen3_5 import (
    _Builder,
    _f32,
    _fixed,
    _positive,
    text_fields as qwen3_5_text_fields,
    vision_config as qwen3_5_vision_config,
)
from .resources import load_resources
from .sources.logical import LogicalSource
from .sources.modelopt import (
    fp8_block_source,
    fp8_tensor_rows_source,
    nvfp4_row_global_scale,
    nvfp4_row_table_source,
    nvfp4_source,
)
from .sources.safetensors import SafetensorsSource, tensor_source

# Source architectures of the Qwen4Exp mathematics (Qwen3.8-Flash-Next); the artifact's Text is
# Qwen4ExpForCausalLM (qwen4_exp_text) and its MTP layer Qwen4ExpMTP.
ARCHITECTURES = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")

_QWEN4_FIELDS = (
    "hc_count",
    "hc_lowrank",
    "indexer_n_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)
_PLE_FIELDS = (
    "ple_embed_dim",
    "ple_conv_kernel_size",
    "ngram_size",
    "heads_per_ngram",
    "ngram_vocab_size_base",
    "make_ngram_vocab_size_divisible_by",
)
_MOE_FIELDS = ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size")


def is_qwen4_exp(config: dict) -> bool:
    architectures = config.get("architectures")
    return isinstance(architectures, list) and bool(set(architectures) & set(ARCHITECTURES))


def is_dense(raw: dict) -> bool:
    """A dense Qwen4 FFN: ``intermediate_size`` and no routed experts."""
    experts = raw.get("num_experts")
    if experts in (None, 0):
        if "intermediate_size" not in raw:
            raise ValueError("text: a Qwen4 config needs num_experts (MoE) or intermediate_size (dense)")
        return True
    return False


def text_config(source: dict, *, mtp: bool) -> dict:
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or not is_qwen4_exp(source):
        raise ValueError(f"unsupported Qwen4Exp architecture {architectures!r}")
    raw = source.get("text_config", source)
    dense = is_dense(raw)
    # The shared fields follow the Qwen3.5 normalization (dense, or the routed-expert fields). A dense
    # config may carry the HF MoE defaults: only intermediate_size is read.
    base_raw = {k: v for k, v in raw.items() if k not in _MOE_FIELDS} if dense else raw
    shared = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text"}
    shared.update(qwen3_5_text_fields(base_raw, mtp=mtp, moe=not dense))
    if not dense:
        _fixed(raw, "norm_topk_prob", True, "text")
    _fixed(raw, "indexer_kv_heads", 1, "text")
    _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
    gate = raw.get("output_gate_type") or raw.get("hidden_act", "silu")
    if gate not in ("sigmoid", "silu"):
        raise ValueError(f"text.output_gate_type: unsupported {gate!r}")
    shared["output_gate_type"] = gate
    for key in _QWEN4_FIELDS:
        shared[key] = _positive(raw.get(key), "text." + key)
    shared["indexer_kv_heads"] = 1
    layers = raw.get("ple_layer_ids") or []
    if layers:
        if (
            not isinstance(layers, list)
            or len(layers) != 1
            or type(layers[0]) is not int
            or not 1 <= layers[0] <= shared["num_hidden_layers"]
            or shared["layer_types"][layers[0] - 1] != "linear_attention"
        ):
            raise ValueError("text.ple_layer_ids must be empty or name exactly one one-indexed GDN block")
        for key in _PLE_FIELDS:
            shared[key] = _positive(raw.get(key), "text." + key)
        seed = raw.get("seed", 1234)
        if type(seed) is not int or seed < 0:
            raise ValueError("text.seed must be a nonnegative integer")
        shared["seed"] = seed
        eos = raw.get("eos_token_id", source.get("eos_token_id"))
        eos = eos[0] if isinstance(eos, list) and eos else eos
        if type(eos) is not int or not 0 <= eos < shared["vocab_size"]:
            raise ValueError("text.eos_token_id must name one embedding row")
        shared["eos_token_id"] = eos
        shared["ple_layer_ids"] = list(layers)
        heads = (shared["ngram_size"] - 1) * shared["heads_per_ngram"]
        if shared["ngram_size"] < 2 or shared["ple_embed_dim"] % heads:
            raise ValueError("n-gram heads must divide the PLE embedding width")
    if shared["indexer_budget"] % shared["indexer_compress_ratio"]:
        raise ValueError("QSA budget must be a whole number of blocks")
    rotary = int(shared["head_dim"] * shared["rope_parameters"]["partial_rotary_factor"])
    if rotary > shared["indexer_head_dim"]:
        raise ValueError("QSA indexer head is narrower than the rotary width")
    if shared["hc_count"] < 2:
        raise ValueError("hyper-connections need at least two streams")
    shared["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    return shared


def vision_config(source: dict, text: dict) -> dict:
    result = qwen3_5_vision_config(source, text)
    result["model_type"] = "qwen4_exp_vision"
    return result


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def ngram_table_rows(config: dict) -> int:
    """Padded row count of the single PLE layer's n-gram table."""
    heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
    prime, total = config["ngram_vocab_size_base"] - 1, 0
    for _ in range(heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        total += prime
    divisor = config["make_ngram_vocab_size_divisible_by"]
    return -(-total // divisor) * divisor


class _Qwen4Builder(_Builder):
    """Adds Qwen4Exp roles; Qwen3.5 roles (attention, GDN, Vision) keep their mapping."""

    def value_source(self, name, shape, factory, *, inputs=(), direct="bf16"):
        self.model.add(
            Parameter(
                name,
                tuple(shape),
                factory(),
                None,
                tuple(inputs),
                direct,
                residency=name.split("/", 1)[0],
            )
        )

    def hyper_connection(self, prefix, source_prefix, store, config, *, inject=True):
        h, hc, rank = config["hidden_size"], config["hc_count"], config["hc_lowrank"]
        wide = hc * h
        self.add(prefix + "norm", store, source_prefix + "hc_norm.weight", (wide,))
        self.add(
            prefix + "down",
            store,
            source_prefix + "input_mix_weight_down.weight",
            (rank, wide),
            inputs=(prefix + "normed",),
        )
        self.add(
            prefix + "up",
            store,
            source_prefix + "input_mix_weight_up.weight",
            (wide, rank),
            inputs=(prefix + "lowrank",),
        )
        if inject:
            self.add(
                prefix + "inject",
                store,
                source_prefix + "block_inject_weight.weight",
                (hc, wide),
                inputs=(prefix + "normed",),
            )
            self.group(prefix + "down", prefix + "inject")

    def indexer(self, prefix, source_prefix, store, config):
        h, di = config["hidden_size"], config["indexer_head_dim"]
        heads = config["indexer_n_heads"]
        name = source_prefix + "self_attn.indexer.index_qk_proj.weight"
        for role, begin, rows in (("query", 0, heads * di), ("key", heads * di, di)):
            self.add(
                prefix + "attention/indexer_" + role,
                store,
                name,
                (rows, h),
                source_shape=((heads + 1) * di, h),
                rows=((begin, begin + rows),),
                inputs=(prefix + "mixer_input",),
            )
        self.group(prefix + "attention/indexer_query", prefix + "attention/indexer_key")
        for role, field in (("query_norm", "q_layernorm"), ("key_norm", "k_layernorm")):
            self.add(
                prefix + "attention/indexer_" + role,
                store,
                source_prefix + "self_attn.indexer." + field + ".weight",
                (di,),
            )

    def experts(self, prefix, source_prefix, store, config):
        """Routed experts as per-expert matrices, whatever the checkpoint representation."""
        h, e, ir = config["hidden_size"], config["num_experts"], config["moe_intermediate_size"]
        sp, p = source_prefix + "mlp.experts.", prefix + "moe/"
        input_name = prefix + "ffn_input"
        fused = store.has(sp + "gate_up_proj")
        gate_up, downs = [], []
        for expert in range(e):
            ep = p + f"experts/{expert}/"
            for index, role in enumerate(("gate", "up")):
                shape = (ir, h)
                if fused:
                    self.add(
                        ep + role,
                        store,
                        sp + "gate_up_proj",
                        shape,
                        source_shape=(e, 2 * ir, h),
                        offset=(expert * 2 + index) * ir * h,
                        inputs=(input_name,),
                    )
                else:
                    leaf = f"{sp}{expert}.{role}_proj"
                    self.value_source(
                        ep + role,
                        shape,
                        lambda leaf=leaf, shape=shape: _expert_source(store, leaf, shape),
                        inputs=(input_name,),
                    )
                gate_up.append(ep + role)
            shape = (h, ir)
            if fused:
                self.add(
                    ep + "down",
                    store,
                    sp + "down_proj",
                    shape,
                    source_shape=(e, h, ir),
                    offset=expert * h * ir,
                    inputs=(ep + "product",),
                )
            else:
                leaf = f"{sp}{expert}.down_proj"
                self.value_source(
                    ep + "down",
                    shape,
                    lambda leaf=leaf, shape=shape: _expert_source(store, leaf, shape),
                    inputs=(ep + "product",),
                )
            downs.append(ep + "down")
        self.group(*gate_up)
        self.group(*downs)

    def moe(self, prefix, source_prefix, store, config):
        h, e = config["hidden_size"], config["num_experts"]
        shared = config["shared_expert_intermediate_size"]
        sp, p = source_prefix + "mlp.", prefix + "moe/"
        input_name = prefix + "ffn_input"
        self.add(p + "router", store, sp + "gate.weight", (e, h), inputs=(input_name,))
        self.add(
            p + "shared_score",
            store,
            sp + "shared_expert_gate.weight",
            (1, h),
            inputs=(input_name,),
        )
        self.group(p + "router", p + "shared_score")
        self.experts(prefix, source_prefix, store, config)
        for role in ("gate", "up"):
            self.add(
                p + "shared/" + role,
                store,
                sp + "shared_expert." + role + "_proj.weight",
                (shared, h),
                inputs=(input_name,),
            )
        self.add(
            p + "shared/down",
            store,
            sp + "shared_expert.down_proj.weight",
            (h, shared),
            inputs=(p + "shared/product",),
        )
        self.group(p + "shared/gate", p + "shared/up")

    def ple(self, prefix, source_prefix, store, config):
        h, hc = config["hidden_size"], config["hc_count"]
        wide, width = hc * h, config["ple_embed_dim"]
        heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
        rows = ngram_table_rows(config)
        sp, p = source_prefix + "ple.", prefix + "ple/"
        table = sp + "ple_embedding.ngram_embedding"
        self.value_source(
            p + "table",
            (rows, width // heads),
            lambda: _table_source(store, table, rows, width // heads),
        )
        if _row_nvfp4_table(store, table):
            # Row-NVFP4 tables share one FP32 multiplier G; the engine reads it beside the table.
            self.value_source(
                p + "table_global_scale",
                (1,),
                lambda: nvfp4_row_global_scale(store, table + ".weight_scale_2"),
                direct="fp32",
            )
        self.add(p + "key", store, sp + "key_proj.weight", (wide, width), inputs=(p + "embedding",))
        self.add(p + "value", store, sp + "value_proj.weight", (h, width), inputs=(p + "embedding",))
        self.group(p + "key", p + "value")
        for role, field in (
            ("key_norm", "norm_key"),
            ("query_norm", "norm_query"),
            ("conv_norm", "norm_conv"),
        ):
            self.add(p + role, store, sp + field + ".weight", (wide,))
        taps = config["ple_conv_kernel_size"]
        self.add(
            p + "convolution",
            store,
            sp + "conv1d.weight",
            (taps, wide),
            source_shape=(wide, 1, taps),
            transpose=(2, 0, 1),
        )

    def block(self, prefix, source_prefix, store, config, mixer, *, ple=False):
        self.hyper_connection(prefix + "attn_hc/", source_prefix + "attn_hyper_connection.", store, config)
        self.hyper_connection(prefix + "ffn_hc/", source_prefix + "mlp_hyper_connection.", store, config)
        if mixer == "full_attention":
            self.attention(prefix, source_prefix, store, config)
            self.indexer(prefix, source_prefix, store, config)
        else:
            self.gdn(prefix, source_prefix, store, config)
        if "num_experts" in config:
            self.moe(prefix, source_prefix, store, config)
        else:
            self.dense(prefix, source_prefix, store, config["hidden_size"], config["intermediate_size"])
        if ple:
            self.ple(prefix, source_prefix, store, config)


def _expert_source(store: SafetensorsSource, leaf: str, shape) -> LogicalSource:
    if store.has(leaf + ".weight_scale_2"):
        return nvfp4_source(store, leaf, shape)
    if store.has(leaf + ".weight_scale_inv"):
        return fp8_block_source(store, leaf, shape)
    return tensor_source(store, leaf + ".weight", shape)


def _row_nvfp4_table(store: SafetensorsSource, prefix: str) -> bool:
    """A sharded n-gram table with per-shard E4M3 row-block scales and one ``weight_scale_2`` (FP4-Local)."""
    return store.has(prefix + ".weight_scale_2") and store.has(prefix + ".shard_0.weight_scale")


def _table_source(store: SafetensorsSource, prefix: str, rows: int, width: int) -> LogicalSource:
    shards = []
    while store.has(f"{prefix}.shard_{len(shards)}.weight"):
        shards.append(f"{prefix}.shard_{len(shards)}.weight")
    if not shards:
        return tensor_source(store, prefix + ".weight", (rows, width))
    if store.has(prefix + ".weight_scale"):
        return fp8_tensor_rows_source(store, shards, prefix + ".weight_scale", (rows, width))
    if _row_nvfp4_table(store, prefix):
        names = [name.removesuffix(".weight") for name in shards]
        return nvfp4_row_table_source(store, names, prefix + ".weight_scale_2", (rows, width))
    parts = [tensor_source(store, name, tuple(store.describe(name).shape)) for name in shards]
    counts = [part.shape[0] for part in parts]
    if sum(counts) != rows:
        raise ValueError(f"{prefix}: shard rows {sum(counts)} differ from {rows}")

    def read(begin: int, end: int):
        import torch

        pieces, cursor = [], 0
        for part in parts:
            size = part.shape[0] * width
            low, high = max(begin, cursor), min(end, cursor + size)
            if low < high:
                pieces.append(part.values(low - cursor, high - cursor))
            cursor += size
        return torch.cat(pieces) if len(pieces) != 1 else pieces[0]

    return LogicalSource((rows, width), f"{store.path}:{prefix} ({len(shards)} shards)", read)


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "mtp"}:
        raise ValueError("Qwen4Exp supports text, vision and mtp components")
    config = text_config(base.config, mtp="mtp" in selected)
    records = {"text": {"config": config}}
    if "vision" in selected:
        records["vision"] = {"config": vision_config(base.config, config), "target": "text"}
    if "mtp" in selected:
        records["mtp"] = {"config": {"architectures": ["Qwen4ExpMTP"]}, "target": "text"}
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=records["vision"]["config"] if "vision" in selected else None,
        overrides=resource_overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Qwen4Builder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    builder.add("text/token_embedding", base, text_prefix + "embed_tokens.weight", (r, h))
    head_inputs = ("text/final_hidden",) + (("mtp/final_hidden",) if "mtp" in selected else ())
    builder.add("text/output_head", base, "lm_head.weight", (r, h), inputs=head_inputs)
    builder.hyper_connection("text/hc_head/", text_prefix + "hyper_connection_mixer.", base, config, inject=False)
    ple_block = config["ple_layer_ids"][0] - 1 if "ple_layer_ids" in config else -1
    for i, kind in enumerate(config["layer_types"]):
        builder.block(
            f"text/layers/{i}/",
            text_prefix + f"layers.{i}.",
            base,
            config,
            kind,
            ple=i == ple_block,
        )
    if "mtp" in selected:
        wide = config["hc_count"] * h
        for role, field, shape in (
            ("embedding_norm", "pre_fc_norm_embedding", (h,)),
            ("hidden_norm", "pre_fc_norm_hidden", (wide,)),
        ):
            builder.add("mtp/" + role, base, "mtp." + field + ".weight", shape)
        builder.add(
            "mtp/embedding_projection",
            base,
            "mtp.fc_embedding.weight",
            (h, h),
            inputs=("mtp/embedding_input",),
        )
        builder.add(
            "mtp/hidden_projection",
            base,
            "mtp.fc_hidden.weight",
            (h, h),
            inputs=("mtp/hidden_input",),
        )
        builder.hyper_connection("mtp/hc_head/", "mtp.hyper_connection_mixer.", base, config, inject=False)
        builder.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention")
    if "vision" in selected:
        builder.vision(base, records["vision"]["config"], h)
    return model
