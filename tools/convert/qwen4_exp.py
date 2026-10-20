"""Qwen4Exp architecture, checkpoint axes and logical parameter mapping."""

from __future__ import annotations

from copy import deepcopy
from math import prod, isfinite
from pathlib import Path

import torch

from .model import Model, Parameter
from .resources import load_resources
from .sources.logical import LogicalSource, select_rows, transpose_source
from .sources.safetensors import SafetensorsSource, tensor_source


def _positive(raw, key):
    value = raw.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"qwen4_exp.{key}: expected positive integer")
    return value


def text_config(source):
    if source.get("architectures") not in (
        ["Qwen4ExpForConditionalGeneration"],
        ["Qwen4ExpForCausalLM"],
    ):
        raise ValueError("expected Qwen4Exp architecture")
    raw = source.get("text_config", source)
    keys = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "max_position_embeddings",
        "hc_count",
        "hc_lowrank",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_compress_ratio",
        "indexer_budget",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "ple_embed_dim",
        "ngram_size",
        "heads_per_ngram",
        "split_ngram_parts",
        "ple_conv_kernel_size",
    )
    cfg = {key: _positive(raw, key) for key in keys}
    cfg.update(architectures=["Qwen4ExpForCausalLM"], model_type="qwen4_exp_text")
    for key, expected in (
        ("hidden_act", "silu"),
        ("output_gate_type", "sigmoid"),
        ("tie_word_embeddings", False),
        ("attention_bias", False),
        ("mamba_ssm_dtype", "float32"),
    ):
        value = raw.get(key, expected)
        if value != expected:
            raise ValueError(f"qwen4_exp.{key}: expected {expected!r}")
        cfg[key] = value
    cfg["rms_norm_eps"] = float(raw.get("rms_norm_eps", 1e-6))
    if not isfinite(cfg["rms_norm_eps"]) or cfg["rms_norm_eps"] <= 0:
        raise ValueError("invalid norm epsilon")
    cfg["norm_topk_prob"] = raw.get("norm_topk_prob", True)
    if type(cfg["norm_topk_prob"]) is not bool:
        raise ValueError("norm_topk_prob must be boolean")
    cfg["layer_types"] = [
        {
            "full_attention": "indexed_attention",
            "indexed_attention": "indexed_attention",
            "linear_attention": "linear_attention",
        }[x]
        for x in raw["layer_types"]
    ]
    if len(cfg["layer_types"]) != cfg["num_hidden_layers"]:
        raise ValueError("layer_types must cover every layer")
    cfg["ple_layer_ids"] = [x - 1 for x in raw["ple_layer_ids"]]
    if len(set(cfg["ple_layer_ids"])) != len(cfg["ple_layer_ids"]) or any(
        x < 0 or x >= cfg["num_hidden_layers"] for x in cfg["ple_layer_ids"]
    ):
        raise ValueError("invalid PLE layer positions")
    cfg["ple_conv_dilation"] = cfg["ngram_size"]
    cfg["eos_token_id"] = raw["eos_token_id"]
    cfg["rope_parameters"] = deepcopy(raw["rope_parameters"])
    for key in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        if key in source:
            cfg[key] = source[key]
    return cfg


def bank_source(store, name, shape, role):
    e, n, k = shape
    source_shape = (e, 2 * n, k) if role in ("gate", "up") else shape
    info = store.describe(name)
    if info.shape != source_shape or info.dtype != "BF16":
        raise ValueError(f"{name}: expert bank shape mismatch")
    width = n * k

    def read(begin, end):
        pieces = []
        while begin < end:
            expert, local = divmod(begin, width)
            stop = min(end, (expert + 1) * width)
            offset = (
                (2 * expert + int(role == "up")) * width
                if role in ("gate", "up")
                else expert * width
            )
            pieces.append(
                store.read_flat(name, offset + local, offset + local + stop - begin)
            )
            begin = stop
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.bfloat16)

    return LogicalSource(shape, f"{store.path}:{name}/{role}", read)


class Builder:
    def __init__(self, model, base, cfg):
        self.model, self.base, self.cfg = model, base, cfg
        self.consumed = set()

    def add(
        self,
        name,
        source,
        shape,
        *,
        inputs=(),
        direct="bf16",
        source_shape=None,
        offset=0,
        rows=None,
        transpose=None,
    ):
        shape = tuple(shape)
        original = tuple(source_shape or shape)

        def factory(store, format=None):
            if format not in (None, direct):
                raise ValueError(f"{name}: this source mapping requires {direct}")
            info = store.describe(source)
            if info.shape != original or info.dtype != "BF16":
                raise ValueError(
                    f"{source}: expected BF16 {original}, got {info.dtype} {info.shape}"
                )
            ref = tensor_source(
                store,
                source,
                shape if offset else original,
                offset=offset,
                source_shape=original,
            )
            if transpose is not None:
                ref = transpose_source(ref, transpose, shape)
            elif rows is not None:
                ref = select_rows(ref, tuple(rows))
            elif ref.shape != shape:
                ref = tensor_source(
                    store, source, shape, offset=offset, source_shape=original
                )
            return ref

        self.consumed.add(source)
        self.model.add(
            Parameter(
                name,
                shape,
                factory(self.base),
                source_factory=factory,
                inputs=tuple(inputs),
                direct_format=direct,
                residency=name.split("/")[0],
            )
        )

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def hc(self, p, s, inject=True):
        h, c, r = (self.cfg[x] for x in ("hidden_size", "hc_count", "hc_lowrank"))
        self.add(p + "norm", s + "hc_norm.weight", (c * h,))
        self.add(
            p + "down",
            s + "input_mix_weight_down.weight",
            (r, c * h),
            inputs=(p + "normalized_streams",),
        )
        self.add(
            p + "up",
            s + "input_mix_weight_up.weight",
            (c * h, r),
            inputs=(p + "lowrank_activation",),
        )
        if inject:
            self.add(
                p + "inject",
                s + "block_inject_weight.weight",
                (c, c * h),
                inputs=(p + "normalized_streams",),
            )
            self.group(p + "down", p + "inject")

    def gdn(self, p, s):
        h = self.cfg["hidden_size"]
        nv = self.cfg["linear_num_value_heads"]
        dv = self.cfg["linear_value_head_dim"]
        q = self.cfg["linear_num_key_heads"] * self.cfg["linear_key_head_dim"]
        v = nv * dv
        c = 2 * q + v
        for role, a, b in [("query", 0, q), ("key", q, 2 * q), ("value", 2 * q, c)]:
            self.add(
                p + "gdn/" + role,
                s + "linear_attn.in_proj_qkv.weight",
                (b - a, h),
                source_shape=(c, h),
                rows=((a, b),),
                inputs=(p + "attention_input",),
            )
        for role, field, n in [
            ("z", "in_proj_z", v),
            ("a_projection", "in_proj_a", nv),
            ("b_projection", "in_proj_b", nv),
        ]:
            self.add(
                p + "gdn/" + role,
                s + "linear_attn." + field + ".weight",
                (n, h),
                inputs=(p + "attention_input",),
            )
        self.add(
            p + "gdn/output",
            s + "linear_attn.out_proj.weight",
            (h, v),
            inputs=(p + "gdn/gated_output",),
        )
        taps = self.cfg["linear_conv_kernel_dim"]
        self.add(
            p + "gdn/convolution",
            s + "linear_attn.conv1d.weight",
            (taps, c),
            source_shape=(c, 1, taps),
            transpose=(2, 0, 1),
        )
        for role, field in [("a_log", "A_log"), ("dt_bias", "dt_bias")]:
            self.add(
                p + "gdn/" + role, s + "linear_attn." + field, (nv,), direct="fp32"
            )
        self.add(p + "gdn/norm", s + "linear_attn.norm.weight", (dv,))
        self.group(*(p + "gdn/" + x for x in ("query", "key", "value", "z")))
        self.group(p + "gdn/a_projection", p + "gdn/b_projection")

    def qsa(self, p, s, mtp=False):
        h = self.cfg["hidden_size"]
        d = self.cfg["head_dim"]
        heads = self.cfg["num_attention_heads"]
        q = heads * d
        kv = self.cfg["num_key_value_heads"] * d
        for role, half in [("query", 0), ("output_gate", 1)]:
            self.add(
                p + "attention/" + role,
                s + "self_attn.q_proj.weight",
                (q, h),
                source_shape=(2 * q, h),
                rows=tuple(
                    (j * 2 * d + half * d, j * 2 * d + (half + 1) * d)
                    for j in range(heads)
                ),
                inputs=(p + "attention_input",),
            )
        for role in ("key", "value"):
            self.add(
                p + "attention/" + role,
                s + "self_attn." + role[0] + "_proj.weight",
                (kv, h),
                inputs=(p + "attention_input",),
            )
        iq = self.cfg["indexer_n_heads"] * self.cfg["indexer_head_dim"]
        ik = self.cfg["indexer_kv_heads"] * self.cfg["indexer_head_dim"]
        for role, a, b in [("index_query", 0, iq), ("index_key", iq, iq + ik)]:
            self.add(
                p + "attention/" + role,
                s + "self_attn.indexer.index_qk_proj.weight",
                (b - a, h),
                source_shape=(iq + ik, h),
                rows=((a, b),),
                inputs=(p + "attention_input",),
            )
        self.add(
            p + "attention/output",
            s + "self_attn.o_proj.weight",
            (h, q),
            inputs=(p + "attention/gated_output",),
        )
        for role, field, n in [
            ("query_norm", "q_norm", d),
            ("key_norm", "k_norm", d),
            ("index_query_norm", "indexer.q_layernorm", self.cfg["indexer_head_dim"]),
            ("index_key_norm", "indexer.k_layernorm", self.cfg["indexer_head_dim"]),
        ]:
            self.add(
                p + "attention/" + role, s + "self_attn." + field + ".weight", (n,)
            )
        groups = (
            [("key", "value", "index_key", "query", "output_gate", "index_query")]
            if mtp
            else [
                ("query", "output_gate"),
                ("key", "value", "index_query", "index_key"),
            ]
        )
        for group in groups:
            self.group(*(p + "attention/" + x for x in group))

    def moe(self, p, s):
        h, e, i, is_ = (
            self.cfg[x]
            for x in (
                "hidden_size",
                "num_experts",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            )
        )
        for role, field, n in [
            ("router", "gate", e),
            ("shared_score", "shared_expert_gate", 1),
            ("shared/gate", "shared_expert.gate_proj", is_),
            ("shared/up", "shared_expert.up_proj", is_),
        ]:
            self.add(
                p + "moe/" + role,
                s + "mlp." + field + ".weight",
                (n, h),
                inputs=(p + "moe_input",),
            )
        self.group(
            *(
                p + "moe/" + x
                for x in ("router", "shared_score", "shared/gate", "shared/up")
            )
        )
        self.add(
            p + "moe/shared/down",
            s + "mlp.shared_expert.down_proj.weight",
            (h, is_),
            inputs=(p + "moe/shared/product",),
        )
        for role in ("gate", "up", "down"):
            shape = (e, h, i) if role == "down" else (e, i, h)
            source = (
                s + "mlp.experts." + ("down_proj" if role == "down" else "gate_up_proj")
            )
            self.consumed.add(source)
            ref = bank_source(self.base, source, shape, role)
            inp = p + "moe/experts/product" if role == "down" else p + "moe_input"
            name = p + "moe/experts/" + role
            self.model.add(
                Parameter(name, shape, ref, inputs=(inp,), residency=p.split("/")[0])
            )

    def ple(self, p, s):
        h, c, ed = (self.cfg[x] for x in ("hidden_size", "hc_count", "ple_embed_dim"))
        for role, n in [("key", h * c), ("value", h)]:
            self.add(
                p + "ple/" + role,
                s + "ple." + role + "_proj.weight",
                (n, ed),
                inputs=(p + "ple/embedding",),
            )
        self.group(p + "ple/key", p + "ple/value")
        for role in ("norm_key", "norm_query", "norm_conv"):
            self.add(p + "ple/" + role, s + "ple." + role + ".weight", (h * c,))
        taps = self.cfg["ple_conv_kernel_size"]
        self.add(
            p + "ple/convolution",
            s + "ple.conv1d.weight",
            (taps, h * c),
            source_shape=(h * c, 1, taps),
            transpose=(2, 0, 1),
        )
        names = [
            s + f"ple.ple_embedding.ngram_embedding.shard_{j}.weight"
            for j in range(self.cfg["split_ngram_parts"])
        ]
        shard_shape = self.base.describe(names[0]).shape
        for name in names:
            if (
                self.base.describe(name).shape != shard_shape
                or self.base.describe(name).dtype != "BF16"
            ):
                raise ValueError("inconsistent PLE shards")
        width = prod(shard_shape)

        def read(begin, end):
            chunks = []
            while begin < end:
                j, local = divmod(begin, width)
                stop = min(end, (j + 1) * width)
                chunks.append(
                    self.base.read_flat(names[j], local, local + stop - begin)
                )
                begin = stop
            return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.bfloat16)

        shape = (len(names) * shard_shape[0], shard_shape[1])
        self.consumed.update(names)
        self.model.add(
            Parameter(
                p + "ple/table", shape, LogicalSource(shape, "PLE shard bank", read)
            )
        )
        controls = []
        for role, n in [
            ("layer_multipliers", self.cfg["ngram_size"]),
            (
                "ngram_heads_vocab_sizes",
                (self.cfg["ngram_size"] - 1) * self.cfg["heads_per_ngram"],
            ),
            (
                "ngram_heads_offsets",
                (self.cfg["ngram_size"] - 1) * self.cfg["heads_per_ngram"],
            ),
        ]:
            name = s + "ple.ple_embedding." + role
            info = self.base.describe(name)
            if info.dtype != "I64" or info.shape != (n,):
                raise ValueError(f"{name}: invalid hash control")
            controls.append(
                self.base.read_flat(name).view(torch.uint8).numpy().tobytes()
            )
            self.consumed.add(name)
        rid = "resource/text/ple_hash_controls"
        self.model.resources[rid] = b"".join(controls)
        self.model.components["text"]["resources"]["ple_hash_controls"] = rid
        self.cfg["ple_shard_shape"] = list(shard_shape)

    def vision(self, cfg):
        h = cfg["hidden_size"]
        i = cfg["intermediate_size"]
        k = cfg["in_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
        self.add(
            "vision/patch_embedding",
            "model.visual.patch_embed.proj.weight",
            (h, k),
            source_shape=(
                h,
                cfg["in_channels"],
                cfg["temporal_patch_size"],
                cfg["patch_size"],
                cfg["patch_size"],
            ),
            inputs=("vision/patch_input",),
        )
        self.add(
            "vision/patch_embedding_bias", "model.visual.patch_embed.proj.bias", (h,)
        )
        self.add(
            "vision/position_embedding",
            "model.visual.pos_embed.weight",
            (cfg["num_position_embeddings"], h),
        )
        for layer in range(cfg["depth"]):
            p = f"vision/layers/{layer}/"
            s = f"model.visual.blocks.{layer}."
            for norm in ("norm1", "norm2"):
                for kind in ("weight", "bias"):
                    self.add(p + norm + "_" + kind, s + norm + "." + kind, (h,))
            for j, role in enumerate(("query", "key", "value")):
                self.add(
                    p + "attention/" + role,
                    s + "attn.qkv.weight",
                    (h, h),
                    source_shape=(3 * h, h),
                    rows=((j * h, (j + 1) * h),),
                    inputs=(p + "attention_input",),
                )
                self.add(
                    p + "attention/" + role + "_bias",
                    s + "attn.qkv.bias",
                    (h,),
                    source_shape=(3 * h,),
                    offset=j * h,
                )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            self.group(
                *(p + "attention/" + r + "_bias" for r in ("query", "key", "value"))
            )
            for role, field, n, k, inp in [
                ("attention/output", "attn.proj", h, h, "attention_output"),
                ("mlp/fc1", "mlp.linear_fc1", i, h, "mlp_input"),
                ("mlp/fc2", "mlp.linear_fc2", h, i, "mlp_activation"),
            ]:
                self.add(p + role, s + field + ".weight", (n, k), inputs=(p + inp,))
                self.add(p + role + "_bias", s + field + ".bias", (n,))
        s = "model.visual.merger."
        p = "vision/merger/"
        m = h * cfg["spatial_merge_size"] ** 2
        for kind in ("weight", "bias"):
            self.add(p + "norm_" + kind, s + "norm." + kind, (h,))
        for role, n, k, inp in [
            ("fc1", m, m, "input"),
            ("fc2", self.cfg["hidden_size"], m, "activation"),
        ]:
            self.add(
                p + role, s + "linear_" + role + ".weight", (n, k), inputs=(p + inp,)
            )
            self.add(p + role + "_bias", s + "linear_" + role + ".bias", (n,))


def build_model(
    base, *, components=("text",), companions=None, resource_overrides=None
):
    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "mtp"}:
        raise ValueError("Qwen4Exp supports text, vision and mtp components")
    cfg = text_config(base.config)
    records = {"text": {"config": cfg}}
    if "vision" in selected:
        v = deepcopy(base.config["vision_config"])
        v["model_type"] = "qwen4_exp_vision"
        if (
            v.get("deepstack_visual_indexes", [])
            or v["out_hidden_size"] != cfg["hidden_size"]
        ):
            raise ValueError("unsupported Vision merger/deepstack configuration")
        records["vision"] = {"config": v, "target": "text"}
    if "mtp" in selected:
        m = deepcopy(base.config["text_config"]["mtp"])
        if (
            m["num_hidden_layers"] != 1
            or m.get("mtp_use_hidden_state_from_layer") is not None
            or m["layer_types"] != ["full_attention"]
            or base.config["text_config"].get("mtp_use_dedicated_embeddings", False)
        ):
            raise ValueError("unsupported Qwen4Exp MTP structure")
        m.update(
            architectures=["Qwen4ExpMTP"],
            model_type="qwen4_exp_mtp",
            layer_types=["indexed_attention"],
        )
        records["mtp"] = {"config": m, "target": "text"}
    overrides = {
        "chat_template.jinja": Path(__file__).resolve().parents[1]
        / "chat_templates"
        / "qwen3_8_flash_next.jinja",
        **(resource_overrides or {}),
    }
    refs, res, count, special = load_resources(
        base.root,
        vocab_size=cfg["vocab_size"],
        vision_config=records.get("vision", {}).get("config"),
        overrides=overrides,
    )
    for c, values in refs.items():
        records[c]["resources"] = values
    model = Model(records, resources=res, token_count=count, special_token_ids=special)
    b = Builder(model, base, cfg)
    h = cfg["hidden_size"]
    b.add(
        "text/token_embedding",
        "model.language_model.embed_tokens.weight",
        (cfg["vocab_size"], h),
    )
    b.add(
        "text/output_head",
        "lm_head.weight",
        (cfg["vocab_size"], h),
        inputs=("text/final_hidden",),
    )
    b.hc("text/final_hc/", "model.language_model.hyper_connection_mixer.", False)
    for l, kind in enumerate(cfg["layer_types"]):
        p = f"text/layers/{l}/"
        s = f"model.language_model.layers.{l}."
        b.hc(p + "attention_hc/", s + "attn_hyper_connection.")
        b.hc(p + "moe_hc/", s + "mlp_hyper_connection.")
        b.moe(p, s)
        if kind == "linear_attention":
            b.gdn(p, s)
        else:
            b.qsa(p, s)
        if l in cfg["ple_layer_ids"]:
            b.ple(p, s)
    if "mtp" in selected:
        for role, target in [
            ("token_embedding", "text/token_embedding"),
            ("output_head", "text/output_head"),
        ]:
            original = model.parameters[target]
            model.add(
                Parameter(
                    "mtp/" + role,
                    original.shape,
                    original.source,
                    inputs=("mtp/final_hidden",) if role == "output_head" else (),
                    residency="mtp",
                )
            )
        b.hc("mtp/final_hc/", "mtp.hyper_connection_mixer.", False)
        for role, field, n in [
            ("pre_fc_embedding_norm", "pre_fc_norm_embedding", h),
            ("pre_fc_hidden_norm", "pre_fc_norm_hidden", h * cfg["hc_count"]),
        ]:
            b.add("mtp/" + role, "mtp." + field + ".weight", (n,))
        for role, inp in [
            ("fc_embedding", "normalized_embedding"),
            ("fc_hidden", "normalized_hidden_streams"),
        ]:
            b.add(
                "mtp/" + role, "mtp." + role + ".weight", (h, h), inputs=("mtp/" + inp,)
            )
        p = "mtp/layers/0/"
        s = "mtp.layers.0."
        b.hc(p + "attention_hc/", s + "attn_hyper_connection.")
        b.hc(p + "moe_hc/", s + "mlp_hyper_connection.")
        b.qsa(p, s, True)
        b.moe(p, s)
    if "vision" in selected:
        b.vision(records["vision"]["config"])
    expected = {
        key
        for key in base.weight_map
        if not (key.startswith("model.visual.") and "vision" not in selected)
        and not (key.startswith("mtp.") and "mtp" not in selected)
    }
    if b.consumed != expected:
        raise ValueError(
            f"Qwen4Exp source coverage differs: missing {sorted(expected-b.consumed)}, extra {sorted(b.consumed-expected)}"
        )
    return model
