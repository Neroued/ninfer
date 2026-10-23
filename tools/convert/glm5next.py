"""GLM5Next checkpoint axes, logical parameters and component resources."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import prod

import torch

from .model import Model, Parameter
from .resources import TEXT_RESOURCES, load_resources
from .sources.logical import LogicalSource, select_rows
from .sources.modelopt_nvfp4 import ModelOptNvfp4
from .sources.safetensors import tensor_source


def text_config(source):
    if source.get("architectures") not in (
        ["Glm5NextForConditionalGeneration"],
        ["Glm5NextForCausalLM"],
    ):
        raise ValueError("expected Glm5Next architecture")
    raw = source.get("text_config", source)
    keys = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "max_position_embeddings",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mla_use_nope",
        "index_n_heads",
        "index_head_dim",
        "index_kpool",
        "index_topk",
        "index_kpool_always_select_tail",
        "index_kpool_compress",
        "index_share_for_mtp_iteration",
        "indexer_types",
        "indexer_rope_interleave",
        "linear_num_heads",
        "linear_head_dim",
        "linear_conv_kernel_dim",
        "linear_lower_bound",
        "hc_mult",
        "hc_eps",
        "hc_sinkhorn_iters",
        "mhc",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "n_group",
        "topk_group",
        "norm_topk_prob",
        "routed_scaling_factor",
        "scoring_func",
        "topk_method",
        "moe_router_dtype",
        "swiglu_limit",
        "hidden_act",
        "rms_norm_eps",
        "attention_bias",
        "tie_word_embeddings",
        "layer_types",
        "mlp_layer_types",
        "num_nextn_predict_layers",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    cfg = {key: deepcopy(raw[key]) for key in keys}
    cfg.update(architectures=["Glm5NextForCausalLM"], model_type="glm5next_text")
    for key in ("layer_types", "mlp_layer_types"):
        if len(cfg[key]) != cfg["num_hidden_layers"]:
            raise ValueError(f"glm5next.{key}: must cover every main layer")
    if set(cfg["layer_types"]) - {"linear_attention", "deepseek_sparse_attention"}:
        raise ValueError("unsupported GLM5Next attention mathematics")
    if set(cfg["mlp_layer_types"]) - {"dense", "sparse"}:
        raise ValueError("unsupported GLM5Next MLP mathematics")
    for key in (
        "image_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "video_token_id",
        "video_start_token_id",
        "video_end_token_id",
    ):
        if key in source:
            cfg[key] = source[key]
    return cfg


def _nvfp4_source(store, prefix, shape):
    n, k = shape
    for suffix, dims, dtype in (
        ("weight", (n, k // 2), "U8"),
        ("weight_scale", (n, k // 16), "F8_E4M3"),
        ("weight_scale_2", (), "F32"),
        ("input_scale", (), "F32"),
    ):
        info = store.describe(prefix + "." + suffix)
        if info.shape != dims or info.dtype != dtype:
            raise ValueError(f"{prefix}.{suffix}: expected {dtype} {dims}")

    @lru_cache(maxsize=1)
    def source():
        return ModelOptNvfp4(store).matrix(prefix, shape)

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (modelopt nvfp4)",
        lambda a, b: source().values(a, b),
        lambda a, b: source().read_encoded(a, b),
        lambda: source().weight_divisor(),
        lambda: source().input_divisor(),
        encoded_format="nvfp4",
    )


def bank_source(sources, shape):
    width = prod(shape[1:])

    def read(begin, end):
        parts = []
        while begin < end:
            expert, local = divmod(begin, width)
            stop = min(end, (expert + 1) * width)
            parts.append(sources[expert].values(local, local + stop - begin))
            begin = stop
        return torch.cat(parts) if parts else sources[0].values(0, 0)

    return LogicalSource(
        shape, f"bank({sources[0].label}, {len(sources)} experts)", read
    )


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
        axes=None,
    ):
        shape = tuple(shape)
        original = tuple(source_shape or shape)
        dtype = {"bf16": "BF16", "fp32": "F32"}[direct]

        def factory(store, format=None):
            if format not in (None, direct):
                raise ValueError(f"{name}: source representation is {direct}")
            info = store.describe(source)
            if info.shape != original or info.dtype != dtype:
                raise ValueError(
                    f"{source}: expected {dtype} {original}, got {info.dtype} {info.shape}"
                )
            ref = tensor_source(store, source, original)
            if rows is not None:
                ref = select_rows(ref, tuple(rows))
                return LogicalSource(shape, ref.label, ref.read_values)
            if axes is not None:
                # These transforms preserve each output row; read only requested rows.
                if axes[0] != 0 or shape[0] != original[0]:
                    raise ValueError("row transpose must preserve the leading axis")
                width = prod(original[1:])

                def read(begin, end):
                    if begin == end:
                        return ref.values(0, 0)
                    first, last = begin // width, (end + width - 1) // width
                    chunk = ref.values(first * width, last * width)
                    chunk = chunk.reshape(last - first, *original[1:]).permute(axes)
                    return chunk.contiguous().flatten()[
                        begin - first * width : end - first * width
                    ]

                return LogicalSource(shape, f"transpose({ref.label},{axes})", read)
            return tensor_source(
                store, source, shape, offset=offset, source_shape=original
            )

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

    def encoded(self, name, prefix, shape, *, inputs):
        def factory(store, format=None):
            if format not in (None, "nvfp4"):
                raise ValueError(f"{name}: source representation is nvfp4")
            return _nvfp4_source(store, prefix, shape)

        self.consumed.update(
            prefix + "." + field
            for field in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        )
        self.model.add(
            Parameter(
                name,
                shape,
                factory(self.base),
                source_factory=factory,
                inputs=tuple(inputs),
                direct_format="nvfp4",
                residency=name.split("/")[0],
            )
        )

    def hc(self, p, s, site):
        h, c = self.cfg["hidden_size"], self.cfg["hc_mult"]
        n = c * c + 2 * c
        self.add(p + "projection", s + f"hc_{site}_fn", (n, c * h))
        self.add(p + "base", s + f"hc_{site}_base", (n,))
        self.add(p + "scale", s + f"hc_{site}_scale", (3,))

    def kda(self, p, s):
        cfg = self.cfg
        h, heads, d, r = (
            cfg[k]
            for k in (
                "hidden_size",
                "linear_num_heads",
                "linear_head_dim",
                "linear_gate_lowrank",
            )
        )
        c, taps = heads * d, cfg["linear_conv_kernel_dim"]
        a = s + "self_attn."
        for role in ("query", "key", "value"):
            self.add(
                p + "kda/" + role,
                a + role[0] + "_proj.weight",
                (c, h),
                inputs=(p + "attention_input",),
            )
        self.group(*(p + "kda/" + role for role in ("query", "key", "value")))
        for role, field, dims, inp in (
            ("beta", "b_proj", (heads, h), "attention_input"),
            ("forget_a", "f_a_proj", (r, h), "attention_input"),
            ("forget_b", "f_b_proj", (c, r), "kda/forget_lowrank"),
            ("gate_a", "g_a_proj", (r, h), "attention_input"),
            ("gate_b", "g_b_proj", (c, r), "kda/gate_lowrank"),
            ("output", "o_proj", (h, c), "kda/gated_output"),
        ):
            self.add(p + "kda/" + role, a + field + ".weight", dims, inputs=(p + inp,))
        self.group(*(p + "kda/" + role for role in ("beta", "forget_a", "gate_a")))
        self.add(p + "kda/norm", a + "o_norm.weight", (d,))
        self.add(p + "kda/a_log", a + "A_log", (heads,), direct="fp32")
        self.add(p + "kda/dt_bias", a + "dt_bias", (c,), direct="fp32")
        names = tuple(a + role + "_conv1d.weight" for role in ("q", "k", "v"))
        self.consumed.update(names)

        def factory(store, format=None):
            if format not in (None, "fp32"):
                raise ValueError("KDA convolution preserves source FP32 words")
            for name in names:
                info = store.describe(name)
                if info.shape != (c, 1, taps) or info.dtype != "F32":
                    raise ValueError(f"{name}: expected F32 {(c, 1, taps)}")

            def read(begin, end):
                values = torch.cat(
                    [store.read_flat(name).reshape(c, taps).T for name in names], dim=1
                )
                return values.contiguous().flatten()[begin:end]

            return LogicalSource((taps, 3 * c), f"KDA convolution({a})", read)

        self.model.add(
            Parameter(
                p + "kda/convolution",
                (taps, 3 * c),
                factory(self.base),
                source_factory=factory,
                direct_format="fp32",
                residency=p.split("/")[0],
            )
        )

    def mla(self, p, s):
        cfg = self.cfg
        h, q, c, heads, dq, dv, ih, di = (
            cfg[k]
            for k in (
                "hidden_size",
                "q_lora_rank",
                "kv_lora_rank",
                "num_attention_heads",
                "qk_head_dim",
                "v_head_dim",
                "index_n_heads",
                "index_head_dim",
            )
        )
        a = s + "self_attn."
        for role, field, n in (
            ("latent", "kv_a_proj_with_mqa.weight", c),
            ("index_key", "indexer.wk.weight", di),
            ("pool_gate", "indexer.index_kpool_compress_gate", di),
            ("query_a", "q_a_proj.weight", q),
            ("index_weight", "indexer.weights_proj.weight", ih),
        ):
            self.add(
                p + "attention/" + role,
                a + field,
                (n, h),
                inputs=(p + "attention_input",),
            )
        self.group(
            *(
                p + "attention/" + role
                for role in (
                    "latent",
                    "index_key",
                    "pool_gate",
                    "query_a",
                    "index_weight",
                )
            )
        )
        for role, field, n in (
            ("query_b", "q_b_proj", heads * dq),
            ("index_query", "indexer.wq_b", ih * di),
        ):
            self.add(
                p + "attention/" + role,
                a + field + ".weight",
                (n, q),
                inputs=(p + "attention/query_lowrank",),
            )
        self.group(p + "attention/query_b", p + "attention/index_query")
        for role, start, count in (("key_up", 0, dq), ("value_up", dq, dv)):
            self.add(
                p + "attention/" + role,
                a + "kv_b_proj.weight",
                (heads, count, c),
                source_shape=(heads * (dq + dv), c),
                rows=tuple(
                    (j * (dq + dv) + start, j * (dq + dv) + start + count)
                    for j in range(heads)
                ),
            )
        self.add(
            p + "attention/output",
            a + "o_proj.weight",
            (h, heads * dv),
            inputs=(p + "attention/output_heads",),
        )
        for role, field, dims in (
            ("query_norm", "q_a_layernorm.weight", (q,)),
            ("latent_norm", "kv_a_layernorm.weight", (c,)),
            ("index_norm", "indexer.k_norm.weight", (di,)),
            ("index_bias", "indexer.k_norm.bias", (di,)),
            ("pool_ape", "indexer.index_kpool_compress_ape", (cfg["index_kpool"], di)),
        ):
            self.add(p + "attention/" + role, a + field, dims)

    def dense(self, p, s):
        h, i = self.cfg["hidden_size"], self.cfg["intermediate_size"]
        for role in ("gate", "up", "down"):
            self.encoded(
                p + "mlp/" + role,
                s + "mlp." + role + "_proj",
                (h, i) if role == "down" else (i, h),
                inputs=(p + ("mlp/product" if role == "down" else "mlp_input"),),
            )
        self.group(p + "mlp/gate", p + "mlp/up")

    def moe(self, p, s, *, encoded):
        cfg = self.cfg
        h, e, i = (
            cfg[k] for k in ("hidden_size", "n_routed_experts", "moe_intermediate_size")
        )
        shared = i * cfg["n_shared_experts"]
        self.add(
            p + "moe/router", s + "mlp.gate.weight", (e, h), inputs=(p + "moe_input",)
        )
        self.add(
            p + "moe/correction_bias",
            s + "mlp.gate.e_score_correction_bias",
            (e,),
            direct="fp32",
        )
        for role in ("gate", "up", "down"):
            inp = p + ("moe/shared/product" if role == "down" else "moe_input")
            self.add(
                p + "moe/shared/" + role,
                s + "mlp.shared_experts." + role + "_proj.weight",
                (h, shared) if role == "down" else (shared, h),
                inputs=(inp,),
            )
            matrix = (h, i) if role == "down" else (i, h)
            shape = (e, *matrix)
            prefixes = tuple(s + f"mlp.experts.{j}.{role}_proj" for j in range(e))
            for prefix in prefixes:
                self.consumed.update(
                    prefix + "." + field
                    for field in (
                        ("weight", "weight_scale", "weight_scale_2", "input_scale")
                        if encoded
                        else ("weight",)
                    )
                )

            def factory(
                store, format=None, *, prefixes=prefixes, matrix=matrix, shape=shape
            ):
                expected = "nvfp4" if encoded else "bf16"
                if format not in (None, expected):
                    raise ValueError(f"expert source representation is {expected}")
                refs = []
                for prefix in prefixes:
                    if encoded:
                        refs.append(_nvfp4_source(store, prefix, matrix))
                    else:
                        info = store.describe(prefix + ".weight")
                        if info.shape != matrix or info.dtype != "BF16":
                            raise ValueError(f"{prefix}: expected BF16 {matrix}")
                        refs.append(tensor_source(store, prefix + ".weight", matrix))
                return bank_source(refs, shape)

            inp = p + ("moe/experts/product" if role == "down" else "moe_input")
            self.model.add(
                Parameter(
                    p + "moe/experts/" + role,
                    shape,
                    factory(self.base),
                    source_factory=factory,
                    inputs=(inp,),
                    direct_format="nvfp4" if encoded else "bf16",
                    residency=p.split("/")[0],
                )
            )
        self.group(p + "moe/shared/gate", p + "moe/shared/up")

    def vision(self, cfg):
        h, i, out, merge = (
            cfg[k]
            for k in (
                "hidden_size",
                "intermediate_size",
                "out_hidden_size",
                "spatial_merge_size",
            )
        )
        patch_shape = (
            h,
            cfg["in_channels"],
            cfg["temporal_patch_size"],
            cfg["patch_size"],
            cfg["patch_size"],
        )
        self.add(
            "vision/patch_embedding",
            "model.visual.patch_embed.proj.weight",
            (h, prod(patch_shape[1:])),
            source_shape=patch_shape,
            inputs=("vision/patch_input",),
        )
        self.add(
            "vision/patch_embedding_bias", "model.visual.patch_embed.proj.bias", (h,)
        )
        self.add("vision/post_norm", "model.visual.post_layernorm.weight", (h,))
        for layer in range(cfg["depth"]):
            p, s = f"vision/layers/{layer}/", f"model.visual.blocks.{layer}."
            for role, field in (
                ("input_norm", "norm1"),
                ("post_attention_norm", "norm2"),
            ):
                self.add(p + role, s + field + ".weight", (h,))
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
            for role in ("query", "key"):
                self.add(
                    p + "attention/" + role + "_norm",
                    s + "attn." + role[0] + "_norm.weight",
                    (h // cfg["num_heads"],),
                )
            for role, field, dims, inp in (
                ("attention/output", "attn.proj", (h, h), "attention_output"),
                ("mlp/gate", "mlp.gate_proj", (i, h), "mlp_input"),
                ("mlp/up", "mlp.up_proj", (i, h), "mlp_input"),
                ("mlp/down", "mlp.down_proj", (h, i), "mlp/product"),
            ):
                self.add(p + role, s + field + ".weight", dims, inputs=(p + inp,))
                self.add(p + role + "_bias", s + field + ".bias", (dims[0],))
            self.group(p + "mlp/gate", p + "mlp/up")
            self.group(p + "mlp/gate_bias", p + "mlp/up_bias")
        self.add(
            "vision/downsample",
            "model.visual.downsample.weight",
            (out, h * merge * merge),
            source_shape=(out, h, merge, merge),
            axes=(0, 2, 3, 1),
            inputs=("vision/downsample_input",),
        )
        self.add("vision/downsample_bias", "model.visual.downsample.bias", (out,))
        p, s, mid = (
            "vision/merger/",
            "model.visual.merger.",
            cfg["projection_intermediate_size"],
        )
        self.add(p + "projection", s + "proj.weight", (out, out), inputs=(p + "input",))
        self.add(p + "norm_weight", s + "post_projection_norm.weight", (out,))
        self.add(p + "norm_bias", s + "post_projection_norm.bias", (out,))
        for role in ("gate", "up", "down"):
            self.add(
                p + role,
                s + role + "_proj.weight",
                (out, mid) if role == "down" else (mid, out),
                inputs=(p + ("product" if role == "down" else "activation"),),
            )
        self.group(p + "gate", p + "up")


def build_model(
    base, *, components=("text",), companions=None, resource_overrides=None
):
    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "mtp"}:
        raise ValueError("GLM5Next supports text, vision and mtp components")
    if companions:
        raise ValueError("GLM5Next components are read from its primary checkpoint")
    cfg = text_config(base.config)
    lowranks = set()
    for layer, kind in enumerate(cfg["layer_types"]):
        if kind == "linear_attention":
            for role in ("f", "g"):
                shape = base.describe(
                    f"model.language_model.layers.{layer}.self_attn.{role}_a_proj.weight"
                ).shape
                if len(shape) != 2 or shape[1] != cfg["hidden_size"]:
                    raise ValueError("invalid KDA lowrank projection geometry")
                lowranks.add(shape[0])
    if len(lowranks) != 1:
        raise ValueError("GLM5Next KDA lowrank widths must agree")
    cfg["linear_gate_lowrank"] = lowranks.pop()
    records = {"text": {"config": cfg}}
    roles = {"text": TEXT_RESOURCES}
    if "vision" in selected:
        v = deepcopy(base.config["vision_config"])
        v["model_type"] = "glm5next_vision"
        records["vision"] = {"config": v, "target": "text"}
        roles["vision"] = ("processor_config.json",)
    if "mtp" in selected:
        count = cfg["num_nextn_predict_layers"]
        if count != 1:
            raise ValueError("GLM5Next conversion requires one MTP prediction layer")
        records["mtp"] = {
            "config": {
                "architectures": ["Glm5NextMTP"],
                "model_type": "glm5next_mtp",
                "num_hidden_layers": count,
                "layer_types": ["deepseek_sparse_attention"] * count,
            },
            "target": "text",
        }
    refs, payloads, count, special = load_resources(
        base.root,
        vocab_size=cfg["vocab_size"],
        roles=roles,
        overrides=resource_overrides,
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
    b, h = Builder(model, base, cfg), cfg["hidden_size"]
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
    b.add("text/final_norm", "model.language_model.norm.weight", (h,))
    for layer, (attention, mlp) in enumerate(
        zip(cfg["layer_types"], cfg["mlp_layer_types"])
    ):
        p, s = f"text/layers/{layer}/", f"model.language_model.layers.{layer}."
        b.add(p + "input_norm", s + "input_layernorm.weight", (h,))
        b.add(p + "post_attention_norm", s + "post_attention_layernorm.weight", (h,))
        b.hc(p + "attention_hc/", s, "attn")
        b.hc(p + "mlp_hc/", s, "ffn")
        (b.kda if attention == "linear_attention" else b.mla)(p, s)
        if mlp == "dense":
            b.dense(p, s)
        else:
            b.moe(p, s, encoded=True)
    if "mtp" in selected:
        for role in ("token_embedding", "output_head"):
            value = model.parameters["text/" + role]
            model.add(
                Parameter(
                    "mtp/" + role,
                    value.shape,
                    value.source,
                    inputs=("mtp/final_hidden",) if role == "output_head" else (),
                    residency="mtp",
                )
            )
        s = f"model.language_model.layers.{cfg['num_hidden_layers']}."
        for role, field in (
            ("pre_fc_embedding_norm", "enorm"),
            ("pre_fc_hidden_norm", "hnorm"),
            ("final_norm", "shared_head.norm"),
        ):
            b.add("mtp/" + role, s + field + ".weight", (h,))
        b.add(
            "mtp/fc",
            s + "eh_proj.weight",
            (h, 2 * h),
            inputs=("mtp/normalized_input",),
        )
        p = "mtp/layers/0/"
        b.add(p + "input_norm", s + "input_layernorm.weight", (h,))
        b.add(p + "post_attention_norm", s + "post_attention_layernorm.weight", (h,))
        b.mla(p, s)
        b.moe(p, s, encoded=False)
    if "vision" in selected:
        b.vision(records["vision"]["config"])
    mtp_prefix = f"model.language_model.layers.{cfg['num_hidden_layers']}."
    expected = {
        name
        for name in base.weight_map
        if not (name.startswith("model.visual.") and "vision" not in selected)
        and not (name.startswith(mtp_prefix) and "mtp" not in selected)
    }
    if b.consumed != expected:
        raise ValueError(
            f"GLM5Next source coverage differs: missing {sorted(expected - b.consumed)}, extra {sorted(b.consumed - expected)}"
        )
    return model
