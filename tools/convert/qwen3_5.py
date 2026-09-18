"""Qwen3.5 architecture adapter: mathematical config, source axes, and packing groups.

Dense/MoE text, Vision, MTP, DFlash, and DFlash2 share this explicit mapping.
Official checkpoint names and quantization assignments live in official_recipes.
"""

from __future__ import annotations

from math import isfinite, isqrt
from pathlib import Path
import struct
from typing import Mapping

from .model import Model, Parameter
from .resources import load_resources
from .sources.logical import LogicalSource, select_rows, transpose_source
from .sources.safetensors import SafetensorsSource, tensor_source
from .sources.compressed_tensors import matrix_source

_TEXT_ARCHITECTURES = {
    "Qwen3_5ForCausalLM": False,
    "Qwen3_5ForConditionalGeneration": False,
    "Qwen3_5MoeForCausalLM": True,
    "Qwen3_5MoeForConditionalGeneration": True,
}


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def _f32(value, name):
    if type(value) not in (float, int):
        raise ValueError(f"{name}: expected a real value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name}: not representable as F32") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected positive finite F32")
    return result


def _fixed(config, key, value, label):
    if key in config and (
        config[key] != value
        or isinstance(value, bool)
        and type(config[key]) is not bool
    ):
        raise ValueError(f"{label}.{key}: expected {value!r}, got {config[key]!r}")


def _rope_source(raw: dict, label: str) -> dict:
    rope = {}
    for field in ("rope_scaling", "rope_parameters"):
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{label}.{field}: expected an object")
        for alias in ("type", "rope_type"):
            _fixed(value, alias, "default", label + "." + field)
        if "factor" in value and _f32(value["factor"], label + ".factor") != 1.0:
            raise ValueError(f"{label}: scaled RoPE is not implemented")
        for key, item in value.items():
            if key in rope and rope[key] != item:
                raise ValueError(f"{label}: conflicting RoPE field {key}")
            rope[key] = item
    return rope


def text_config(source: dict, *, mtp: bool) -> dict:
    architectures = source.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in _TEXT_ARCHITECTURES
    ):
        raise ValueError(f"unsupported Qwen text architecture {architectures!r}")
    moe = _TEXT_ARCHITECTURES[architectures[0]]
    raw = source.get("text_config", source)
    _fixed(raw, "hidden_act", "silu", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "attn_output_gate", True, "text")
    result = {
        "architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
    }
    for key in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "max_position_embeddings",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    tied = raw.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
    if type(tied) is not bool:
        raise ValueError("text.tie_word_embeddings must be boolean")
    result["tie_word_embeddings"] = tied
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    layers = raw.get("layer_types")
    if layers is None:
        interval = _positive(
            raw.get("full_attention_interval", 4), "full_attention_interval"
        )
        layers = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(result["num_hidden_layers"])
        ]
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(k not in ("full_attention", "linear_attention") for k in layers)
    ):
        raise ValueError("text.layer_types must describe every block")
    result["layer_types"] = list(layers)
    if "full_attention" in layers or mtp:
        for key in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            result[key] = _positive(raw.get(key), "text." + key)
        if result["num_attention_heads"] % result["num_key_value_heads"]:
            raise ValueError("text attention heads must be divisible by KV heads")
        rope = _rope_source(raw, "text")
        _fixed(rope, "mrope_interleaved", True, "text.rope_parameters")
        factor = _f32(
            rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 0.25)),
            "partial_rotary_factor",
        )
        theta = _f32(
            rope.get("rope_theta", raw.get("rope_theta", 10_000_000)), "rope_theta"
        )
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list)
            or len(sections) != 3
            or any(type(v) is not int or v < 0 for v in sections)
        ):
            raise ValueError("mrope_section must contain three nonnegative integers")
        rotary = int(result["head_dim"] * factor)
        if factor > 1 or rotary <= 0 or rotary % 2 or sum(sections) != rotary // 2:
            raise ValueError("MRoPE sections and rotary width disagree")
        axes = [
            (
                1
                if r % 3 == 1 and r < 3 * sections[1]
                else 2 if r % 3 == 2 and r < 3 * sections[2] else 0
            )
            for r in range(rotary // 2)
        ]
        if [axes.count(i) for i in range(3)] != sections:
            raise ValueError("MRoPE sections exceed the interleaved axis ranges")
        result["rope_parameters"] = {
            "rope_theta": theta,
            "partial_rotary_factor": factor,
            "mrope_section": list(sections),
        }
    if "linear_attention" in layers:
        _fixed(raw, "mamba_ssm_dtype", "float32", "text")
        for key in (
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ):
            result[key] = _positive(raw.get(key), "text." + key)
        if result["linear_num_value_heads"] % result["linear_num_key_heads"]:
            raise ValueError("linear value heads must be divisible by key heads")
    fields = (
        (
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        )
        if moe
        else ("intermediate_size",)
    )
    for key in fields:
        result[key] = _positive(raw.get(key), "text." + key)
    if moe and result["num_experts_per_tok"] > result["num_experts"]:
        raise ValueError("selected experts exceed expert count")
    if mtp:
        _fixed(raw, "mtp_num_hidden_layers", 1, "text")
        _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
    return result


def vision_config(source: dict, text: dict) -> dict:
    raw = source.get("vision_config")
    if not isinstance(raw, dict):
        raise ValueError("Vision requires vision_config")
    _fixed(raw, "in_channels", 3, "vision")
    _fixed(raw, "hidden_act", "gelu_pytorch_tanh", "vision")
    _fixed(raw, "deepstack_visual_indexes", [], "vision")
    _fixed(raw, "out_hidden_size", text["hidden_size"], "vision")
    result = {
        "model_type": (
            "qwen3_5_moe_vision" if "num_experts" in text else "qwen3_5_vision"
        )
    }
    for key in (
        "depth",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "num_position_embeddings",
    ):
        result[key] = _positive(raw.get(key), "vision." + key)
    h, heads = result["hidden_size"], result["num_heads"]
    positions = result["num_position_embeddings"]
    if h % heads or (h // heads) % 4 or isqrt(positions) ** 2 != positions:
        raise ValueError("Vision head or square position geometry is invalid")
    return result


def draft_config(raw: dict, target: dict, backend: str) -> dict:
    architecture = "DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"
    _fixed(raw, "architectures", [architecture], backend)
    _fixed(raw, "model_type", "qwen3", backend)
    _fixed(raw, "hidden_act", "silu", backend)
    _fixed(raw, "attention_bias", False, backend)
    _fixed(raw, "is_causal", False, backend)
    for key, value in (
        ("hidden_size", target["hidden_size"]),
        ("vocab_size", target["vocab_size"]),
        ("num_target_layers", target["num_hidden_layers"]),
    ):
        _fixed(raw, key, value, backend)
    result = {"architectures": [architecture], "model_type": "qwen3"}
    for key in (
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_hidden_layers",
        "max_position_embeddings",
    ):
        result[key] = _positive(raw.get(key), backend + "." + key)
    if (
        result["num_attention_heads"] % result["num_key_value_heads"]
        or result["head_dim"] % 2
    ):
        raise ValueError(f"{backend}: invalid attention geometry")
    result["rms_norm_eps"] = _f32(
        raw.get("rms_norm_eps", 1e-6), backend + ".rms_norm_eps"
    )
    rope = _rope_source(raw, backend)
    result["rope_parameters"] = {
        "rope_theta": _f32(
            rope.get("rope_theta", raw.get("rope_theta", 10_000_000)),
            backend + ".rope_theta",
        )
    }
    layers = raw.get("layer_types")
    if layers is None:
        enabled = raw.get("use_sliding_window", False)
        if type(enabled) is not bool:
            raise ValueError(f"{backend}.use_sliding_window must be boolean")
        # Qwen3Config applies sliding attention at and above this layer index.
        threshold = raw.get("max_window_layers", 28)
        if type(threshold) is not int or threshold < 0:
            raise ValueError(f"{backend}.max_window_layers must be nonnegative")
        layers = [
            "sliding_attention" if enabled and i >= threshold else "full_attention"
            for i in range(result["num_hidden_layers"])
        ]
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(k not in ("sliding_attention", "full_attention") for k in layers)
    ):
        raise ValueError(f"{backend}: invalid layer_types")
    result["layer_types"] = list(layers)
    if "sliding_attention" in layers:
        result["sliding_window"] = _positive(
            raw.get("sliding_window"), backend + ".sliding_window"
        )
    draft = raw.get("dflash_config", {})
    taps = draft.get("target_layer_ids")
    if (
        not isinstance(taps, list)
        or not taps
        or any(
            type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in taps
        )
        or len(set(taps)) != len(taps)
    ):
        raise ValueError(f"{backend}: invalid target_layer_ids")
    mask = draft.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < target["vocab_size"]:
        raise ValueError(f"{backend}: mask token outside target embedding")
    output = {"target_layer_ids": list(taps), "mask_token_id": mask}
    if backend == "dflash2":
        for key in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            output[key] = _positive(draft.get(key), backend + "." + key)
        if target["hidden_size"] % output["conv_group_size"]:
            raise ValueError(
                "DFlash2 hidden width must be divisible by conv_group_size"
            )
        for scope in (raw, draft):
            for key, expected in (
                ("sample_from_anchor", False),
                ("input_embedding_scale", 1.0),
                ("output_multiplier", 1.0),
            ):
                _fixed(scope, key, expected, backend)
            if scope.get("final_logit_softcapping") not in (None, 0, 0.0):
                raise ValueError("DFlash2 final_logit_softcapping must be disabled")
    result["dflash_config"] = output
    return result


def _has_model_config(source: SafetensorsSource) -> bool:
    return bool(
        source.config.keys()
        - {
            "quantization_config",
            "_name_or_path",
            "transformers_version",
            "torch_dtype",
            "dtype",
        }
    )


def _check_source_fields(name, actual, expected, fields):
    for field in sorted(fields):
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"{name}: additional source {field} differs; provide an explicit logical source"
            )


class _Builder:
    def __init__(self, model: Model):
        self.model = model

    def validate_source(self, selected, original, name):
        if selected is original or not _has_model_config(selected):
            return
        component = name.split("/", 1)[0]
        target = self.model.config
        if component in ("text", "mtp"):
            actual = text_config(selected.config, mtp=component == "mtp")
            expected = target
            fields = {"hidden_size"}
            if name.endswith(("token_embedding", "output_head")):
                fields.add("vocab_size")
            if name == "text/output_head":
                fields.add("tie_word_embeddings")
            if "/attention/" in name:
                fields.update(
                    ("num_attention_heads", "num_key_value_heads", "head_dim")
                )
            if "/gdn/" in name:
                fields.update(
                    (
                        "linear_num_key_heads",
                        "linear_key_head_dim",
                        "linear_num_value_heads",
                        "linear_value_head_dim",
                    )
                )
                if name.endswith("/convolution"):
                    fields.add("linear_conv_kernel_dim")
            if "/mlp/" in name:
                fields.add("intermediate_size")
            if "/moe/experts/" in name:
                fields.update(("num_experts", "moe_intermediate_size"))
            if "/moe/shared/" in name:
                fields.add("shared_expert_intermediate_size")
            if name.endswith("/moe/router"):
                fields.add("num_experts")
        elif component == "vision":
            actual = vision_config(selected.config, target)
            expected = self.model.components[component]["config"]
            fields = {"hidden_size"}
            if "/attention/" in name:
                fields.add("num_heads")
            if "/mlp/" in name:
                fields.add("intermediate_size")
            if name == "vision/patch_embedding":
                fields.update(("patch_size", "temporal_patch_size"))
            if name.startswith("vision/merger/fc"):
                fields.add("spatial_merge_size")
            if "position_embedding" in name:
                fields.add("num_position_embeddings")
        else:
            actual = draft_config(selected.config, target, component)
            expected = self.model.components[component]["config"]
            fields = set()
            if "/attention/" in name:
                fields.update(
                    ("num_attention_heads", "num_key_value_heads", "head_dim")
                )
            if "/mlp/" in name:
                fields.add("intermediate_size")
            draft_fields = set()
            if name.endswith("/feature_projection"):
                draft_fields.add("target_layer_ids")
            if "conv/" in name:
                draft_fields.add("conv_kernel_size")
                if name.endswith("/kernel_projection"):
                    draft_fields.add("conv_group_size")
            if "/candidate_selector/" in name:
                draft_fields.add("selector_rank")
            _check_source_fields(
                name, actual["dflash_config"], expected["dflash_config"], draft_fields
            )
        if "/layers/" in name:
            index = int(name.split("/layers/", 1)[1].split("/", 1)[0])
            if component == "vision":
                compatible = index < actual["depth"]
            elif component == "mtp":
                compatible = index == 0
            else:
                compatible = index < len(actual["layer_types"]) and (
                    actual["layer_types"][index] == expected["layer_types"][index]
                )
            if not compatible:
                raise ValueError(
                    f"{name}: additional source layer topology differs; provide an explicit logical source"
                )
        _check_source_fields(name, actual, expected, fields)

    def add(
        self,
        name,
        store,
        source_name,
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
        original = shape if source_shape is None else tuple(source_shape)

        def factory(selected, format=None):
            self.validate_source(selected, store, name)
            selected_name = source_name
            if name.startswith("text/") and _has_model_config(selected):
                if "text_config" not in selected.config:
                    selected_name = selected_name.replace(
                        "model.language_model.", "model.", 1
                    )
                elif "text_config" not in store.config and selected_name.startswith(
                    "model."
                ):
                    selected_name = selected_name.replace(
                        "model.", "model.language_model.", 1
                    )
            if transpose is not None:
                if format is not None:
                    raise ValueError(f"{name}: transpose source requires value access")
                ref = tensor_source(selected, selected_name, original)
                return transpose_source(ref, transpose, shape)
            if len(original) == 2 and offset == 0:
                ref = matrix_source(selected, selected_name, original, format)
                return select_rows(ref, rows) if rows is not None else ref
            if format is not None:
                raise ValueError(f"{name}: provide an explicit encoded source mapping")
            return tensor_source(
                selected, selected_name, shape, offset=offset, source_shape=original
            )

        self.model.add(
            Parameter(
                name,
                shape,
                factory(store),
                factory,
                tuple(inputs),
                direct,
                residency=name.split("/", 1)[0],
            )
        )

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def attention(self, prefix, source_prefix, store, config):
        h, d = config["hidden_size"], config["head_dim"]
        q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
        for role, gate in (("query", False), ("gate", True)):
            ranges = tuple(
                (head * 2 * d + int(gate) * d, head * 2 * d + (int(gate) + 1) * d)
                for head in range(config["num_attention_heads"])
            )
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn.q_proj.weight",
                (q, h),
                source_shape=(2 * q, h),
                rows=ranges,
                inputs=(prefix + "mixer_input",),
            )
        for role, source in (("key", "k_proj"), ("value", "v_proj")):
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn." + source + ".weight",
                (k, h),
                inputs=(prefix + "mixer_input",),
            )
        for role, source in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn." + source + ".weight",
                (d,),
            )
        self.add(
            prefix + "attention/output",
            store,
            source_prefix + "self_attn.o_proj.weight",
            (h, q),
            inputs=(prefix + "attention/gated_output",),
        )
        roles = [
            prefix + "attention/" + role for role in ("query", "key", "gate", "value")
        ]
        self.group(*roles)
        self.group(*roles[:2])
        self.group(*roles[2:])

    def gdn(self, prefix, source_prefix, store, config):
        h = config["hidden_size"]
        kg = config["linear_num_key_heads"] * config["linear_key_head_dim"]
        vg = config["linear_num_value_heads"] * config["linear_value_head_dim"]
        nv, dv, taps = (
            config["linear_num_value_heads"],
            config["linear_value_head_dim"],
            config["linear_conv_kernel_dim"],
        )
        channels = 2 * kg + vg
        source_prefix += "linear_attn."
        for role, field in (("a_log", "A_log"), ("dt_bias", "dt_bias")):
            self.add(
                prefix + "gdn/" + role,
                store,
                source_prefix + field,
                (nv,),
                direct="fp32",
            )
        self.add(
            prefix + "gdn/convolution",
            store,
            source_prefix + "conv1d.weight",
            (taps, channels),
            source_shape=(channels, 1, taps),
            transpose=(2, 0, 1),
        )
        for role, field in (
            ("a_projection", "in_proj_a"),
            ("b_projection", "in_proj_b"),
        ):
            self.add(
                prefix + "gdn/" + role,
                store,
                source_prefix + field + ".weight",
                (nv, h),
                inputs=(prefix + "mixer_input",),
            )
        for role, start, count in (
            ("query", 0, kg),
            ("key", kg, kg),
            ("value", 2 * kg, vg),
        ):
            self.add(
                prefix + "gdn/" + role,
                store,
                source_prefix + "in_proj_qkv.weight",
                (count, h),
                source_shape=(channels, h),
                rows=((start, start + count),),
                inputs=(prefix + "mixer_input",),
            )
        self.add(
            prefix + "gdn/z",
            store,
            source_prefix + "in_proj_z.weight",
            (vg, h),
            inputs=(prefix + "mixer_input",),
        )
        self.add(prefix + "gdn/norm", store, source_prefix + "norm.weight", (dv,))
        self.add(
            prefix + "gdn/output",
            store,
            source_prefix + "out_proj.weight",
            (h, vg),
            inputs=(prefix + "gdn/gated_output",),
        )
        roles = [prefix + "gdn/" + role for role in ("query", "key", "value", "z")]
        self.group(*roles)
        self.group(*roles[:2])
        self.group(*roles[2:])
        self.group(prefix + "gdn/a_projection", prefix + "gdn/b_projection")

    def dense(self, prefix, source_prefix, store, h, intermediate, *, draft=False):
        stem_input = prefix + ("mlp_input" if draft else "ffn_input")
        for role in ("gate", "up"):
            self.add(
                prefix + "mlp/" + role,
                store,
                source_prefix + "mlp." + role + "_proj.weight",
                (intermediate, h),
                inputs=(stem_input,),
            )
        self.add(
            prefix + "mlp/down",
            store,
            source_prefix + "mlp.down_proj.weight",
            (h, intermediate),
            inputs=(prefix + ("mlp_product" if draft else "mlp/product"),),
        )
        self.group(prefix + "mlp/gate", prefix + "mlp/up")

    def moe(self, prefix, source_prefix, store, config):
        h, e = config["hidden_size"], config["num_experts"]
        ir, shared = (
            config["moe_intermediate_size"],
            config["shared_expert_intermediate_size"],
        )
        sp, p = source_prefix + "mlp.", prefix + "moe/"
        self.add(
            p + "router",
            store,
            sp + "gate.weight",
            (e, h),
            inputs=(prefix + "ffn_input",),
        )
        self.add(
            p + "shared_score",
            store,
            sp + "shared_expert_gate.weight",
            (1, h),
            inputs=(prefix + "ffn_input",),
        )
        self.group(p + "router", p + "shared_score")
        gate_up, downs = [], []
        for expert in range(e):
            ep = p + f"experts/{expert}/"
            for role, half in (("gate", 0), ("up", 1)):
                self.add(
                    ep + role,
                    store,
                    sp + "experts.gate_up_proj",
                    (ir, h),
                    source_shape=(e, 2 * ir, h),
                    offset=(expert * 2 + half) * ir * h,
                    inputs=(prefix + "ffn_input",),
                )
                gate_up.append(ep + role)
            self.add(
                ep + "down",
                store,
                sp + "experts.down_proj",
                (h, ir),
                source_shape=(e, h, ir),
                offset=expert * h * ir,
                inputs=(ep + "product",),
            )
            downs.append(ep + "down")
        self.group(*gate_up)
        self.group(*downs)
        for role in ("gate", "up"):
            self.add(
                p + "shared/" + role,
                store,
                sp + "shared_expert." + role + "_proj.weight",
                (shared, h),
                inputs=(prefix + "ffn_input",),
            )
        self.add(
            p + "shared/down",
            store,
            sp + "shared_expert.down_proj.weight",
            (h, shared),
            inputs=(p + "shared/product",),
        )
        self.group(p + "shared/gate", p + "shared/up")

    def block(self, prefix, source_prefix, store, config, mixer):
        h = config["hidden_size"]
        self.add(
            prefix + "input_norm", store, source_prefix + "input_layernorm.weight", (h,)
        )
        if mixer == "full_attention":
            self.attention(prefix, source_prefix, store, config)
        else:
            self.gdn(prefix, source_prefix, store, config)
        self.add(
            prefix + "post_attention_norm",
            store,
            source_prefix + "post_attention_layernorm.weight",
            (h,),
        )
        if "num_experts" in config:
            self.moe(prefix, source_prefix, store, config)
        else:
            self.dense(prefix, source_prefix, store, h, config["intermediate_size"])

    def vision(self, store, config, output_width):
        h, intermediate = config["hidden_size"], config["intermediate_size"]
        ps, pt = config["patch_size"], config["temporal_patch_size"]
        self.add(
            "vision/patch_embedding",
            store,
            "model.visual.patch_embed.proj.weight",
            (h, 3 * pt * ps * ps),
            source_shape=(h, 3, pt, ps, ps),
            inputs=("vision/patch_input",),
        )
        self.add(
            "vision/patch_embedding_bias",
            store,
            "model.visual.patch_embed.proj.bias",
            (h,),
        )
        self.add(
            "vision/position_embedding",
            store,
            "model.visual.pos_embed.weight",
            (config["num_position_embeddings"], h),
        )
        for i in range(config["depth"]):
            p, sp = f"vision/layers/{i}/", f"model.visual.blocks.{i}."
            for role in ("norm1", "norm2"):
                for kind in ("weight", "bias"):
                    self.add(p + role + "_" + kind, store, sp + role + "." + kind, (h,))
            for number, role in enumerate(("query", "key", "value")):
                self.add(
                    p + "attention/" + role,
                    store,
                    sp + "attn.qkv.weight",
                    (h, h),
                    source_shape=(3 * h, h),
                    rows=((number * h, (number + 1) * h),),
                    inputs=(p + "attention_input",),
                )
                self.add(
                    p + "attention/" + role + "_bias",
                    store,
                    sp + "attn.qkv.bias",
                    (h,),
                    source_shape=(3 * h,),
                    offset=number * h,
                )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            self.group(
                *(p + "attention/" + r + "_bias" for r in ("query", "key", "value"))
            )
            for role, field, dims, inputs in (
                (
                    "attention/output",
                    "attn.proj.weight",
                    (h, h),
                    (p + "attention_output",),
                ),
                ("attention/output_bias", "attn.proj.bias", (h,), ()),
                (
                    "mlp/fc1",
                    "mlp.linear_fc1.weight",
                    (intermediate, h),
                    (p + "mlp_input",),
                ),
                ("mlp/fc1_bias", "mlp.linear_fc1.bias", (intermediate,), ()),
                (
                    "mlp/fc2",
                    "mlp.linear_fc2.weight",
                    (h, intermediate),
                    (p + "mlp_activation",),
                ),
                ("mlp/fc2_bias", "mlp.linear_fc2.bias", (h,), ()),
            ):
                self.add(p + role, store, sp + field, dims, inputs=inputs)
        merger = config["spatial_merge_size"] ** 2 * h
        for role, field, dims, inputs in (
            ("norm_weight", "norm.weight", (h,), ()),
            ("norm_bias", "norm.bias", (h,), ()),
            ("fc1", "linear_fc1.weight", (merger, merger), ("vision/merger/input",)),
            ("fc1_bias", "linear_fc1.bias", (merger,), ()),
            (
                "fc2",
                "linear_fc2.weight",
                (output_width, merger),
                ("vision/merger/activation",),
            ),
            ("fc2_bias", "linear_fc2.bias", (output_width,), ()),
        ):
            self.add(
                "vision/merger/" + role,
                store,
                "model.visual.merger." + field,
                dims,
                inputs=inputs,
            )

    def draft(self, store, config, target, backend):
        h, d = target["hidden_size"], config["head_dim"]
        q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
        draft = config["dflash_config"]
        self.add(
            backend + "/feature_projection",
            store,
            "fc.weight",
            (h, len(draft["target_layer_ids"]) * h),
            inputs=(backend + "/target_features",),
        )
        self.add(backend + "/context_norm", store, "hidden_norm.weight", (h,))
        self.add(backend + "/final_norm", store, "norm.weight", (h,))
        for i in range(config["num_hidden_layers"]):
            p, sp = f"{backend}/layers/{i}/", f"layers.{i}."
            self.add(p + "input_norm", store, sp + "input_layernorm.weight", (h,))
            self.add(
                p + "post_attention_norm",
                store,
                sp + "post_attention_layernorm.weight",
                (h,),
            )
            for role, field, rows in (
                ("query", "q_proj", q),
                ("key", "k_proj", k),
                ("value", "v_proj", k),
            ):
                self.add(
                    p + "attention/" + role,
                    store,
                    sp + "self_attn." + field + ".weight",
                    (rows, h),
                    inputs=(p + "query_projection_input",),
                )
                if role in ("key", "value"):
                    self.add(
                        p + "attention/context_" + role,
                        store,
                        sp + "self_attn." + field + ".weight",
                        (rows, h),
                        inputs=(backend + "/context_input",),
                    )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
                self.add(
                    p + "attention/" + role,
                    store,
                    sp + "self_attn." + field + ".weight",
                    (d,),
                )
            self.add(
                p + "attention/output",
                store,
                sp + "self_attn.o_proj.weight",
                (h, q),
                inputs=(p + "attention_output",),
            )
            self.dense(p, sp, store, h, config["intermediate_size"], draft=True)
            if backend == "dflash2":
                taps, groups = draft["conv_kernel_size"], h // draft["conv_group_size"]
                for branch in ("attention_conv", "mlp_conv"):
                    self.add(
                        p + branch + "/base_kernel",
                        store,
                        sp + branch + ".base_kernel",
                        (2, taps, h),
                    )
                    self.add(
                        p + branch + "/kernel_projection",
                        store,
                        sp + branch + ".kernel_projection.weight",
                        (2 * taps * groups, h),
                        inputs=(p + branch + "_input",),
                    )
        if backend == "dflash2":
            rank = draft["selector_rank"]
            self.add(
                backend + "/candidate_selector/hidden_projection",
                store,
                "candidate_selector.hidden_projection.weight",
                (rank, h),
                inputs=(backend + "/final_hidden",),
            )
            for role in ("predecessor_codebook", "successor_codebook"):
                self.add(
                    backend + "/candidate_selector/" + role,
                    store,
                    "candidate_selector." + role,
                    (target["vocab_size"], rank),
                )


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {
        "text",
        "vision",
        "mtp",
        "dflash",
        "dflash2",
    }:
        raise ValueError("select text and supported optional components")
    companions = {} if companions is None else companions
    config = text_config(base.config, mtp="mtp" in selected)
    records = {"text": {"config": config}}
    if "vision" in selected:
        records["vision"] = {
            "config": vision_config(base.config, config),
            "target": "text",
        }
    if "mtp" in selected:
        architecture = "Qwen3_5MoeMTP" if "num_experts" in config else "Qwen3_5MTP"
        records["mtp"] = {"config": {"architectures": [architecture]}, "target": "text"}
    for backend in ("dflash", "dflash2"):
        if backend in selected:
            if backend not in companions:
                raise ValueError(f"selected {backend} requires its source")
            records[backend] = {
                "config": draft_config(companions[backend].config, config, backend),
                "target": "text",
            }
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=records["vision"]["config"] if "vision" in selected else None,
        overrides=resource_overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(
        records, resources=resources, token_count=count, special_token_ids=special
    )
    builder = _Builder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    builder.add(
        "text/token_embedding", base, text_prefix + "embed_tokens.weight", (r, h)
    )
    head_source = (
        text_prefix + "embed_tokens.weight"
        if config["tie_word_embeddings"]
        else "lm_head.weight"
    )
    head_inputs = ("text/final_hidden",) + tuple(
        c + "/final_hidden" for c in ("mtp", "dflash", "dflash2") if c in selected
    )
    builder.add("text/output_head", base, head_source, (r, h), inputs=head_inputs)
    builder.add("text/final_norm", base, text_prefix + "norm.weight", (h,))
    for i, kind in enumerate(config["layer_types"]):
        builder.block(
            f"text/layers/{i}/", text_prefix + f"layers.{i}.", base, config, kind
        )
    if "mtp" in selected:
        builder.add(
            "mtp/input_projection",
            base,
            "mtp.fc.weight",
            (h, 2 * h),
            inputs=("mtp/stem_input",),
        )
        for role, field in (
            ("embedding_norm", "pre_fc_norm_embedding"),
            ("hidden_norm", "pre_fc_norm_hidden"),
            ("final_norm", "norm"),
        ):
            builder.add("mtp/" + role, base, "mtp." + field + ".weight", (h,))
        builder.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention")
    if "vision" in selected:
        builder.vision(base, records["vision"]["config"], h)
    for backend in ("dflash", "dflash2"):
        if backend in selected:
            if (
                backend == "dflash2"
                and records[backend]["config"]["dflash_config"]["selector_top_k"]
                > count
            ):
                raise ValueError("DFlash2 selector_top_k exceeds public token domain")
            builder.draft(
                companions[backend], records[backend]["config"], config, backend
            )
    return model
