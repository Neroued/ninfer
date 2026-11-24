"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    AuxiliaryValue,
    cast_direct,
    fp8_row_maxabs,
    fp8_block_maxabs,
    grouped_absmax,
    import_encoded,
    stored_scale_mse,
)
from .recipe import AuxiliaryReference
from .sources.fp8_block import fp8_block_source
from .sources.modelopt_nvfp4 import ModelOptNvfp4
from .sources.safetensors import tensor_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_flash_next(model, recipe, sources):
    """Qwen4Exp: NVIDIA NVFP4/FP8 expert banks, Q8 projections and paged NF4 PLE."""
    recipe.provenance.update(
        mapping="qwen4_exp_v1",
        nvidia_nvfp4="preserve_codes_scales; RN_FP32(1/multiplier) divisors",
        nvidia_mtp="preserve_e4m3fn_codes_bf16_block128x128_scales",
        stored_scale_mse="g16_fp64_exhaustive_e4m3fn_v1",
        q_gate_order="all_query_heads_then_all_gate_heads",
        convolution_order="tap_channel",
    )
    cfg = model.config
    expected = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 2560,
        "vocab_size": 248320,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "ple_layer_ids": [1],
        "split_ngram_parts": 128,
        "ple_shard_shape": [2500012, 160],
        "ple_embed_dim": 2560,
        "layer_types": [
            "indexed_attention" if l % 4 == 3 else "linear_attention" for l in range(48)
        ],
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"qwen3_8_flash_next requires {key}={value!r}")
    recipe.place("text/*", (0,))
    for component in ("vision", "mtp"):
        if component in model.components:
            recipe.place(component + "/*", (1 if component == "vision" else 2, 0))
    for name in model.parameters:
        if name.startswith("text/layers/") and name.endswith(
            (
                "/gdn/query",
                "/gdn/key",
                "/gdn/value",
                "/gdn/z",
                "/gdn/output",
                "/attention/query",
                "/attention/output_gate",
                "/attention/output",
            )
        ):
            _assign(recipe, name, Q8)

    h, intermediate, experts = 2560, 640, 512
    width = h * intermediate
    nvidia = ModelOptNvfp4(sources["nvidia"])
    controls = []
    covered = set()
    for layer in range(48):
        p = f"text/layers/{layer}/moe/experts/"
        layer_inputs = None
        gate_up_parts = []
        down_parts = []
        for expert in range(experts):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            refs = {}
            scalars = {}
            for role in ("gate", "up", "down"):
                source = prefix + role + "_proj"
                refs[role] = nvidia.matrix(
                    source, (h, intermediate) if role == "down" else (intermediate, h)
                )
                scalars[role] = nvidia.scales(source)
                covered.update(
                    source + "." + field
                    for field in (
                        "weight",
                        "weight_scale",
                        "weight_scale_2",
                        "input_scale",
                    )
                )
            if scalars["gate"] != scalars["up"]:
                raise ValueError(f"{prefix}: gate/up multiplier words differ")
            input_words = (scalars["gate"][2], scalars["down"][2])
            if layer_inputs is None:
                layer_inputs = input_words
                controls.extend((scalars["gate"][3], scalars["down"][3]))
            elif layer_inputs != input_words:
                raise ValueError(
                    f"{prefix}: activation multipliers differ within the layer"
                )
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for part, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                part,
                shape=shape,
                format="nvfp4",
                method=import_encoded,
                layout="batched_block_scale_k16_m128x4_v1",
                order=(3, layer, order),
            )
    expected_experts = {
        key
        for key in nvidia.store.weight_map
        if key.startswith("model.language_model.layers.") and ".mlp.experts." in key
    }
    if covered != expected_experts:
        raise ValueError("NVIDIA source must contain the complete main expert tensors")
    recipe.auxiliary_tensor(
        "expert_activation_divisors",
        AuxiliaryValue("fp32", (48, 2), b"".join(controls)),
    )
    for layer in range(48):
        for role in ("gate", "up", "down"):
            name = f"text/layers/{layer}/moe/experts/{role}"
            recipe.use(
                name,
                model.parameters[name].inputs[0],
                activation_policy="AllowA4",
                auxiliaries={
                    "activation_input_divisor": AuxiliaryReference(
                        "expert_activation_divisors", 2 * layer + int(role == "down")
                    )
                },
            )

    if "mtp" in model.components:
        for role in ("token_embedding", "output_head"):
            recipe.share("mtp/" + role, "text/" + role)
        for name, parameter in model.parameters.items():
            if name.startswith("mtp/") and parameter.source_factory is not None:
                recipe.assign(name, source=model.source(name, nvidia.store))
        p = "mtp/layers/0/moe/experts/"
        gate_up_parts = []
        down_parts = []
        for expert in range(experts):
            refs = {}
            for role in ("gate", "up", "down"):
                refs[role] = fp8_block_source(
                    nvidia.store,
                    f"mtp.layers.0.mlp.experts.{expert}.{role}_proj",
                    (h, intermediate) if role == "down" else (intermediate, h),
                )
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for part, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                part,
                shape=shape,
                format="fp8_e4m3fn_block128x128_bf16",
                method=import_encoded,
                order=(2, 1, order),
            )
        for role in ("gate", "up", "down"):
            name = p + role
            recipe.use(
                name,
                model.parameters[name].inputs[0],
                activation_policy="AllowA8",
            )

    rows, k = cfg["ple_shard_shape"]
    for shard in range(cfg["split_ngram_parts"]):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        source = tensor_source(sources["base"], name, (rows, k))
        recipe.parent(
            [
                (
                    "text/layers/1/ple/table",
                    shard * rows * k,
                    (shard + 1) * rows * k,
                    source,
                )
            ],
            shape=(rows, k),
            format="nf4_g16_e4m3fn",
            method=stored_scale_mse,
            order=(4, shard),
        )


def glm5next(model, recipe, sources):
    """NVIDIA Main NVFP4, Q8 projections/shared FFN and block-FP8 MTP experts."""
    if model.config["architectures"] != ["Glm5NextForCausalLM"]:
        raise ValueError("glm5next requires GLM5Next mathematics")
    cfg = model.config
    h, intermediate, experts = (
        cfg[key] for key in ("hidden_size", "moe_intermediate_size", "n_routed_experts")
    )
    recipe.provenance.update(
        mapping="glm5next",
        nvidia_nvfp4="preserve_codes_scales; RN_FP32(1/multiplier) divisors",
        convolution_order="tap_then_qkv_channels",
        mla_prepare_order="latent_index_key_pool_gate_query_a_index_weight",
        mla_factor_order="head_channel_latent",
    )
    for name, parameter in model.parameters.items():
        if name.startswith("text/") and "/mlp/" in name and parameter.projection:
            recipe.assign(
                name, format="nvfp4", method=import_encoded, activation_policy="AllowA4"
            )
        elif name.startswith(("text/", "mtp/")) and (
            name.endswith(
                (
                    "/kda/query",
                    "/kda/key",
                    "/kda/value",
                    "/kda/output",
                    "/attention/output",
                )
            )
            or "/moe/shared/" in name
        ):
            _assign(recipe, name, Q8)
    recipe.place("text/*", (0,))
    for component in ("vision", "mtp"):
        if component in model.components:
            recipe.place(component + "/*", (1 if component == "vision" else 2, 0))
    nvidia = ModelOptNvfp4(sources["base"])
    for layer, kind in enumerate(cfg["mlp_layer_types"]):
        if kind != "dense":
            continue
        prefix = f"model.language_model.layers.{layer}.mlp."
        if nvidia.scales(prefix + "gate_proj") != nvidia.scales(prefix + "up_proj"):
            raise ValueError(f"layer {layer}: dense gate/up multipliers differ")
    controls = []
    layers = [
        layer for layer, kind in enumerate(cfg["mlp_layer_types"]) if kind == "sparse"
    ]
    width = h * intermediate
    for ordinal, layer in enumerate(layers):
        p = f"text/layers/{layer}/moe/experts/"
        gate_up_parts, down_parts = [], []
        layer_inputs = None
        for expert in range(experts):
            refs, scalars = {}, {}
            for role in ("gate", "up", "down"):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj"
                refs[role] = nvidia.matrix(
                    prefix, (h, intermediate) if role == "down" else (intermediate, h)
                )
                scalars[role] = nvidia.scales(prefix)
            if scalars["gate"] != scalars["up"]:
                raise ValueError(
                    f"layer {layer} expert {expert}: gate/up multipliers differ"
                )
            input_words = (scalars["gate"][2], scalars["down"][2])
            if layer_inputs is None:
                layer_inputs = input_words
                controls.extend((scalars["gate"][3], scalars["down"][3]))
            elif layer_inputs != input_words:
                raise ValueError(f"layer {layer}: expert activation multipliers differ")
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for parts, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                parts,
                shape=shape,
                format="nvfp4",
                method=import_encoded,
                layout="batched_block_scale_k16_m128x4_v1",
                order=(3, layer, order),
            )
    if layers:
        recipe.auxiliary_tensor(
            "expert_activation_divisors",
            AuxiliaryValue("fp32", (len(layers), 2), b"".join(controls)),
        )
        for ordinal, layer in enumerate(layers):
            for role in ("gate", "up", "down"):
                name = f"text/layers/{layer}/moe/experts/{role}"
                recipe.use(
                    name,
                    model.parameters[name].inputs[0],
                    activation_policy="AllowA4",
                    auxiliaries={
                        "activation_input_divisor": AuxiliaryReference(
                            "expert_activation_divisors",
                            2 * ordinal + int(role == "down"),
                        )
                    },
                )
    if "mtp" in model.components:
        for role in ("token_embedding", "output_head"):
            recipe.share("mtp/" + role, "text/" + role)
        p = "mtp/layers/0/moe/experts/"
        gate_up_parts, down_parts = [], []
        for expert in range(experts):
            refs = {}
            for role in ("gate", "up", "down"):
                refs[role] = tensor_source(
                    sources["base"],
                    f"model.language_model.layers.{cfg['num_hidden_layers']}.mlp.experts.{expert}.{role}_proj.weight",
                    (h, intermediate) if role == "down" else (intermediate, h),
                )
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for parts, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                parts,
                shape=shape,
                format="fp8_e4m3fn_block128x128_bf16",
                method=fp8_block_maxabs,
                order=(2, 1, order),
            )
        for role in ("gate", "up", "down"):
            name = p + role
            recipe.use(
                name, model.parameters[name].inputs[0], activation_policy="AllowA8"
            )


RECIPES = {
    "glm5next": glm5next,
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_8_flash_next": qwen3_8_flash_next,
}
