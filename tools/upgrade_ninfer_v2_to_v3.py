#!/usr/bin/env python3
"""One-time, standard-library upgrade of the seven known official NInfer v2 inputs.

Run: python3 upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer
Weight bytes are preserved and the maintained Qwen chat template is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import uuid

if hasattr(os, "posix_fadvise"):
    _fdatasync = os.fdatasync

    def _dontneed(fd, offset=0, length=0):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
else:
    # Windows: no fdatasync/fadvise; fsync plus a periodic FlushFileBuffers via
    # os.sync-free fsync is the closest available durability primitive.
    def _fdatasync(fd):
        os.fsync(fd)

    def _dontneed(fd, offset=0, length=0):
        del fd, offset, length  # cache hint unavailable


FORMATS = {
    "BF16": "bf16",
    "FP32": "fp32",
    "I32": "int32",
    "Q4G64_F16S": "q4_g64_fp16",
    "Q5G64_F16S": "q5_g64_fp16",
    "Q6G64_F16S": "q6_g64_fp16",
    "W8G32_F16S": "q8_g32_fp16",
    "NVFP4": "nvfp4",
    "FP8_E4M3FN_ROW_BF16S": "fp8_e4m3fn_row_bf16",
}
LAYOUTS = {
    "contiguous-le-v1": "contiguous_le_v1",
    "row-split-k128-v1": "row_split_k128_v1",
    "blockscale-k16-m128x4-v1": "block_scale_k16_m128x4_v1",
    "row-scale-v1": "row_scale_v1",
}
KNOWN_COUNTS = {
    ("qwen3.6-27b", "groupwise-int"): (1124,),
    ("qwen3.6-27b", "nvfp4"): (1307,),
    ("qwen3.8-27b", "groupwise-int"): (1124, 1190),
    ("qwen3.8-27b", "nvfp4"): (1124, 1190),
    ("qwen3.6-35b-a3b", "groupwise-int"): (940,),
}
LIMIT = 32_000_000_000
HEADER = struct.Struct("<8sQ16s")
CHUNK = 8 * 1024 * 1024
WRITEBACK = 64 * 1024 * 1024


def align(value, amount=4096):
    return (value + amount - 1) // amount * amount


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def text_config(moe):
    layers = 40 if moe else 64
    result = {
        "architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "hidden_size": 2048 if moe else 5120,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
        "num_hidden_layers": layers,
        "layer_types": [
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(layers)
        ],
        "max_position_embeddings": 262144,
        "rms_norm_eps": f32(1e-6),
        "num_attention_heads": 16 if moe else 24,
        "num_key_value_heads": 2 if moe else 4,
        "head_dim": 256,
        "rope_parameters": {
            "rope_theta": f32(10000000),
            "partial_rotary_factor": f32(0.25),
            "mrope_section": [11, 11, 10],
        },
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 32 if moe else 48,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if moe:
        result.update(
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        )
    else:
        result["intermediate_size"] = 17408
    return result


def vision_config(moe):
    return {
        "model_type": "qwen3_5_moe_vision" if moe else "qwen3_5_vision",
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
    }


def draft_config(second):
    draft = {
        "target_layer_ids": (
            [5, 19, 33, 47, 61] if second else [1, 6, 11, 16, 22, 27, 32, 37]
        ),
        "mask_token_id": 248070 if second else 248077,
    }
    if second:
        draft.update(
            conv_kernel_size=2, conv_group_size=16, selector_rank=256, selector_top_k=16
        )
    return {
        "architectures": ["DFlash2DraftModel" if second else "DFlashDraftModel"],
        "model_type": "qwen3",
        "intermediate_size": 17408 if second else 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 5 if second else 6,
        "rms_norm_eps": f32(1e-6),
        "rope_parameters": {"rope_theta": f32(10000000)},
        "max_position_embeddings": 262144,
        "layer_types": ["sliding_attention"] * 5
        + ([] if second else ["full_attention"]),
        "sliding_window": 2048 if second else 4096,
        "dflash_config": draft,
    }


def encoded_bytes(obj):
    dims, format, layout = obj["shape"], obj["format"], obj["layout"]
    if any(type(d) is not int or d <= 0 for d in dims):
        raise ValueError(f"{obj['id']}: invalid shape")
    if layout == "contiguous_le_v1" and format in ("bf16", "fp32", "int32"):
        return math.prod(dims) * (2 if format == "bf16" else 4)
    if len(dims) != 2:
        raise ValueError(f"{obj['id']}: expected matrix encoding")
    n, k = dims
    if layout == "row_split_k128_v1" and format in (
        "q4_g64_fp16",
        "q5_g64_fp16",
        "q6_g64_fp16",
        "q8_g32_fp16",
    ):
        bits = int(format[1])
        group = 32 if bits == 8 else 64
        groups = align(k, 128) // group
        low = n * groups * (group if bits == 8 else group // 2)
        high = n * groups * (0 if bits in (4, 8) else group * (bits - 4) // 8)
        return align(low, 256) + align(high, 256) + n * groups * 2
    if (
        layout == "block_scale_k16_m128x4_v1"
        and format == "nvfp4"
        and n % 128 == 0
        and k % 64 == 0
    ):
        return align(n * k // 2, 256) + n * k // 16 + 4
    if layout == "row_scale_v1" and format == "fp8_e4m3fn_row_bf16":
        return align(n * k, 256) + n * 2
    raise ValueError(f"{obj['id']}: unknown format/layout combination")


def _input_names(name, components):
    backends = [c for c in ("mtp", "dflash", "dflash2") if c in components]
    if name == "text/output_head":
        return ["text/final_hidden"] + [c + "/final_hidden" for c in backends]
    if name == "proposal/head":
        return [c + "/final_hidden" for c in backends]
    if name == "mtp/input_projection":
        return ["mtp/stem_input"]
    match = re.fullmatch(r"(text/layers/\d+/|mtp/layers/0/)(.+)", name)
    if match:
        block, role = match.groups()
        if role in (
            "attention/query",
            "attention/key",
            "attention/gate",
            "attention/value",
            "gdn/query",
            "gdn/key",
            "gdn/value",
            "gdn/z",
            "gdn/a_projection",
            "gdn/b_projection",
        ):
            return [block + "mixer_input"]
        if role in ("attention/output", "gdn/output"):
            return [block + role.rsplit("/", 1)[0] + "/gated_output"]
        if role in (
            "mlp/gate",
            "mlp/up",
            "moe/router",
            "moe/shared_score",
        ) or re.fullmatch(r"moe/(experts/\d+|shared)/(gate|up)", role):
            return [block + "ffn_input"]
        if role == "mlp/down":
            return [block + "mlp/product"]
        if re.fullmatch(r"moe/(experts/\d+|shared)/down", role):
            return [block + role.rsplit("/", 1)[0] + "/product"]
    if name == "vision/patch_embedding":
        return ["vision/patch_input"]
    if name in ("vision/merger/fc1", "vision/merger/fc2"):
        return [
            (
                "vision/merger/input"
                if name.endswith("fc1")
                else "vision/merger/activation"
            )
        ]
    match = re.fullmatch(r"(vision/layers/\d+/)(.+)", name)
    if match:
        block, role = match.groups()
        position = {
            "attention/query": "attention_input",
            "attention/key": "attention_input",
            "attention/value": "attention_input",
            "attention/output": "attention_output",
            "mlp/fc1": "mlp_input",
            "mlp/fc2": "mlp_activation",
        }.get(role)
        return [] if position is None else [block + position]
    match = re.fullmatch(r"(dflash2?)/(.*)", name)
    if match:
        backend, role = match.groups()
        if role == "feature_projection":
            return [backend + "/target_features"]
        if role == "candidate_selector/hidden_projection":
            return [backend + "/final_hidden"]
        match = re.fullmatch(r"(layers/\d+/)(.+)", role)
        if match:
            block, role = match.groups()
            block = backend + "/" + block
            if role in ("attention/context_key", "attention/context_value"):
                return [backend + "/context_input"]
            if role in ("attention/query", "attention/key", "attention/value"):
                return [block + "query_projection_input"]
            if role == "attention/output":
                return [block + "attention_output"]
            if role in ("mlp/gate", "mlp/up"):
                return [block + "mlp_input"]
            if role == "mlp/down":
                return [block + "mlp_product"]
            if role in (
                "attention_conv/kernel_projection",
                "mlp_conv/kernel_projection",
            ):
                return [block + role.split("/")[0] + "_input"]
    return []


def make_directory(identity, old_objects):
    key = (identity["model_id"], identity["weights_id"])
    if key not in KNOWN_COUNTS or len(old_objects) not in KNOWN_COUNTS[key]:
        raise ValueError(f"unsupported v2 input {key} with {len(old_objects)} objects")
    moe = key[0] == "qwen3.6-35b-a3b"
    has_dflash2 = any(o["name"].startswith("dflash2/") for o in old_objects)
    text = text_config(moe)
    components = {
        "text": {
            "config": text,
            "resources": {},
            "proposal": {"domain": "indexed", "rows": 131072},
        },
        "vision": {"config": vision_config(moe), "target": "text", "resources": {}},
        "mtp": {
            "config": {"architectures": ["Qwen3_5MoeMTP" if moe else "Qwen3_5MTP"]},
            "target": "text",
        },
    }
    if moe:
        components["dflash"] = {"config": draft_config(False), "target": "text"}
    if has_dflash2:
        components["dflash2"] = {"config": draft_config(True), "target": "text"}
    objects = []
    for value in old_objects:
        obj = {
            "id": value["name"],
            "kind": value["kind"],
            "offset": value["offset"],
            "bytes": value["bytes"],
        }
        if obj["kind"] == "tensor":
            obj.update(
                shape=value["shape"],
                format=FORMATS[value["format"]],
                layout=LAYOUTS[value["layout"]],
            )
            if obj["offset"] % 256 or obj["bytes"] != encoded_bytes(obj):
                raise ValueError(
                    f"{obj['id']}: invalid tensor geometry/encoding length"
                )
        elif obj["kind"] == "resource" and value["encoding"] == "raw-bytes-v1":
            obj["encoding"] = "raw_bytes_v1"
        else:
            raise ValueError(f"{obj['id']}: unsupported object kind or encoding")
        objects.append(obj)
    by_id = {o["id"]: o for o in objects}
    if len(by_id) != len(objects):
        raise ValueError("duplicate v2 object names")
    bindings, parameter_formats, scalar_uses = {}, {}, {}

    def put(name, obj, shape, bounds=None):
        if name in bindings:
            raise ValueError(f"duplicate logical mapping {name}")
        if bounds is None:
            if tuple(obj["shape"]) != tuple(shape):
                raise ValueError(f"{obj['id']}: expected {shape}, got {obj['shape']}")
            binding = {"object": obj["id"]}
        else:
            begin, end = bounds
            if not 0 <= begin < end <= math.prod(
                obj["shape"]
            ) or end - begin != math.prod(shape):
                raise ValueError(f"{name}: invalid known row mapping")
            binding = {"parts": [{"object": obj["id"], "range": [begin, end]}]}
        bindings[name] = binding
        parameter_formats[name] = obj["format"]

    def split(obj, entries, width):
        cursor = 0
        for name, rows in entries:
            put(name, obj, (rows, width), (cursor * width, (cursor + rows) * width))
            cursor += rows
        if tuple(obj["shape"]) != (cursor, width):
            raise ValueError(f"{obj['id']}: unexpected fused parent shape")

    h, r = text["hidden_size"], text["vocab_size"]
    q, k, nv = (
        text["num_attention_heads"] * 256,
        text["num_key_value_heads"] * 256,
        text["linear_num_value_heads"],
    )
    kg, vg, channels = 2048, nv * 128, 4096 + nv * 128
    for obj in objects:
        name = obj["id"]
        if obj["kind"] == "resource":
            if not name.startswith("frontend/"):
                raise ValueError(f"unknown official resource {name}")
            role = name.split("/", 1)[1]
            component = (
                "vision"
                if role
                in ("preprocessor_config.json", "video_preprocessor_config.json")
                else "text"
            )
            components[component]["resources"][role] = name
            continue
        if name.endswith("/input_scale_divisor"):
            if obj["format"] != "fp32" or obj["shape"] != []:
                raise ValueError(f"{name}: invalid activation scalar")
            prefix = name.removesuffix("/input_scale_divisor")
            group, operation = prefix.rsplit("/", 1)
            roles = {
                "input_projection": (
                    ("query", "key", "gate", "value")
                    if group.endswith("/attention")
                    else ("query", "key", "value", "z")
                ),
                "output_projection": ("output",),
                "gate_up_projection": ("gate", "up"),
                "down_projection": ("down",),
            }.get(operation)
            if roles is None:
                raise ValueError(f"unknown activation scalar {name}")
            for role in roles:
                scalar_uses[group + "/" + role] = {"object": name}
            continue
        if name in ("text/token_embedding", "text/output_head", "text/final_norm"):
            put(name, obj, (h,) if name.endswith("final_norm") else (r, h))
            continue
        if name in ("text/draft_head", "text/draft_head_token_ids"):
            put(
                (
                    "proposal/head"
                    if name.endswith("draft_head")
                    else "proposal/token_ids"
                ),
                obj,
                (131072, h) if name.endswith("draft_head") else (131072,),
            )
            continue
        if name in (
            "mtp/input_projection",
            "mtp/embedding_norm",
            "mtp/hidden_norm",
            "mtp/final_norm",
        ):
            put(name, obj, (h, 2 * h) if name.endswith("input_projection") else (h,))
            continue
        match = re.fullmatch(r"(text/layers/(\d+)/|mtp/layer/)(.+)", name)
        if match:
            old_block, layer, role = match.groups()
            if layer is not None and not 0 <= int(layer) < text["num_hidden_layers"]:
                raise ValueError(f"{name}: layer outside known model")
            block = "mtp/layers/0/" if old_block == "mtp/layer/" else old_block
            if role in ("input_norm", "post_attention_norm"):
                put(block + role, obj, (h,))
            elif role in ("attention/query_norm", "attention/key_norm"):
                put(block + role, obj, (256,))
            elif role == "attention/output":
                put(block + role, obj, (h, q))
            elif role in (
                "attention/query_key",
                "attention/gate_value",
                "attention/query_key_gate_value",
            ):
                roles = (
                    ("query", "key")
                    if role.endswith("/query_key")
                    else (
                        ("gate", "value")
                        if role.endswith("/gate_value")
                        else ("query", "key", "gate", "value")
                    )
                )
                split(
                    obj,
                    [
                        (block + "attention/" + v, q if v in ("query", "gate") else k)
                        for v in roles
                    ],
                    h,
                )
            elif role in ("gdn/query_key", "gdn/value_z", "gdn/query_key_value_z"):
                roles = (
                    ("query", "key")
                    if role.endswith("/query_key")
                    else (
                        ("value", "z")
                        if role.endswith("/value_z")
                        else ("query", "key", "value", "z")
                    )
                )
                split(
                    obj,
                    [
                        (block + "gdn/" + v, kg if v in ("query", "key") else vg)
                        for v in roles
                    ],
                    h,
                )
            elif role == "gdn/a_b_projection":
                split(
                    obj,
                    [
                        (block + "gdn/a_projection", nv),
                        (block + "gdn/b_projection", nv),
                    ],
                    h,
                )
            elif role in ("gdn/a_projection", "gdn/b_projection"):
                put(block + role, obj, (nv, h))
            elif role in (
                "gdn/a_log",
                "gdn/dt_bias",
                "gdn/norm",
                "gdn/convolution",
                "gdn/output",
            ):
                shape = (
                    (nv,)
                    if role in ("gdn/a_log", "gdn/dt_bias")
                    else (
                        (128,)
                        if role == "gdn/norm"
                        else (4, channels) if role == "gdn/convolution" else (h, vg)
                    )
                )
                put(block + role, obj, shape)
            elif role == "mlp/gate_up" and not moe:
                split(obj, [(block + "mlp/gate", 17408), (block + "mlp/up", 17408)], h)
            elif role == "mlp/down" and not moe:
                put(block + role, obj, (h, 17408))
            elif role.startswith("moe/") and moe:
                prefix = block + "moe/"
                if role == "moe/router_shared_gate":
                    split(
                        obj, [(prefix + "router", 256), (prefix + "shared_score", 1)], h
                    )
                elif role == "moe/routed_gate_up":
                    split(
                        obj,
                        [
                            (prefix + f"experts/{e}/" + part, 512)
                            for e in range(256)
                            for part in ("gate", "up")
                        ],
                        h,
                    )
                elif role == "moe/routed_down":
                    split(
                        obj,
                        [(prefix + f"experts/{e}/down", h) for e in range(256)],
                        512,
                    )
                elif role == "moe/shared_gate_up":
                    split(
                        obj,
                        [(prefix + "shared/gate", 512), (prefix + "shared/up", 512)],
                        h,
                    )
                elif role == "moe/shared_down":
                    put(prefix + "shared/down", obj, (h, 512))
                else:
                    raise ValueError(f"unknown MoE object {name}")
            else:
                raise ValueError(f"unknown decoder object {name}")
            continue
        if name.startswith("vision/"):
            if name in (
                "vision/patch_embedding",
                "vision/patch_embedding_bias",
                "vision/position_embedding",
            ):
                shape = (
                    (1152, 1536)
                    if name.endswith("patch_embedding")
                    else (
                        (1152,)
                        if name.endswith("patch_embedding_bias")
                        else (2304, 1152)
                    )
                )
                put(name, obj, shape)
                continue
            match = re.fullmatch(r"(vision/layers/(\d+)/)(.+)", name)
            if match:
                block, layer, role = match.groups()
                if not 0 <= int(layer) < 27:
                    raise ValueError("Vision layer outside known model")
                if role == "attention/qkv":
                    split(
                        obj,
                        [
                            (block + "attention/" + v, 1152)
                            for v in ("query", "key", "value")
                        ],
                        1152,
                    )
                elif role == "attention/qkv_bias":
                    if obj["shape"] != [3456]:
                        raise ValueError("unexpected Vision qkv bias shape")
                    for index, v in enumerate(("query", "key", "value")):
                        put(
                            block + "attention/" + v + "_bias",
                            obj,
                            (1152,),
                            (index * 1152, (index + 1) * 1152),
                        )
                else:
                    sizes = {
                        "attention/output": (1152, 1152),
                        "attention/output_bias": (1152,),
                        "mlp/fc1": (4304, 1152),
                        "mlp/fc1_bias": (4304,),
                        "mlp/fc2": (1152, 4304),
                        "mlp/fc2_bias": (1152,),
                        "norm1/weight": (1152,),
                        "norm1/bias": (1152,),
                        "norm2/weight": (1152,),
                        "norm2/bias": (1152,),
                    }
                    role_name = role.replace("norm1/", "norm1_").replace(
                        "norm2/", "norm2_"
                    )
                    put(block + role_name, obj, sizes[role])
                continue
            role = name.removeprefix("vision/merger/")
            sizes = {
                "fc1": (4608, 4608),
                "fc1_bias": (4608,),
                "fc2": (h, 4608),
                "fc2_bias": (h,),
                "norm/weight": (1152,),
                "norm/bias": (1152,),
            }
            if role not in sizes:
                raise ValueError(f"unknown Vision object {name}")
            put("vision/merger/" + role.replace("norm/", "norm_"), obj, sizes[role])
            continue
        match = re.fullmatch(r"(dflash2?)/(.*)", name)
        if match:
            backend, role = match.groups()
            if backend not in components:
                raise ValueError(f"unexpected backend {backend}")
            config = components[backend]["config"]
            inter = config["intermediate_size"]
            if role in ("feature_projection", "context_norm", "final_norm"):
                shape = (
                    (h, len(config["dflash_config"]["target_layer_ids"]) * h)
                    if role == "feature_projection"
                    else (h,)
                )
                put(name, obj, shape)
                continue
            if role.startswith("candidate_selector/") and backend == "dflash2":
                shape = (256, h) if role.endswith("hidden_projection") else (r, 256)
                put(name, obj, shape)
                continue
            match = re.fullmatch(r"(layers/(\d+)/)(.+)", role)
            if match is None:
                raise ValueError(f"unknown draft object {name}")
            relative, layer, role = match.groups()
            if not 0 <= int(layer) < config["num_hidden_layers"]:
                raise ValueError(f"{name}: layer outside known draft")
            block = backend + "/" + relative
            if role == "attention/query_key_value":
                split(
                    obj,
                    [
                        (block + "attention/query", 4096),
                        (block + "attention/key", 1024),
                        (block + "attention/value", 1024),
                    ],
                    h,
                )
                put(
                    block + "attention/context_key",
                    obj,
                    (1024, h),
                    (4096 * h, 5120 * h),
                )
                put(
                    block + "attention/context_value",
                    obj,
                    (1024, h),
                    (5120 * h, 6144 * h),
                )
            elif role == "mlp/gate_up":
                split(obj, [(block + "mlp/gate", inter), (block + "mlp/up", inter)], h)
            else:
                sizes = {
                    "input_norm": (h,),
                    "post_attention_norm": (h,),
                    "attention/query_norm": (128,),
                    "attention/key_norm": (128,),
                    "attention/output": (h, 4096),
                    "mlp/down": (h, inter),
                }
                if backend == "dflash2":
                    for branch in ("attention_conv", "mlp_conv"):
                        sizes[branch + "/base_kernel"] = (2, 2, h)
                        sizes[branch + "/kernel_projection"] = (h // 4, h)
                put(block + role, obj, sizes[role])
            continue
        raise ValueError(f"unknown stored tensor {name}")

    uses = []
    for name in bindings:
        format = parameter_formats[name]
        policy = (
            "AllowA4"
            if format == "nvfp4"
            else "AllowA8" if format == "fp8_e4m3fn_row_bf16" else "A16Only"
        )
        for input_name in _input_names(name, components):
            use = {"parameter": name, "input": input_name, "activation_policy": policy}
            if name in scalar_uses:
                use["auxiliaries"] = {"activation_input_divisor": scalar_uses[name]}
            uses.append(use)
    if scalar_uses.keys() - bindings.keys():
        raise ValueError("activation scalar refers to an absent logical projection")
    expected_bindings = 32713 if moe else 1513 if has_dflash2 else 1422
    if len(bindings) != expected_bindings:
        raise ValueError("the known logical parameter mapping is incomplete")
    for component, required in (
        (
            "text",
            {
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
            },
        ),
        ("vision", {"preprocessor_config.json", "video_preprocessor_config.json"}),
    ):
        if set(components[component]["resources"]) != required:
            raise ValueError("the known frontend resource set is incomplete")
    return {
        "components": components,
        "objects": objects,
        "bindings": bindings,
        "uses": uses,
        "metadata": {"name": key[0]},
        "provenance": {
            "upgraded_from": {"version": 2, "model_id": key[0], "weights_id": key[1]}
        },
    }


def read_v2(path):
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16 or header[:8] != b"NINFER\0\2":
            raise ValueError("expected NInfer v2 input")
        count = struct.unpack_from("<Q", header, 8)[0]
        size = os.fstat(stream.fileno()).st_size
        if count == 0 or count > size - 16:
            raise ValueError("invalid v2 directory length")
        directory = json.loads(stream.read(count))
        start = align(16 + count)
        if start >= size or set(directory) != {"identity", "objects"}:
            raise ValueError("invalid v2 framing or directory")
        previous = 0
        for obj in directory["objects"]:
            begin, length = obj["offset"], obj["bytes"]
            if (
                type(begin) is not int
                or type(length) is not int
                or length <= 0
                or not previous <= begin
                or begin + length > size - start
            ):
                raise ValueError("invalid v2 object ranges")
            previous = begin + length
            if obj["name"].endswith("/input_scale_divisor"):
                stream.seek(start + begin)
                scalar = struct.unpack("<f", stream.read(4))[0]
                if not math.isfinite(scalar) or scalar <= 0:
                    raise ValueError("invalid v2 activation scalar")
        return directory, start, size - start


def layout(entry, directory, payload):
    reserve = 4096 - 32
    while True:
        start = reserve + 32
        if start + payload <= LIMIT:
            files = [{"path": None, "payload_bytes": payload}]
        else:
            first = (LIMIT - start) // 4096 * 4096
            capacity = (LIMIT - 4096) // 4096 * 4096
            if min(first, capacity) <= 0:
                raise ValueError("metadata cannot fit the file limit")
            files = [{"path": None, "payload_bytes": first}]
            remaining = payload - first
            while remaining:
                count = min(remaining, capacity)
                files.append(
                    {
                        "path": f"{entry.name}.part-{len(files):04d}",
                        "payload_bytes": count,
                    }
                )
                remaining -= count
        directory["files"] = files
        data = json.dumps(
            directory, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        if len(data) <= reserve:
            return data + b" " * (reserve - len(data)), start
        reserve = align(32 + len(data)) - 32


def upgrade(input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must differ")
    old, source_start, source_payload = read_v2(input_path)
    directory = make_directory(old["identity"], old["objects"])
    template_name = (
        "qwen3_8.jinja"
        if old["identity"]["model_id"].startswith("qwen3.8-")
        else "qwen3_6.jinja"
    )
    template = (
        Path(__file__).resolve().parent / "chat_templates" / template_name
    ).read_bytes()
    template_id = directory["components"]["text"]["resources"]["chat_template.jinja"]
    template_object = next(
        obj for obj in directory["objects"] if obj["id"] == template_id
    )
    # Append the replacement so every existing weight keeps its original payload offset.
    template_offset = align(source_payload, 256)
    template_object.update(offset=template_offset, bytes=len(template))
    directory["objects"].remove(template_object)
    directory["objects"].append(template_object)
    payload = template_offset + len(template)
    data, entry_start = layout(output_path, directory, payload)
    targets = [output_path] + [
        output_path.parent / f["path"] for f in directory["files"][1:]
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("an output file already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().bytes
    temporary, published = [], []
    try:
        with input_path.open("rb") as source:
            source.seek(source_start)
            cursor = 0
            for index, (target, file) in enumerate(zip(targets, directory["files"])):
                fd, temp = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
                temporary.append(Path(temp))
                with os.fdopen(fd, "wb") as output:
                    output.write(
                        HEADER.pack(
                            b"NINFER\0\3" if index == 0 else b"NINPRT\0\3",
                            len(data) if index == 0 else index,
                            identity,
                        )
                    )
                    if index == 0:
                        output.write(data)
                    start = entry_start if index == 0 else 4096
                    output.write(bytes(start - output.tell()))
                    remaining = file["payload_bytes"]
                    pending = output.tell()
                    while remaining:
                        if cursor < source_payload:
                            chunk = source.read(
                                min(remaining, CHUNK, source_payload - cursor)
                            )
                            if not chunk:
                                raise ValueError("v2 payload ended prematurely")
                            _dontneed(
                                source.fileno(),
                                source.tell() - len(chunk),
                                len(chunk),
                            )
                        elif cursor < template_offset:
                            chunk = bytes(min(remaining, template_offset - cursor))
                        else:
                            begin = cursor - template_offset
                            chunk = template[begin : begin + min(remaining, CHUNK)]
                        output.write(chunk)
                        cursor += len(chunk)
                        remaining -= len(chunk)
                        pending += len(chunk)
                        if pending >= WRITEBACK:
                            output.flush()
                            _fdatasync(output.fileno())
                            _dontneed(output.fileno())
                            pending = 0
                    output.flush()
                    _fdatasync(output.fileno())
                    _dontneed(output.fileno())
            _dontneed(source.fileno())
        for index in [*range(1, len(targets)), 0]:
            try:
                os.link(temporary[index], targets[index])
            except OSError:
                # NTFS hard links work, but cross-volume or locked targets fall
                # back to a same-directory-safe copy-then-rename publish.
                import shutil

                shutil.copyfile(temporary[index], targets[index])
            published.append(targets[index])
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    print(
        f"upgraded {input_path} -> {output_path}: weights preserved, {template_name} installed, {len(targets)} files",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    upgrade(args.input, args.output)


if __name__ == "__main__":
    main()
