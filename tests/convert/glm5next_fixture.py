"""Small GLM checkpoint shared by converter and C++ weight-binding tests."""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from tools.convert.sources.safetensors import SafetensorsSource


def build_checkpoint(path: Path) -> SafetensorsSource:
    path.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        hidden_size=128,
        vocab_size=16,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        mla_use_nope=True,
        index_n_heads=2,
        index_head_dim=64,
        index_kpool=4,
        index_topk=16,
        index_kpool_always_select_tail=True,
        index_kpool_compress=True,
        index_share_for_mtp_iteration=True,
        indexer_types=["full", "full"],
        indexer_rope_interleave=True,
        linear_num_heads=2,
        linear_head_dim=64,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        mhc=True,
        intermediate_size=256,
        moe_intermediate_size=128,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        moe_router_dtype="float32",
        swiglu_limit=10.0,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        attention_bias=False,
        tie_word_embeddings=False,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        num_nextn_predict_layers=1,
        bos_token_id=None,
        eos_token_id=[2],
        pad_token_id=2,
    )
    vision = dict(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=1,
        out_hidden_size=128,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        projection_intermediate_size=128,
        rms_norm_eps=1e-5,
        swiglu_limit=10.0,
        hidden_act="silu",
        attention_bias=True,
    )
    config = dict(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=cfg,
        vision_config=vision,
    )
    tensors = {}

    def add(name, shape, dtype=torch.bfloat16):
        count = prod(shape)
        tensors[name] = (
            (((torch.arange(count) + len(tensors) * 7) % 31 - 15) / 16)
            .to(dtype)
            .reshape(shape)
        )

    def nvfp4(prefix, n, k, multiplier=1.0):
        tensors[prefix + ".weight"] = torch.full((n, k // 2), 0x21, dtype=torch.uint8)
        tensors[prefix + ".weight_scale"] = torch.ones(n, k // 16).to(
            torch.float8_e4m3fn
        )
        tensors[prefix + ".weight_scale_2"] = torch.tensor(
            multiplier, dtype=torch.float32
        )
        tensors[prefix + ".input_scale"] = torch.tensor(2.0, dtype=torch.float32)

    def mla(s):
        for field, shape in {
            "kv_a_proj_with_mqa.weight": (128, 128),
            "q_a_proj.weight": (128, 128),
            "q_b_proj.weight": (128, 128),
            "kv_b_proj.weight": (256, 128),
            "o_proj.weight": (128, 128),
            "q_a_layernorm.weight": (128,),
            "kv_a_layernorm.weight": (128,),
            "indexer.wk.weight": (64, 128),
            "indexer.wq_b.weight": (128, 128),
            "indexer.weights_proj.weight": (2, 128),
            "indexer.index_kpool_compress_gate": (64, 128),
            "indexer.index_kpool_compress_ape": (4, 64),
            "indexer.k_norm.weight": (64,),
            "indexer.k_norm.bias": (64,),
        }.items():
            add(s + "self_attn." + field, shape)

    def moe(s, encoded):
        add(s + "mlp.gate.weight", (2, 128))
        add(s + "mlp.gate.e_score_correction_bias", (2,), torch.float32)
        for role in ("gate", "up", "down"):
            add(s + "mlp.shared_experts." + role + "_proj.weight", (128, 128))
            for expert in range(2):
                prefix = s + f"mlp.experts.{expert}.{role}_proj"
                if encoded:
                    nvfp4(prefix, 128, 128, multiplier=expert + 1.0)
                else:
                    add(prefix + ".weight", (128, 128))

    add("model.language_model.embed_tokens.weight", (16, 128))
    add("lm_head.weight", (16, 128))
    add("model.language_model.norm.weight", (128,))
    for layer in range(2):
        s = f"model.language_model.layers.{layer}."
        add(s + "input_layernorm.weight", (128,))
        add(s + "post_attention_layernorm.weight", (128,))
        for site in ("attn", "ffn"):
            add(s + f"hc_{site}_fn", (24, 512))
            add(s + f"hc_{site}_base", (24,))
            add(s + f"hc_{site}_scale", (3,))
        if layer == 0:
            a = s + "self_attn."
            for role in ("q", "k", "v"):
                add(a + role + "_proj.weight", (128, 128))
                add(a + role + "_conv1d.weight", (128, 1, 4), torch.float32)
            add(a + "o_proj.weight", (128, 128))
            add(a + "b_proj.weight", (2, 128))
            for role in ("f", "g"):
                add(a + role + "_a_proj.weight", (32, 128))
                add(a + role + "_b_proj.weight", (128, 32))
            add(a + "o_norm.weight", (64,))
            add(a + "A_log", (2,), torch.float32)
            add(a + "dt_bias", (128,), torch.float32)
            for role in ("gate", "up", "down"):
                nvfp4(
                    s + "mlp." + role + "_proj",
                    128 if role == "down" else 256,
                    256 if role == "down" else 128,
                )
        else:
            mla(s)
            moe(s, True)
    s = "model.language_model.layers.2."
    for field in (
        "input_layernorm",
        "post_attention_layernorm",
        "enorm",
        "hnorm",
        "shared_head.norm",
    ):
        add(s + field + ".weight", (128,))
    add(s + "eh_proj.weight", (128, 256))
    mla(s)
    moe(s, False)
    add("model.visual.patch_embed.proj.weight", (64, 3, 2, 2, 2))
    add("model.visual.patch_embed.proj.bias", (64,))
    add("model.visual.post_layernorm.weight", (64,))
    s = "model.visual.blocks.0."
    for field, shape in {
        "norm1.weight": (64,),
        "norm2.weight": (64,),
        "attn.q_norm.weight": (64,),
        "attn.k_norm.weight": (64,),
        "attn.qkv.weight": (192, 64),
        "attn.qkv.bias": (192,),
        "attn.proj.weight": (64, 64),
        "attn.proj.bias": (64,),
        "mlp.gate_proj.weight": (128, 64),
        "mlp.up_proj.weight": (128, 64),
        "mlp.gate_proj.bias": (128,),
        "mlp.up_proj.bias": (128,),
        "mlp.down_proj.weight": (64, 128),
        "mlp.down_proj.bias": (64,),
    }.items():
        add(s + field, shape)
    add("model.visual.downsample.weight", (128, 64, 2, 2))
    add("model.visual.downsample.bias", (128,))
    for field, shape in {
        "proj.weight": (128, 128),
        "post_projection_norm.weight": (128,),
        "post_projection_norm.bias": (128,),
        "gate_proj.weight": (128, 128),
        "up_proj.weight": (128, 128),
        "down_proj.weight": (128, 128),
    }.items():
        add("model.visual.merger." + field, shape)
    save_file(tensors, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config))
    tokenizer = json.loads(
        (
            Path(__file__).parents[1] / "models/glm5next/fixtures/prompt-tokenizer.json"
        ).read_text()
    )
    tokenizer["model"]["vocab"] = {"a": 0, "b": 1}
    tokenizer["model"]["merges"] = []
    tokenizer["added_tokens"] = [
        dict(
            id=2,
            content="<eos>",
            special=True,
            single_word=False,
            normalized=False,
            lstrip=False,
            rstrip=False,
        )
    ]
    tokenizer["added_tokens"].append(
        dict(
            id=3,
            content="</think>",
            special=False,
            single_word=False,
            normalized=False,
            lstrip=False,
            rstrip=False,
        )
    )
    processor = dict(
        do_rescale=True,
        patch_expand_factor=1,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        temporal_patch_size=2,
        patch_size=2,
        min_image_tokens=1,
        max_image_tokens=64,
    )
    for name, content in {
        "tokenizer.json": tokenizer,
        "tokenizer_config.json": {},
        "generation_config.json": {"eos_token_id": [2]},
        "processor_config.json": {
            "image_processor": processor,
            "video_processor": {**processor, "fps": 2},
        },
    }.items():
        (path / name).write_text(json.dumps(content))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def build_native_checkpoint(path: Path) -> SafetensorsSource:
    """Small native checkpoint with independent expert calibrations and CSF exceptions."""
    with build_checkpoint(path):
        pass
    tensors = load_file(path / "model.safetensors")
    config = json.loads((path / "config.json").read_text())
    text = config["text_config"]
    text["linear_attn_config"] = {
        "num_heads": text.pop("linear_num_heads"),
        "head_dim": text.pop("linear_head_dim"),
        "short_conv_kernel_size": text.pop("linear_conv_kernel_dim"),
        "gate_lower_bound": text.pop("linear_lower_bound"),
        "kda_layers": [0],
        "full_attn_layers": [1],
    }
    text.pop("bos_token_id")
    config["quantization_config"] = {
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
    }

    def encode(prefix, format, *, multiplier=1.0, activation=None):
        weight = tensors[prefix + ".weight"]
        n, k = weight.shape
        if weight.dtype == torch.uint8:
            k *= 2
        for name in tuple(tensors):
            if name.startswith(prefix + ".") and name.rsplit(".", 1)[-1] != "bias":
                tensors.pop(name)
        if format == "mxfp8":
            tensors[prefix + ".weight"] = torch.ones(n, k).to(torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale"] = torch.full(
                (n, k // 32), 127, dtype=torch.uint8
            )
        elif format == "block":
            tensors[prefix + ".weight"] = torch.ones(n, k).to(torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale"] = torch.full(
                (n // 128, 1, k // 128, 1), 1.000123, dtype=torch.float32
            )
        else:
            tensors[prefix + ".weight"] = torch.full(
                (n, k // 2), 0x21, dtype=torch.uint8
            )
            tensors[prefix + ".weight_scale_2"] = torch.tensor(
                multiplier, dtype=torch.float32
            )
            if activation is not None:
                tensors[prefix + ".input_scale"] = torch.tensor(
                    activation, dtype=torch.float32
                )
            if format == "csf":
                # All words are 0x38 except the first block's first lane (0x40).
                blocks = n // 32 * (k // 32)
                assert blocks < 256
                base = prefix + ".weight_scale."
                tensors[base + "csf32_bases"] = torch.full(
                    (n,), 0x38, dtype=torch.uint8
                )
                tensors[base + "csf32_codes"] = torch.zeros(
                    n * k // 32, dtype=torch.uint8
                )
                offsets = torch.ones(blocks + 1, dtype=torch.int32)
                offsets[0] = 0
                tensors[base + "csf32_offsets"] = offsets.to(torch.uint16)
                tensors[base + "csf32_page_bases"] = torch.zeros(1, dtype=torch.uint32)
                tensors[base + "csf32_exceptions"] = torch.tensor(
                    [0x4000], dtype=torch.uint16
                )
            else:
                tensors[prefix + ".weight_scale"] = torch.ones(n, k // 16).to(
                    torch.float8_e4m3fn
                )

    encode("lm_head", "mxfp8")
    for layer in range(2):
        p = f"model.language_model.layers.{layer}."
        for name in (
            "post_attention_layernorm.weight",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            tensors[p + name] = tensors[p + name].float() + 0.000123
    tensors["model.language_model.norm.weight"] = (
        tensors["model.language_model.norm.weight"].float() + 0.000123
    )
    p = "model.language_model.layers.0."
    for role in ("q", "k", "v", "o"):
        encode(p + "self_attn." + role + "_proj", "mxfp8")
        if role != "o":
            name = p + "self_attn." + role + "_conv1d.weight"
            tensors[name] = tensors[name].bfloat16()
    for role in ("gate", "up", "down"):
        encode(p + "mlp." + role + "_proj", "mxfp8")
    for layer in (1, 2):
        p = f"model.language_model.layers.{layer}."
        projections = ["kv_a_proj_with_mqa", "q_a_proj", "q_b_proj", "o_proj"]
        if layer == 2:
            projections += [
                "kv_b_proj",
                "indexer.wk",
                "indexer.wq_b",
                "indexer.weights_proj",
            ]
        for name in projections:
            encode(p + "self_attn." + name, "block" if layer == 1 else "mxfp8")
        for role in ("gate", "up", "down"):
            encode(
                p + "mlp.shared_experts." + role + "_proj",
                "block" if layer == 1 else "mxfp8",
            )
            for expert in range(2):
                encode(
                    p + f"mlp.experts.{expert}.{role}_proj",
                    "csf" if layer == 1 else "nvfp4",
                    multiplier=expert + 1.25,
                    activation=(
                        (expert + 1.5 + (role == "down")) if layer == 1 else None
                    ),
                )
    # Keep the Vision mathematics while using N128 matrices for its native NVFP4 projections.
    config["vision_config"]["hidden_size"] = 128
    config["vision_config"]["num_heads"] = 2
    shapes = {
        "model.visual.patch_embed.proj.weight": (128, 3, 2, 2, 2),
        "model.visual.patch_embed.proj.bias": (128,),
        "model.visual.post_layernorm.weight": (128,),
        "model.visual.downsample.weight": (128, 128, 2, 2),
    }
    v = "model.visual.blocks.0."
    shapes.update(
        {
            v + name: dims
            for name, dims in {
                "norm1.weight": (128,),
                "norm2.weight": (128,),
                "attn.qkv.weight": (384, 128),
                "attn.qkv.bias": (384,),
                "attn.proj.weight": (128, 128),
                "attn.proj.bias": (128,),
                "mlp.gate_proj.weight": (128, 128),
                "mlp.up_proj.weight": (128, 128),
                "mlp.down_proj.weight": (128, 128),
                "mlp.down_proj.bias": (128,),
            }.items()
        }
    )
    for name, shape in shapes.items():
        tensors[name] = torch.ones(shape, dtype=torch.bfloat16)
    for name in ("attn.qkv", "attn.proj"):
        encode(v + name, "mxfp8")
    for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        encode(v + name, "nvfp4", multiplier=1.25)
    for name in ("proj", "gate_proj", "up_proj", "down_proj"):
        encode("model.visual.merger." + name, "nvfp4", multiplier=1.25)
    nested = {name: value for name, value in tensors.items() if ".q_b_proj." in name}
    main = {name: value for name, value in tensors.items() if name not in nested}
    (path / "tensors").mkdir()
    save_file(main, path / "model.safetensors")
    save_file(nested, path / "tensors/native-QB.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: (
                        "tensors/native-QB.safetensors"
                        if name in nested
                        else "model.safetensors"
                    )
                    for name in tensors
                }
            }
        )
    )
    (path / "config.json").write_text(json.dumps(config))
    return SafetensorsSource(path)
