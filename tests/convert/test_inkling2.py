from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.layouts import block_scale_geometry, row_group_scale_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.__main__ import main as convert_main
from tools.convert.inkling2 import build_model
from tools.convert.sources.safetensors import SafetensorsSource

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def _parts(artifact: Artifact, name: str):
    return binding_parts(artifact.directory.bindings[name], artifact.by_id)


def _direct(artifact: Artifact, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    chunks = []
    for object_id, first, last in _parts(artifact, name):
        obj = artifact.object(object_id)
        data = decode_direct(artifact.read_object(object_id), obj.format, obj.shape)
        chunks.append(data.flatten()[first:last])
    return torch.cat(chunks).reshape(shape)


def _write_tokenizer_resources(root: Path, unpadded_vocab: int, *, with_vision: bool):
    vocab = {f"tok_{i}": i for i in range(unpadded_vocab - 4)}
    added = [
        {"id": unpadded_vocab - 4, "content": "<|endoftext|>", "special": True},
        {"id": unpadded_vocab - 3, "content": "<|content_model_end_sampling|>", "special": True},
        {"id": unpadded_vocab - 2, "content": "<|content_image|>", "special": True},
        {"id": unpadded_vocab - 1, "content": "<|unused_img|>", "special": True},
    ]
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added})
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    str(item["id"]): {
                        "content": item["content"],
                        "special": item["special"],
                    }
                    for item in added
                }
            }
        )
    )
    (root / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    (root / "generation_config.json").write_text(
        json.dumps({"eos_token_id": unpadded_vocab - 3, "pad_token_id": unpadded_vocab - 4})
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<|content_model_end_sampling|>", "pad_token": "<|endoftext|>"})
    )
    if with_vision:
        (root / "processor_config.json").write_text(
            json.dumps(
                {
                    "image_token": "<|unused_img|>",
                    "image_processor": {"size": {"height": 40, "width": 40}},
                }
            )
        )


def _build_inkling2_fixture(root: Path, *, looped: bool = True, expert_mode: str = "mixed"):
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    h, inter, experts = 128, 128, 4
    hq, hkv, d, d_rel, rel_extent, taps = 4, 2, 32, 16, 32, 4
    q_dim, r_dim, kv_dim = hq * d, hq * d_rel, hkv * d
    raw_vocab, unpadded_vocab = 256, 130
    num_layers = 6 if looped else 2
    local_ids = [0, 2, 4] if looped else [0]
    attn_res_block = 2
    gate_rank = 16

    text_cfg = {
        "model_type": "inkling2_loop_text" if looped else "inkling2_text",
        "model_max_length": 4096,
        "hidden_size": h,
        "num_hidden_layers": num_layers,
        "vocab_size": raw_vocab,
        "unpadded_vocab_size": unpadded_vocab,
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": d,
        "swa_num_attention_heads": hq,
        "swa_num_key_value_heads": hkv,
        "swa_head_dim": d,
        "d_rel": d_rel,
        "rel_extent": rel_extent,
        "sliding_window_size": 32,
        "local_layer_ids": local_ids,
        "sconv_kernel_size": taps,
        "n_layers_per_attn_res_block": attn_res_block,
        "attn_res_normalize_values": True,
        "use_value_norm_no_gamma": True,
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "log_scaling_n_floor": 1024,
        "log_scaling_alpha": 0.1,
        "final_logit_softcapping": 30.0,
        "n_routed_experts": experts,
        "num_experts_per_tok": 2,
        "n_shared_experts": 0,
        "intermediate_size": inter,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": True,
    }
    if looped:
        text_cfg["loop_config"] = {
            "core_start": 2,
            "core_end": 4,
            "num_passes": 2,
            "gate_rank": gate_rank,
            "is_varloop": False,
        }

    v_h, v_layers, v_heads, v_patch, v_merge, v_grid, v_proj = 64, 2, 4, 4, 2, 4, 128
    vision_cfg = {
        "model_type": "inkling2_vision",
        "text_hidden_size": h,
        "temporal_patch_size": 2,
        "native_patch_size": v_patch,
        "adapter_merge_size": v_merge,
        "projection_hidden_dim": v_proj,
        "projector_bias": False,
        "hidden_size": v_h,
        "num_hidden_layers": v_layers,
        "num_attention_heads": v_heads,
        "mlp_ratio": 4.0,
        "posemb_grid_size": v_grid,
        "layer_norm_eps": 1e-5,
    }

    d_layers, d_inter, d_rank, d_block = 2, 256, 32, 4
    logical_layers = num_layers + (2 if looped else 0)
    mtp_cfg = {
        "hidden_size": h,
        "num_hidden_layers": d_layers,
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": d,
        "intermediate_size": d_inter,
        "max_position_embeddings": 4096,
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "local_layer_ids": [0],
        "sliding_window_size": 32,
        "block_size": d_block,
        "markov_rank": d_rank,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, logical_layers - 1, logical_layers],
    }

    full_cfg = {
        "architectures": [
            "Inkling2LoopForConditionalGeneration" if looped else "Inkling2ForCausalLM"
        ],
        "model_type": "inkling2_loop" if looped else "inkling2",
        "eos_token_id": unpadded_vocab - 3,
        "text_config": text_cfg,
    }
    if looped:
        full_cfg["vision_config"] = vision_cfg
        full_cfg["mtp_config"] = mtp_cfg
    (root / "config.json").write_text(json.dumps(full_cfg))
    _write_tokenizer_resources(root, unpadded_vocab, with_vision=looped)

    bf16 = lambda *shape: torch.randn(shape, dtype=torch.bfloat16)
    f32 = lambda *shape: torch.randn(shape, dtype=torch.float32)

    tensors: dict[str, torch.Tensor] = {
        "model.llm.embed.weight": bf16(raw_vocab, h),
        "model.llm.embed_norm.weight": bf16(h),
        "model.llm.norm.weight": bf16(h),
        "model.llm.unembed.weight": bf16(raw_vocab, h),
        "model.llm.attn_res.output_query": f32(h),
    }
    if looped:
        tensors["model.llm.middle_cycle_cross_pass_queries"] = f32(4, h)
        tensors["model.llm.middle_cycle_cross_pass_kv_queries"] = f32(4, h)

    for layer in range(num_layers):
        s = f"model.llm.layers.{layer}."
        is_swa = layer in local_ids
        if layer > 0:
            tensors[f"model.llm.attn_res.attn_queries.{layer - 1}"] = (
                f32(h) if is_swa else f32(3, h)
            )
        tensors[f"model.llm.attn_res.mlp_queries.{layer}"] = f32(h)
        if is_swa:
            tensors[s + "attn_norm.weight"] = bf16(h)
        else:
            for stream in ("q", "k", "v"):
                tensors[s + f"{stream}_input_norm.weight"] = bf16(h)
        tensors[s + "mlp_norm.weight"] = bf16(h)

        tensors[s + "attn.wq_du.weight"] = bf16(q_dim, h)
        tensors[s + "attn.wr_du.weight"] = bf16(r_dim, h)
        tensors[s + "attn.gate_attn.weight"] = bf16(q_dim, h)
        tensors[s + "attn.wk_dv.weight"] = bf16(kv_dim, h)
        tensors[s + "attn.wv_dv.weight"] = bf16(kv_dim, h)
        tensors[s + "attn.attn_sink_proj.weight"] = bf16(hq, h)
        tensors[s + "attn.wo_ud.weight"] = bf16(h, q_dim)
        tensors[s + "attn.q_l2_clip.scale"] = bf16(hq, d)
        tensors[s + "attn.rel_l2_clip.scale"] = bf16(hq, d_rel)
        tensors[s + "attn.rel_logits_proj.proj"] = bf16(d_rel, rel_extent)
        if is_swa:
            tensors[s + "attn.k_sconv.weight"] = bf16(kv_dim, 1, taps)
            tensors[s + "attn.v_sconv.weight"] = bf16(kv_dim, 1, taps)
        else:
            tensors[s + "attn.k_sconv.static_w"] = bf16(taps, kv_dim)
            tensors[s + "attn.v_sconv.static_w"] = bf16(taps, kv_dim)
            tensors[s + "attn.k_sconv.weight_proj"] = bf16(taps * hkv, h)
            tensors[s + "attn.v_sconv.weight_proj"] = bf16(taps * hkv, h)
        tensors[s + "attn_sconv.weight"] = bf16(h, 1, taps)
        tensors[s + "mlp_sconv.weight"] = bf16(h, 1, taps)

        if looped and 2 <= layer < 4:
            for st in ("attn", "k", "v", "mlp"):
                tensors[s + f"controllers.{st}.down_weight"] = bf16(gate_rank, 2 * h)
                tensors[s + f"controllers.{st}.gain_weight"] = bf16(h, gate_rank)
                tensors[s + f"controllers.{st}.forget_weight"] = bf16(h, gate_rank)

        tensors[s + "mlp.gate.weight"] = bf16(experts, h)
        tensors[s + "mlp.gate.bias"] = f32(experts)

        use_mode = (
            "bf16"
            if expert_mode == "bf16"
            else ("nvfp4" if layer < num_layers - 2 else "mxfp8")
        )
        ep = s + "mlp.experts."
        if use_mode == "nvfp4":
            # Distinct per-row & per-expert codes/scales so de-interleave bugs are loud
            w13_codes = torch.randint(0, 256, (experts, 2 * inter, h // 2), dtype=torch.uint8)
            w13_scales = torch.randint(
                0x20, 0x58, (experts, 2 * inter, h // 16), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            w2_codes = torch.randint(0, 256, (experts, h, inter // 2), dtype=torch.uint8)
            w2_scales = torch.randint(
                0x20, 0x58, (experts, h, inter // 16), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            tensors[ep + "w13_weight"] = w13_codes
            tensors[ep + "w13_weight_scale"] = w13_scales
            tensors[ep + "w13_weight_amax"] = torch.tensor(
                [0.25 + 0.1 * e + 0.01 * layer for e in range(experts)], dtype=torch.float32
            )
            tensors[ep + "w13_weight_outer_scale_denominator"] = torch.tensor(
                [2688.0], dtype=torch.float32
            )
            tensors[ep + "w2_weight"] = w2_codes
            tensors[ep + "w2_weight_scale"] = w2_scales
            tensors[ep + "w2_weight_amax"] = torch.tensor(
                [0.35 + 0.08 * e + 0.01 * layer for e in range(experts)], dtype=torch.float32
            )
            tensors[ep + "w2_weight_outer_scale_denominator"] = torch.tensor(
                [2688.0], dtype=torch.float32
            )
        elif use_mode == "mxfp8":
            # Avoid E4M3FN NaN bytes (0x7F / 0xFF) and UE8M0 NaN (255)
            w13_raw = torch.randint(0, 126, (experts, 2 * inter, h), dtype=torch.uint8)
            w2_raw = torch.randint(0, 126, (experts, h, inter), dtype=torch.uint8)
            tensors[ep + "w13_weight"] = w13_raw.view(torch.float8_e4m3fn)
            tensors[ep + "w13_weight_scale"] = torch.randint(
                120, 134, (experts, 2 * inter, h // 32), dtype=torch.uint8
            )
            tensors[ep + "w2_weight"] = w2_raw.view(torch.float8_e4m3fn)
            tensors[ep + "w2_weight_scale"] = torch.randint(
                120, 134, (experts, h, inter // 32), dtype=torch.uint8
            )
        else:
            tensors[ep + "w13_weight"] = bf16(experts, 2 * inter, h)
            tensors[ep + "w2_weight"] = bf16(experts, h, inter)

    if looped:
        ve = "model.visual.backbone.embeddings."
        tensors[ve + "patch_embedding.weight"] = bf16(v_h, 3 * v_patch * v_patch)
        tensors[ve + "position_embedding"] = bf16(v_grid * v_grid, v_h)
        tensors[ve + "class_embedding"] = bf16(v_h)
        tensors[ve + "cls_position_embedding"] = bf16(v_h)
        tensors["model.visual.backbone.ln_pre.weight"] = bf16(v_h)
        tensors["model.visual.backbone.ln_pre.bias"] = bf16(v_h)
        for layer in range(v_layers):
            vs = f"model.visual.backbone.layers.{layer}."
            for norm in ("ln_1", "ln_2"):
                tensors[vs + norm + ".weight"] = bf16(v_h)
                tensors[vs + norm + ".bias"] = bf16(v_h)
            tensors[vs + "ls_1.gamma"] = bf16(v_h)
            tensors[vs + "ls_2.gamma"] = bf16(v_h)
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                tensors[vs + f"self_attn.{proj}.weight"] = bf16(v_h, v_h)
                tensors[vs + f"self_attn.{proj}.bias"] = bf16(v_h)
            tensors[vs + "mlp.c_fc.weight"] = bf16(4 * v_h, v_h)
            tensors[vs + "mlp.c_fc.bias"] = bf16(4 * v_h)
            tensors[vs + "mlp.c_proj.weight"] = bf16(v_h, 4 * v_h)
            tensors[vs + "mlp.c_proj.bias"] = bf16(v_h)
        tensors["model.visual.proj_in.weight"] = bf16(v_proj, v_merge * v_merge * v_h)
        tensors["model.visual.proj_out.weight"] = bf16(h, v_proj)

    save_file(tensors, str(root / "model.safetensors"))

    if looped:
        mtp = {
            "fuse_weight": bf16(d_layers + 1, len(mtp_cfg["target_layer_ids"])),
            "mask_embed": bf16(h),
            "norm.weight": bf16(h),
            "markov_head.dflare_norm.weight": bf16(h),
            "markov_head.hidden_out.weight": bf16(h, d_rank),
            "markov_head.hidden_out.bias": bf16(h),
            "markov_head.markov_w1.weight": bf16(raw_vocab, d_rank),
            "markov_head.rnn_proj.weight": bf16(3 * d_rank, 2 * d_rank + 2 * h),
            "markov_head.rnn_proj.bias": bf16(3 * d_rank),
        }
        for layer in range(d_layers):
            ds = f"layers.{layer}."
            mtp[ds + "dflare_norm.weight"] = bf16(h)
            mtp[ds + "input_layernorm.weight"] = bf16(h)
            mtp[ds + "post_attention_layernorm.weight"] = bf16(h)
            mtp[ds + "self_attn.q_proj.weight"] = bf16(q_dim, h)
            mtp[ds + "self_attn.k_proj.weight"] = bf16(kv_dim, h)
            mtp[ds + "self_attn.v_proj.weight"] = bf16(kv_dim, h)
            mtp[ds + "self_attn.k_proj_ctx.weight"] = bf16(kv_dim, h)
            mtp[ds + "self_attn.v_proj_ctx.weight"] = bf16(kv_dim, h)
            mtp[ds + "self_attn.o_proj.weight"] = bf16(h, q_dim)
            mtp[ds + "self_attn.q_norm.weight"] = bf16(d)
            mtp[ds + "self_attn.k_norm.weight"] = bf16(d)
            mtp[ds + "self_attn.k_norm_ctx.weight"] = bf16(d)
            mtp[ds + "mlp.gate_proj.weight"] = bf16(d_inter, h)
            mtp[ds + "mlp.up_proj.weight"] = bf16(d_inter, h)
            mtp[ds + "mlp.down_proj.weight"] = bf16(h, d_inter)
        save_file(mtp, str(root / "mtp.safetensors"))
    return tensors


def test_inkling2_loop_full_components_conversion(tmp_path: Path):
    src_dir = tmp_path / "ckpt"
    src_tensors = _build_inkling2_fixture(src_dir, looped=True, expert_mode="mixed")
    out_path = tmp_path / "inkling2.ninfer"

    convert_main(
        [
            "--model",
            str(src_dir),
            "--recipe",
            "inkling2",
            "--components",
            "text,vision,dspark",
            "--out",
            str(out_path),
            "--device",
            "cpu",
            "--rows-per-chunk",
            "128",
        ]
    )

    with Artifact(out_path) as art, SafetensorsSource(src_dir / "mtp.safetensors") as mtp:
        text_cfg = art.directory.components["text"]["config"]
        assert text_cfg["architectures"] == ["Inkling2ForCausalLM"]
        assert text_cfg["model_type"] == "inkling2_text"
        assert text_cfg["unpadded_vocab_size"] == 130
        assert text_cfg["vocab_size"] == 192
        assert text_cfg["loop_config"]["num_passes"] == 2

        # Vocab truncation & zero padding on rows [130, 192)
        for param_name, src_key in (
            ("text/token_embedding", "model.llm.embed.weight"),
            ("text/output_head", "model.llm.unembed.weight"),
        ):
            got = _direct(art, param_name, (192, 128))
            assert torch.equal(got[:130], src_tensors[src_key][:130])
            assert torch.equal(got[130:], torch.zeros((62, 128), dtype=torch.bfloat16))

        # SWA packed 5-way parent vs Full packed 3-way parent
        swa_spans = [
            _parts(art, f"text/layers/0/attention/{r}")[0]
            for r in ("query", "relative", "output_gate", "key", "value")
        ]
        assert len({s[0] for s in swa_spans}) == 1
        assert all(a[2] == b[1] for a, b in zip(swa_spans, swa_spans[1:]))

        full_q_spans = [
            _parts(art, f"text/layers/1/attention/{r}")[0]
            for r in ("query", "relative", "output_gate")
        ]
        assert len({s[0] for s in full_q_spans}) == 1
        assert _parts(art, "text/layers/1/attention/key")[0][0] != full_q_spans[0][0]
        assert _parts(art, "text/layers/1/attention/value")[0][0] != full_q_spans[0][0]

        # AttnRes query shift (layer 0 absent, layer 1 Full [3, H], layer 2 SWA [H])
        assert "text/layers/0/attention/attn_res_query" not in art.directory.bindings
        assert torch.equal(
            _direct(art, "text/layers/1/attention/attn_res_query", (3, 128)),
            src_tensors["model.llm.attn_res.attn_queries.0"],
        )
        assert torch.equal(
            _direct(art, "text/layers/2/attention/attn_res_query", (128,)),
            src_tensors["model.llm.attn_res.attn_queries.1"],
        )
        # Stacked Full input norm [3, H]
        expected_full_norm = torch.stack(
            [
                src_tensors["model.llm.layers.1.q_input_norm.weight"],
                src_tensors["model.llm.layers.1.k_input_norm.weight"],
                src_tensors["model.llm.layers.1.v_input_norm.weight"],
            ],
            dim=0,
        )
        assert torch.equal(
            _direct(art, "text/layers/1/attention/input_norm", (3, 128)), expected_full_norm
        )

        # Rel logits transpose & sconv transposes
        assert torch.equal(
            _direct(art, "text/layers/0/attention/rel_logits_proj", (32, 16)),
            src_tensors["model.llm.layers.0.attn.rel_logits_proj.proj"].T.contiguous(),
        )
        assert torch.equal(
            _direct(art, "text/layers/0/attention/k_conv", (4, 64)),
            src_tensors["model.llm.layers.0.attn.k_sconv.weight"].squeeze(1).T.contiguous(),
        )
        assert torch.equal(
            _direct(art, "text/layers/1/attention/k_conv", (4, 64)),
            src_tensors["model.llm.layers.1.attn.k_sconv.static_w"],
        )
        assert torch.equal(
            _direct(art, "text/layers/0/attention/short_conv", (4, 128)),
            src_tensors["model.llm.layers.0.attn_sconv.weight"].squeeze(1).T.contiguous(),
        )

        # Loop core cross-pass message queries & controller stacking
        q_raw = src_tensors["model.llm.middle_cycle_cross_pass_queries"].reshape(2, 2, 128)
        kv_raw = src_tensors["model.llm.middle_cycle_cross_pass_kv_queries"].reshape(2, 2, 128)
        expected_msg = torch.stack((q_raw[0, 0], kv_raw[0, 0], kv_raw[0, 1], q_raw[0, 1]), dim=0)
        assert torch.equal(
            _direct(art, "text/layers/2/loop/message_queries", (4, 128)), expected_msg
        )
        expected_up = torch.stack(
            [
                torch.cat(
                    [
                        src_tensors[f"model.llm.layers.2.controllers.{st}.gain_weight"],
                        src_tensors[f"model.llm.layers.2.controllers.{st}.forget_weight"],
                    ],
                    dim=0,
                )
                for st in ("attn", "k", "v")
            ],
            dim=0,
        )
        assert torch.equal(
            _direct(art, "text/layers/2/loop/attn_kv_up", (3, 256, 16)), expected_up
        )

        # NVFP4_MUL layer 0 exact bit-exact de-interleave & multiplier check
        nv_obj = art.object(_parts(art, "text/layers/0/moe/experts/gate")[0][0])
        assert nv_obj.format == "nvfp4_mul"
        assert nv_obj.layout == "batched_block_scale_k16_m128x4_v1"
        assert nv_obj.shape == (4, 256, 128)
        nv_bytes = art.read_object(nv_obj.id)
        nv_geom = block_scale_geometry("nvfp4_mul", nv_obj.shape)
        w13_src_codes = src_tensors["model.llm.layers.0.mlp.experts.w13_weight"]
        w13_src_scales = src_tensors["model.llm.layers.0.mlp.experts.w13_weight_scale"]
        w13_amax = src_tensors["model.llm.layers.0.mlp.experts.w13_weight_amax"]
        w13_denom = src_tensors["model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"]
        nv_codes, nv_scales, nv_scalar = decode_nvfp4_words(
            nv_bytes, nv_obj.shape, nv_obj.format
        )
        for e in range(4):
            mult = float(np.float32(w13_amax[e].item()) / np.float32(w13_denom.item()))
            assert nv_scalar[e].item() == mult
            assert torch.equal(nv_codes[e, :128], w13_src_codes[e, 0::2])
            assert torch.equal(nv_codes[e, 128:], w13_src_codes[e, 1::2])
            assert torch.equal(nv_scales[e, :128], w13_src_scales[e, 0::2].view(torch.uint8))
            assert torch.equal(nv_scales[e, 128:], w13_src_scales[e, 1::2].view(torch.uint8))
            for half, row_off in ((0, 0), (1, 128)):
                c = nv_codes[e, row_off : row_off + 128]
                s = nv_scales[e, row_off : row_off + 128].view(torch.float8_e4m3fn)
                nib = torch.stack((c & 15, c >> 4), dim=-1).reshape(128, 128).long()
                got = (_E2M1[nib] * s.double().repeat_interleave(16, dim=-1) * nv_scalar[e].item()).float()
                src_c = w13_src_codes[e, half::2]
                src_s = w13_src_scales[e, half::2]
                src_nib = torch.stack((src_c & 15, src_c >> 4), dim=-1).reshape(128, 128).long()
                ref = (_E2M1[src_nib] * src_s.double().repeat_interleave(16, dim=-1) * mult).float()
                assert torch.equal(got, ref)

        # MXFP8 layer 4 exact bit-exact de-interleave check
        mx_obj = art.object(_parts(art, "text/layers/4/moe/experts/gate")[0][0])
        assert mx_obj.format == "mxfp8"
        assert mx_obj.layout == "row_group_scale_v1"
        assert mx_obj.shape == (4, 256, 128)
        mx_bytes = art.read_object(mx_obj.id)
        mx_geom = row_group_scale_geometry("mxfp8", mx_obj.shape)
        mx_codes = torch.frombuffer(
            bytearray(mx_bytes[: mx_geom.code_plane_bytes]), dtype=torch.uint8
        ).reshape(4, 256, 128)
        mx_scales = torch.frombuffer(
            bytearray(
                mx_bytes[
                    mx_geom.scale_plane_offset : mx_geom.scale_plane_offset
                    + mx_geom.scale_plane_bytes
                ]
            ),
            dtype=torch.uint8,
        ).reshape(4, 256, 4)
        src_mx_codes = src_tensors["model.llm.layers.4.mlp.experts.w13_weight"].view(torch.uint8)
        src_mx_scales = src_tensors["model.llm.layers.4.mlp.experts.w13_weight_scale"]
        for e in range(4):
            assert torch.equal(mx_codes[e, :128], src_mx_codes[e, 0::2])
            assert torch.equal(mx_codes[e, 128:], src_mx_codes[e, 1::2])
            assert torch.equal(mx_scales[e, :128], src_mx_scales[e, 0::2])
            assert torch.equal(mx_scales[e, 128:], src_mx_scales[e, 1::2])

        # Per-visit Uses on core vs non-core layers
        uses = {(u["parameter"], u["input"]): u["activation_policy"] for u in art.directory.uses}
        assert uses[("text/layers/0/moe/experts/gate", "text/layers/0/mlp_in_p0")] == "A16Only"
        assert ("text/layers/0/moe/experts/gate", "text/layers/0/mlp_in_p1") not in uses
        assert uses[("text/layers/2/moe/experts/gate", "text/layers/2/mlp_in_p0")] == "A16Only"
        assert uses[("text/layers/2/moe/experts/gate", "text/layers/2/mlp_in_p1")] == "A16Only"
        assert uses[("text/layers/4/moe/experts/gate", "text/layers/4/mlp_in_p0")] == "AllowA8"
        assert uses[("text/layers/4/moe/experts/down", "text/layers/4/mlp_act_p0")] == "AllowA8"

        # Vision QKV packing
        vis_qkv = [
            _parts(art, f"vision/layers/0/attention/{r}")[0] for r in ("query", "key", "value")
        ]
        assert len({s[0] for s in vis_qkv}) == 1
        assert art.object(vis_qkv[0][0]).shape == (192, 64)

        # DSpark shared parameters, fuse_alpha, rnn split, and FP64 fold_weight/fold_bias
        for role in ("token_embedding", "embed_norm", "output_head"):
            assert art.directory.bindings["dspark/" + role] == art.directory.bindings["text/" + role]

        expected_alpha = torch.softmax(mtp.read_flat("fuse_weight").reshape(3, 3).float(), dim=-1)
        assert torch.allclose(_direct(art, "dspark/fuse_alpha", (3, 3)), expected_alpha)

        rnn_w = mtp.read_flat("markov_head.rnn_proj.weight").reshape(96, 64 + 256)
        assert torch.equal(_direct(art, "dspark/markov/w_recurrent", (96, 64)), rnn_w[:, :64])
        assert torch.equal(_direct(art, "dspark/markov/w_hidden", (96, 128)), rnn_w[:, 64:192])
        assert torch.equal(_direct(art, "dspark/markov/w_frnn", (96, 128)), rnn_w[:, 192:320])
        assert torch.equal(
            _direct(art, "dspark/markov/rnn_bias", (96,)),
            mtp.read_flat("markov_head.rnn_proj.bias").float(),
        )

        unembed_d = src_tensors["model.llm.unembed.weight"][:130].double()
        hout_w = mtp.read_flat("markov_head.hidden_out.weight").reshape(128, 32).double()
        hout_b = mtp.read_flat("markov_head.hidden_out.bias").reshape(128).double()
        got_fw = _direct(art, "dspark/markov/fold_weight", (192, 32))
        got_fb = _direct(art, "dspark/markov/fold_bias", (192,))
        assert torch.equal(got_fw[:130], (unembed_d @ hout_w).to(torch.bfloat16))
        assert torch.equal(got_fw[130:], torch.zeros((62, 32), dtype=torch.bfloat16))
        assert torch.equal(got_fb[:130], (unembed_d @ hout_b).to(torch.bfloat16))
        assert torch.equal(got_fb[130:], torch.zeros((62,), dtype=torch.bfloat16))


def test_inkling2_non_loop_bf16_text_only(tmp_path: Path):
    src_dir = tmp_path / "ckpt_bf16"
    src_tensors = _build_inkling2_fixture(src_dir, looped=False, expert_mode="bf16")
    out_path = tmp_path / "inkling2_bf16.ninfer"

    convert_main(
        [
            "--model",
            str(src_dir),
            "--recipe",
            "inkling2",
            "--components",
            "text",
            "--out",
            str(out_path),
            "--device",
            "cpu",
        ]
    )
    with Artifact(out_path) as art:
        cfg = art.directory.components["text"]["config"]
        assert "loop_config" not in cfg
        assert set(art.directory.components) == {"text"}
        gate = _direct(art, "text/layers/0/moe/experts/gate", (4, 128, 128))
        up = _direct(art, "text/layers/0/moe/experts/up", (4, 128, 128))
        raw = src_tensors["model.llm.layers.0.mlp.experts.w13_weight"]
        assert torch.equal(gate, raw[:, 0::2, :])
        assert torch.equal(up, raw[:, 1::2, :])
