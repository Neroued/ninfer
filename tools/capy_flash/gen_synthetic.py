# Copied from sybrix/qwen3.8-flash-next-ninfer @a2fd69bb, flash-next-reference/gen_synthetic.py
# Copyright sybrix, Apache License 2.0 (see the repository LICENSE / NOTICE). capy-flash adds the
# fp4_local variant and --ffn dense; the CSF reference encoder is from vLLM PR #60646 (Apache-2.0).
"""Write a small random Qwen3.8-Flash-Next checkpoint in an NVIDIA ModelOpt layout.

Real per-layer geometry (hidden 2560, 512 NVFP4 experts, GDN 16/48 heads, 24/2 attention heads,
4 hyper-connection streams, PLE at block 1) with only `--layers` Text blocks and a tiny n-gram
table, so the converter, NInfer and the layer-streaming reference can be compared end to end
without the 133 GB checkpoint. Tokenizer and template files are copied from a real snapshot.

--variant nvfp4 (default) follows Qwen3.8-Flash-Next-NVFP4: BF16 dense, NVFP4 routed experts, an
FP8 n-gram table, FP8_PB_WO MTP experts. --variant fp4_local follows Qwen3.8-Flash-Next-FP4-Local
(pass that snapshot as --source): MXFP8 dense Text matrices (E4M3FN weight + UE8M0 weight_scale
per 32 columns), NVFP4 routed experts with CSF-compressed scales, a row-scaled NVFP4 n-gram table
(80 code bytes + 10 E4M3FN scales per row, one FP32 weight_scale_2), W4A16 NVFP4 MTP experts,
F32 Text hyper-connection and PLE norms, BF16 elsewhere (embedding, head, router, MTP dense part).

--ffn dense (capy-flash) writes a dense Qwen4 checkpoint instead: every
tensor BF16 (no quantization_config), a SwiGLU MLP (mlp.{gate,up,down}_proj, --intermediate) in every
Text block and in the MTP layer, no MoE field in the config. --hidden / --kv-heads / --hc-lowrank
change the geometry (hidden 5120: --hidden 5120 --kv-heads 4 --intermediate 14336, the q4d-5120
preset of tools/ref_q4/synthetic.sh), --no-ple drops the n-gram embedding (empty ple_layer_ids), --output-gate sets output_gate_type.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


def splitmix64(v):
    v = (v + GAMMA) & MASK64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK64
    return (v ^ (v >> 31)) & MASK64


def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def ple_constants(t, ple_index):
    vocab, order, per = t["vocab_size"], t["ngram_size"], t["heads_per_ngram"]
    bound = max(1, ((1 << 63) - 1) // vocab // 2)
    seed = t.get("seed", 1234) + 10007 * ple_index  # the FP4-Local config omits the default
    mult = [2 * (splitmix64((seed + GAMMA * (i + 1)) & MASK64) % bound) + 1 for i in range(order)]
    heads = (order - 1) * per
    prime, sizes, offsets, total = t["ngram_vocab_size_base"] - 1, [], [], 0
    for _ in range(ple_index * heads):
        prime += 1
        while not is_prime(prime):
            prime += 1
    for _ in range(heads):
        prime += 1
        while not is_prime(prime):
            prime += 1
        sizes.append(prime)
        offsets.append(total)
        total += prime
    div = t["make_ngram_vocab_size_divisible_by"]
    return mult, sizes, offsets, -(-total // div) * div


def bf16(shape, std, gen):
    return (torch.randn(shape, generator=gen) * std).to(torch.bfloat16)


def nvfp4(rows, cols, gen, rng):
    packed = torch.from_numpy(rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8))
    # E4M3 block scales near 1.0 (exponent 7): words 0x34..0x3C.
    scale = torch.from_numpy(rng.integers(0x34, 0x3C, size=(rows, cols // 16), dtype=np.uint8))
    return {
        "weight": packed,
        "weight_scale": scale.view(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.02 / 3.0 * float(rng.uniform(0.5, 1.5)), dtype=torch.float32),
        "input_scale": torch.tensor(float(rng.uniform(0.01, 0.1)), dtype=torch.float32),
    }


def nvfp4_w4a16(rows, cols, gen, rng):
    """NVFP4 weights only (W4A16): no activation input_scale."""
    out = nvfp4(rows, cols, gen, rng)
    del out["input_scale"]
    return out


def mxfp8(rows, cols, std, gen, rng):
    """E4M3FN codes of about N(0, 64^2) and UE8M0 scales near std / 64, one per 32 columns."""
    codes = (torch.randn(rows, cols, generator=gen) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    exponent = 127 + round(math.log2(std / 64))
    scales = rng.integers(exponent - 1, exponent + 2, size=(rows, cols // 32), dtype=np.uint8)
    return {"weight": codes, "weight_scale": torch.from_numpy(scales)}


def csf_encode(scales):
    """CSF (byte-window4-fixed-stream-u24-exceptions/1) reference encoder, vLLM PR #60646 test.

    Row base = row minimum (at most 240), 4-bit codes base + code, outliers as sorted U32 words
    (scale byte << 24 | row * cols + col).
    """
    raw = scales.view(torch.uint8).to(torch.int64)
    rows, cols = raw.shape
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    codes = raw - bases
    outlier = (codes < 0) | (codes > 15)
    codes = torch.where(outlier, 0, codes)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat((bases.to(torch.uint8).reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), dim=1)
    positions = outlier.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return fixed, words.to(torch.int32).view(torch.uint32)


def nvfp4_csf(rows, cols, gen, rng):
    """NVFP4 with CSF scales: a few scales (3.0, word 0x44) leave the 16-word window."""
    out = nvfp4(rows, cols, gen, rng)
    scale = out.pop("weight_scale").view(torch.uint8).clone()
    flat = scale.view(-1)
    flat[torch.from_numpy(rng.choice(flat.numel(), size=8, replace=False))] = 0x44
    fixed, exceptions = csf_encode(scale)
    out["weight_scale.nvfp4_csf_fixed"] = fixed
    out["weight_scale.nvfp4_csf_exceptions"] = exceptions
    return out


def quantization_config(real, layers, mtp):
    """The real ModelOpt quantization_config restricted to the generated modules."""
    kept = {}
    for name, value in real["quantized_layers"].items():
        if name.startswith("model.language_model.layers."):
            if int(name.split(".")[3]) >= layers:
                continue
        elif not (mtp and name.startswith("mtp.")):
            continue
        kept[name] = value
    return {**real, "quantized_layers": kept}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="real snapshot for config and tokenizer files")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ngram-base", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lm-head-std", type=float, default=0.02)
    p.add_argument("--mtp", action="store_true", help="add the MTP head (FP8_PB_WO experts)")
    p.add_argument("--vision", action="store_true", help="add the (Qwen3.5-shaped) vision tower")
    p.add_argument("--variant", choices=("nvfp4", "fp4_local"), default="nvfp4")
    p.add_argument("--ffn", choices=("moe", "dense"), default="moe", help="dense: BF16 dense Qwen4 checkpoint")
    p.add_argument("--intermediate", type=int, default=6912, help="dense FFN width (--ffn dense)")
    p.add_argument("--hidden", type=int, help="hidden_size (default: the source config's)")
    p.add_argument("--kv-heads", type=int, help="num_key_value_heads (default: the source config's)")
    p.add_argument("--hc-lowrank", type=int, help="hc_lowrank (default: the source config's)")
    p.add_argument("--no-ple", action="store_true", help="no n-gram embedding (empty ple_layer_ids)")
    p.add_argument("--output-gate", choices=("sigmoid", "silu"), help="output_gate_type (default: the source's)")
    args = p.parse_args()
    fp4_local = args.variant == "fp4_local"
    dense_ffn = args.ffn == "dense"
    if dense_ffn and (fp4_local or args.vision):
        p.error("--ffn dense writes a BF16 text+MTP checkpoint (no fp4_local variant, no vision)")
    if not dense_ffn and (args.hidden or args.kv_heads or args.hc_lowrank or args.no_ple or args.output_gate):
        p.error("geometry, PLE and output-gate options apply to --ffn dense only")
    if fp4_local and args.vision:
        p.error("the fp4_local variant has no vision tower (served as text,mtp)")
    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    config = json.loads((src / "config.json").read_text())
    real_quantization = config.pop("quantization_config", None)
    if fp4_local:
        if not real_quantization or real_quantization.get("quant_algo") != "MIXED_PRECISION":
            p.error("--variant fp4_local needs the Qwen3.8-Flash-Next-FP4-Local snapshot as --source")
        config["quantization_config"] = quantization_config(real_quantization, args.layers, args.mtp)
    t = config["text_config"]
    n = args.layers
    t["num_hidden_layers"] = n
    t["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]
    t["ngram_vocab_size_base"] = args.ngram_base
    t["mtp_num_hidden_layers"] = 1 if args.mtp else 0
    if dense_ffn:
        config.pop("quantization_config", None)
        config.pop("vision_config", None)
        for key in (
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "norm_topk_prob",
            "router_aux_loss_coef",
            "output_router_logits",
        ):
            t.pop(key, None)
        t["intermediate_size"] = args.intermediate
        for key, value in (
            ("hidden_size", args.hidden),
            ("num_key_value_heads", args.kv_heads),
            ("hc_lowrank", args.hc_lowrank),
            ("output_gate_type", args.output_gate),
        ):
            if value:
                t[key] = value
        if args.no_ple:
            t["ple_layer_ids"] = []
    (out / "config.json").write_text(json.dumps(config, indent=2))

    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    H, S, R = t["hidden_size"], t["hc_count"], t["hc_lowrank"]
    V = t["vocab_size"]
    if not dense_ffn:
        E, I = t["num_experts"], t["moe_intermediate_size"]
        Is = t["shared_expert_intermediate_size"]
    pre = "model.language_model."
    tensors = {}
    tensors[pre + "embed_tokens.weight"] = bf16((V, H), 0.02, gen)
    tensors["lm_head.weight"] = bf16((V, H), args.lm_head_std, gen)

    def norm_weight(shape, std, f32):
        value = bf16(shape, std, gen)
        return value.float() if f32 else value

    def dense(name, shape, std=0.02, quantized=False):
        if not quantized:
            tensors[name + ".weight"] = bf16(shape, std, gen)
            return
        for key, value in mxfp8(*shape, std, gen, rng).items():
            tensors[f"{name}.{key}"] = value

    # FP4-Local: MXFP8 Text matrices and F32 Text norms; the MTP layer stays BF16.
    def gated(prefix, inject=True, f32=False):
        tensors[prefix + "hc_norm.weight"] = norm_weight((S * H,), 0.05, f32)
        tensors[prefix + "input_mix_weight_down.weight"] = bf16((R, S * H), 0.02, gen)
        tensors[prefix + "input_mix_weight_up.weight"] = bf16((S * H, R), 0.02, gen)
        if inject:
            tensors[prefix + "block_inject_weight.weight"] = bf16((S, S * H), 0.02, gen)

    gated(pre + "hyper_connection_mixer.", inject=False, f32=fp4_local)
    nk, kd = t["linear_num_key_heads"], t["linear_key_head_dim"]
    nv, vd = t["linear_num_value_heads"], t["linear_value_head_dim"]
    qh, kvh, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    ih, ihd = t["indexer_n_heads"], t["indexer_head_dim"]
    ple_blocks = [b - 1 for b in t.get("ple_layer_ids") or []]

    def attention(a, quantized=False):
        dense(a + "q_proj", (2 * qh * hd, H), quantized=quantized)
        dense(a + "k_proj", (kvh * hd, H), quantized=quantized)
        dense(a + "v_proj", (kvh * hd, H), quantized=quantized)
        dense(a + "o_proj", (H, qh * hd), quantized=quantized)
        tensors[a + "q_norm.weight"] = bf16((hd,), 0.05, gen)
        tensors[a + "k_norm.weight"] = bf16((hd,), 0.05, gen)
        dense(a + "indexer.index_qk_proj", ((ih + 1) * ihd, H), quantized=quantized)
        tensors[a + "indexer.q_layernorm.weight"] = bf16((ihd,), 0.05, gen)
        tensors[a + "indexer.k_layernorm.weight"] = bf16((ihd,), 0.05, gen)

    def mlp(m, fp8=False, quantized=False, experts=nvfp4):
        if dense_ffn:
            dense(m + "gate_proj", (args.intermediate, H))
            dense(m + "up_proj", (args.intermediate, H))
            dense(m + "down_proj", (H, args.intermediate))
            return
        tensors[m + "gate.weight"] = bf16((E, H), 0.05, gen)
        dense(m + "shared_expert.gate_proj", (Is, H), quantized=quantized)
        dense(m + "shared_expert.up_proj", (Is, H), quantized=quantized)
        dense(m + "shared_expert.down_proj", (H, Is), quantized=quantized)
        tensors[m + "shared_expert_gate.weight"] = bf16((1, H), 0.02, gen)
        for e in range(E):
            for proj, (rows, cols) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                if fp8:
                    # FP8_PB_WO: E4M3 weight, one BF16 multiplier per 128x128 tile.
                    scale = torch.empty(-(-rows // 128), -(-cols // 128)).uniform_(2e-4, 6e-4, generator=gen)
                    w = torch.randn(rows, cols, generator=gen).clamp(-4, 4) * 30
                    tensors[f"{m}experts.{e}.{proj}.weight"] = w.to(torch.float8_e4m3fn)
                    tensors[f"{m}experts.{e}.{proj}.weight_scale_inv"] = scale.to(torch.bfloat16)
                    continue
                for key, value in experts(rows, cols, gen, rng).items():
                    tensors[f"{m}experts.{e}.{proj}.{key}"] = value

    shards = []
    for i in range(n):
        lp = f"{pre}layers.{i}."
        gated(lp + "attn_hyper_connection.", f32=fp4_local)
        gated(lp + "mlp_hyper_connection.", f32=fp4_local)
        if t["layer_types"][i] == "linear_attention":
            g = lp + "linear_attn."
            conv = 2 * nk * kd + nv * vd
            tensors[g + "A_log"] = torch.empty(nv).uniform_(1, 16, generator=gen).log().to(torch.bfloat16)
            tensors[g + "dt_bias"] = torch.ones(nv, dtype=torch.bfloat16)
            tensors[g + "conv1d.weight"] = bf16((conv, 1, 4), 0.2, gen)
            dense(g + "in_proj_a", (nv, H), quantized=fp4_local)
            dense(g + "in_proj_b", (nv, H), quantized=fp4_local)
            dense(g + "in_proj_qkv", (conv, H), quantized=fp4_local)
            dense(g + "in_proj_z", (nv * vd, H), quantized=fp4_local)
            tensors[g + "norm.weight"] = (1.0 + torch.randn(vd, generator=gen) * 0.05).to(torch.bfloat16)
            dense(g + "out_proj", (H, nv * vd), quantized=fp4_local)
        else:
            attention(lp + "self_attn.", quantized=fp4_local)
        mlp(lp + "mlp.", quantized=fp4_local, experts=nvfp4_csf if fp4_local else nvfp4)
        if i in ple_blocks:
            index = ple_blocks.index(i)
            mult, sizes, offsets, rows = ple_constants(t, index)
            q = lp + "ple."
            width = t["ple_embed_dim"] // ((t["ngram_size"] - 1) * t["heads_per_ngram"])
            tensors[q + "conv1d.weight"] = bf16((S * H, 1, t["ple_conv_kernel_size"]), 0.2, gen)
            tensors[q + "key_proj.weight"] = bf16((S * H, t["ple_embed_dim"]), 0.02, gen)
            tensors[q + "value_proj.weight"] = bf16((H, t["ple_embed_dim"]), 0.02, gen)
            for name in ("norm_conv", "norm_key", "norm_query"):
                tensors[q + name + ".weight"] = norm_weight((S * H,), 0.05, fp4_local)
            emb = q + "ple_embedding."
            tensors[emb + "layer_multipliers"] = torch.tensor(mult, dtype=torch.int64)
            tensors[emb + "ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
            tensors[emb + "ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.int64)
            parts = 128
            assert rows % parts == 0, rows
            chunk = rows // parts
            if fp4_local:
                # Row-scaled NVFP4: E2M1 codes, E4M3FN scales 0.5..2 per 16 values, and one FP32
                # multiplier that is not a power of two (exercises the single final rounding).
                codes = torch.from_numpy(rng.integers(0, 256, size=(rows, width // 2), dtype=np.uint8))
                scales = torch.from_numpy(rng.integers(0x30, 0x41, size=(rows, width // 16), dtype=np.uint8))
                tensors[emb + "ngram_embedding.weight_scale_2"] = torch.tensor([0.0071234], dtype=torch.float32)
                for s in range(parts):
                    rows_s = slice(s * chunk, (s + 1) * chunk)
                    tensors[f"{emb}ngram_embedding.shard_{s}.weight"] = codes[rows_s].clone()
                    tensors[f"{emb}ngram_embedding.shard_{s}.weight_scale"] = (
                        scales[rows_s].clone().view(torch.float8_e4m3fn)
                    )
            else:
                tensors[emb + "ngram_embedding.weight_scale"] = torch.tensor([0.015625], dtype=torch.bfloat16)
                table = torch.randn(rows, width, generator=gen).clamp(-4, 4).to(torch.float8_e4m3fn)
                for s in range(parts):
                    tensors[f"{emb}ngram_embedding.shard_{s}.weight"] = table[s * chunk : (s + 1) * chunk].clone()
        # Spill a file per layer to bound host memory.
        if i != n - 1:
            shards.append(dict(tensors))
            tensors = {}
    if args.vision:
        shards.append(tensors)
        tensors = {}
        vc = config["vision_config"]
        vh, vi, depth = vc["hidden_size"], vc["intermediate_size"], vc["depth"]
        ps, pt, merge = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
        out_h, mh = vc["out_hidden_size"], vh * merge * merge
        v = "model.visual."
        tensors[v + "patch_embed.proj.weight"] = bf16((vh, 3, pt, ps, ps), 0.02, gen)
        tensors[v + "patch_embed.proj.bias"] = bf16((vh,), 0.02, gen)
        tensors[v + "pos_embed.weight"] = bf16((vc["num_position_embeddings"], vh), 0.02, gen)
        for i in range(depth):
            b = f"{v}blocks.{i}."
            for norm in ("norm1", "norm2"):
                tensors[b + norm + ".weight"] = (1.0 + torch.randn(vh, generator=gen) * 0.05).to(torch.bfloat16)
                tensors[b + norm + ".bias"] = bf16((vh,), 0.02, gen)
            tensors[b + "attn.qkv.weight"] = bf16((3 * vh, vh), 0.02, gen)
            tensors[b + "attn.qkv.bias"] = bf16((3 * vh,), 0.02, gen)
            tensors[b + "attn.proj.weight"] = bf16((vh, vh), 0.02, gen)
            tensors[b + "attn.proj.bias"] = bf16((vh,), 0.02, gen)
            tensors[b + "mlp.linear_fc1.weight"] = bf16((vi, vh), 0.02, gen)
            tensors[b + "mlp.linear_fc1.bias"] = bf16((vi,), 0.02, gen)
            tensors[b + "mlp.linear_fc2.weight"] = bf16((vh, vi), 0.02, gen)
            tensors[b + "mlp.linear_fc2.bias"] = bf16((vh,), 0.02, gen)
        tensors[v + "merger.norm.weight"] = (1.0 + torch.randn(vh, generator=gen) * 0.05).to(torch.bfloat16)
        tensors[v + "merger.norm.bias"] = bf16((vh,), 0.02, gen)
        tensors[v + "merger.linear_fc1.weight"] = bf16((mh, mh), 0.02, gen)
        tensors[v + "merger.linear_fc1.bias"] = bf16((mh,), 0.02, gen)
        tensors[v + "merger.linear_fc2.weight"] = bf16((out_h, mh), 0.02, gen)
        tensors[v + "merger.linear_fc2.bias"] = bf16((out_h,), 0.02, gen)
    if args.mtp:
        shards.append(tensors)
        tensors = {}
        tensors["mtp.pre_fc_norm_embedding.weight"] = bf16((H,), 0.05, gen)
        tensors["mtp.pre_fc_norm_hidden.weight"] = bf16((S * H,), 0.05, gen)
        tensors["mtp.fc_embedding.weight"] = bf16((H, H), 0.02, gen)
        tensors["mtp.fc_hidden.weight"] = bf16((H, H), 0.02, gen)
        gated("mtp.hyper_connection_mixer.", inject=False)
        gated("mtp.layers.0.attn_hyper_connection.")
        gated("mtp.layers.0.mlp_hyper_connection.")
        attention("mtp.layers.0.self_attn.")
        if fp4_local:
            mlp("mtp.layers.0.mlp.", experts=nvfp4_w4a16)
        else:
            mlp("mtp.layers.0.mlp.", fp8=True)
    shards.append(tensors)
    weight_map = {}
    for k, shard in enumerate(shards):
        name = f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({key: v.contiguous() for key, v in shard.items()}, str(out / name))
        weight_map.update({key: name for key in shard})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    print(f"wrote {len(weight_map)} tensors in {len(shards)} files to {out}")


if __name__ == "__main__":
    main()
