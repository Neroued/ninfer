"""A tiny random Qwen4Exp (Qwen3.8-Flash-Next) checkpoint in the NVIDIA ModelOpt NVFP4 layout.

Same tensor names, dtypes and per-role layouts as ``nvidia/Qwen3.8-Flash-Next-NVFP4`` (NVFP4 routed
experts with ``weight_scale``/``weight_scale_2``/``input_scale``, block-FP8 MTP experts with
``weight_scale_inv``, an FP8 n-gram table in row shards with one BF16 multiplier, BF16 everything
else, no final norm), at dimensions small enough for unit tests yet valid for every Flash-Next recipe:
each dense projection group and the output head have N % 128 == 0 and K % 64 == 0, so the
NVFP4-dense recipe applies too. Adapted from sybrix's flash-next-reference/gen_synthetic.py.
``dense=True`` writes a dense Qwen4 checkpoint instead (BF16 SwiGLU MLP of width DENSE_IR in every
block and in the MTP layer, no MoE field), ``ple=False`` one without the n-gram embedding (empty
ple_layer_ids).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import torch
from safetensors.torch import save_file

from tools.convert.quantization.nvfp4 import quantize_nvfp4, weight_divisor

H, IR, E, VOCAB, SHARED = 128, 128, 4, 384, 128
HC, LOWRANK = 2, 8
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 64
GDN_K_HEADS, GDN_V_HEADS, GDN_DIM = 1, 2, 64
INDEXER_HEADS, INDEXER_DIM = 2, 32
PLE_DIM, NGRAM, HEADS_PER_NGRAM, NGRAM_BASE, NGRAM_DIVISOR = 32, 3, 2, 11, 8
SEED, EOS = 1234, 256  # EOS = <|endoftext|>, the first added token
LAYER_TYPES = ["linear_attention", "full_attention"]
PLE_LAYER_IDS = [1]  # one-indexed: block 0, a GDN block
DENSE_IR = 256


def config(*, dense: bool = False, ple: bool = True) -> dict:
    cfg = _moe_config()
    text = cfg["text_config"]
    if dense:
        for key in (
            "num_experts",
            "num_experts_per_tok",
            "norm_topk_prob",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ):
            del text[key]
        text["intermediate_size"] = DENSE_IR
    if not ple:
        text["ple_layer_ids"] = []
    return cfg


def _moe_config() -> dict:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": H,
            "vocab_size": VOCAB,
            "num_hidden_layers": len(LAYER_TYPES),
            "max_position_embeddings": 4096,
            "layer_types": list(LAYER_TYPES),
            "full_attention_interval": 2,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": GDN_K_HEADS,
            "linear_key_head_dim": GDN_DIM,
            "linear_num_value_heads": GDN_V_HEADS,
            "linear_value_head_dim": GDN_DIM,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "hidden_act": "silu",
            "output_gate_type": "sigmoid",
            "num_experts": E,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "moe_intermediate_size": IR,
            "shared_expert_intermediate_size": SHARED,
            "hc_count": HC,
            "hc_lowrank": LOWRANK,
            "indexer_n_heads": INDEXER_HEADS,
            "indexer_kv_heads": 1,
            "indexer_head_dim": INDEXER_DIM,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": list(PLE_LAYER_IDS),
            "ple_embed_dim": PLE_DIM,
            "ple_conv_kernel_size": 4,
            "ngram_size": NGRAM,
            "heads_per_ngram": HEADS_PER_NGRAM,
            "ngram_vocab_size_base": NGRAM_BASE,
            "make_ngram_vocab_size_divisible_by": NGRAM_DIVISOR,
            "split_ngram_parts": 2,
            "seed": SEED,
            "bos_token_id": EOS,
            "eos_token_id": EOS,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False,
            "rms_norm_eps": 1e-6,
        },
        "tie_word_embeddings": False,
    }


# --- PLE hash constants, written independently of the converter and the C++ loader ----------
# (the checkpoint stores the same values as ple_embedding.{layer_multipliers,ngram_heads_*}).

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    return v >= 2 and (v == 2 or (v % 2 == 1 and all(v % d for d in range(3, math.isqrt(v) + 1, 2))))


def ple_constants(text: dict) -> dict:
    """Hash multipliers, per-head moduli and row offsets, and the padded table rows."""
    vocab, order, per = text["vocab_size"], text["ngram_size"], text["heads_per_ngram"]
    bound = max(1, ((1 << 63) - 1) // vocab // 2)
    seed = text["seed"]  # the single PLE layer has index 0
    multipliers = [2 * (_splitmix64((seed + _GAMMA * (i + 1)) & _MASK64) % bound) + 1 for i in range(order)]
    prime, moduli, offsets, total = text["ngram_vocab_size_base"] - 1, [], [], 0
    for _ in range((order - 1) * per):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        moduli.append(prime)
        offsets.append(total)
        total += prime
    divisor = text["make_ngram_vocab_size_divisible_by"]
    return {
        "multipliers": multipliers,
        "moduli": moduli,
        "offsets": offsets,
        "rows": -(-total // divisor) * divisor,
    }


# --- tensors ----------------------------------------------------------------------------------


def _nvfp4_expert(tensors: dict, prefix: str, shape, generator: torch.Generator) -> None:
    values = torch.randn(shape, generator=generator)
    divisor = weight_divisor(float(values.abs().max()), (6.0,))
    words = quantize_nvfp4(values, divisor)
    tensors[prefix + ".weight"] = words.codes
    tensors[prefix + ".weight_scale"] = words.scales.view(torch.float8_e4m3fn)
    # ModelOpt stores multipliers: the v3 divisor is the FP32 reciprocal of weight_scale_2.
    tensors[prefix + ".weight_scale_2"] = torch.tensor(1.0 / divisor, dtype=torch.float32)
    tensors[prefix + ".input_scale"] = torch.tensor(
        float(torch.rand((), generator=generator)) * 0.09 + 0.01, dtype=torch.float32
    )


def _block_fp8(tensors: dict, prefix: str, shape, generator: torch.Generator) -> None:
    values = torch.randn(shape, generator=generator).clamp(-4, 4) * 30
    tensors[prefix + ".weight"] = values.to(torch.float8_e4m3fn)
    tiles = (-(-shape[0] // 128), -(-shape[1] // 128))
    tensors[prefix + ".weight_scale_inv"] = (torch.rand(tiles, generator=generator) * 4e-4 + 2e-4).to(torch.bfloat16)


def write_checkpoint(path: Path, *, seed: int = 0, dense: bool = False, ple: bool = True) -> dict:
    """Write config, safetensors (two shards + index) and tokenizer files; return the tensors."""
    cfg = config(dense=dense, ple=ple)
    text = cfg["text_config"]
    g = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    wide = HC * H

    def bf16(name, shape, std=0.05):
        tensors[name] = (torch.randn(shape, generator=g) * std).to(torch.bfloat16)

    def mixer(prefix, inject=True):
        bf16(prefix + "hc_norm.weight", (wide,))
        bf16(prefix + "input_mix_weight_down.weight", (LOWRANK, wide))
        bf16(prefix + "input_mix_weight_up.weight", (wide, LOWRANK))
        if inject:
            bf16(prefix + "block_inject_weight.weight", (HC, wide))

    def moe(prefix, expert):
        if dense:
            for role, shape in (("gate", (DENSE_IR, H)), ("up", (DENSE_IR, H)), ("down", (H, DENSE_IR))):
                bf16(prefix + f"mlp.{role}_proj.weight", shape)
            return
        bf16(prefix + "mlp.gate.weight", (E, H))
        bf16(prefix + "mlp.shared_expert_gate.weight", (1, H))
        for role, shape in (("gate", (SHARED, H)), ("up", (SHARED, H)), ("down", (H, SHARED))):
            bf16(prefix + f"mlp.shared_expert.{role}_proj.weight", shape)
        for e in range(E):
            for role, shape in (("gate", (IR, H)), ("up", (IR, H)), ("down", (H, IR))):
                expert(tensors, f"{prefix}mlp.experts.{e}.{role}_proj", shape, g)

    def attention(prefix):
        a = prefix + "self_attn."
        bf16(a + "q_proj.weight", (2 * HEADS * HEAD_DIM, H))
        bf16(a + "k_proj.weight", (KV_HEADS * HEAD_DIM, H))
        bf16(a + "v_proj.weight", (KV_HEADS * HEAD_DIM, H))
        bf16(a + "o_proj.weight", (H, HEADS * HEAD_DIM))
        bf16(a + "q_norm.weight", (HEAD_DIM,))
        bf16(a + "k_norm.weight", (HEAD_DIM,))
        bf16(a + "indexer.index_qk_proj.weight", ((INDEXER_HEADS + 1) * INDEXER_DIM, H))
        bf16(a + "indexer.q_layernorm.weight", (INDEXER_DIM,))
        bf16(a + "indexer.k_layernorm.weight", (INDEXER_DIM,))

    def gdn(prefix):
        q = prefix + "linear_attn."
        conv = 2 * GDN_K_HEADS * GDN_DIM + GDN_V_HEADS * GDN_DIM
        bf16(q + "in_proj_qkv.weight", (conv, H))
        bf16(q + "in_proj_z.weight", (GDN_V_HEADS * GDN_DIM, H))
        bf16(q + "in_proj_a.weight", (GDN_V_HEADS, H))
        bf16(q + "in_proj_b.weight", (GDN_V_HEADS, H))
        bf16(q + "conv1d.weight", (conv, 1, 4), 0.2)
        bf16(q + "norm.weight", (GDN_DIM,))
        bf16(q + "out_proj.weight", (H, GDN_V_HEADS * GDN_DIM))
        tensors[q + "A_log"] = torch.rand(GDN_V_HEADS, generator=g).add(1).log()
        tensors[q + "dt_bias"] = torch.ones(GDN_V_HEADS)

    lm = "model.language_model."
    bf16(lm + "embed_tokens.weight", (VOCAB, H), 0.02)
    bf16("lm_head.weight", (VOCAB, H), 0.02)
    mixer(lm + "hyper_connection_mixer.", inject=False)
    constants = ple_constants(text)
    table = torch.randn(constants["rows"], PLE_DIM // ((NGRAM - 1) * HEADS_PER_NGRAM), generator=g)
    table = table.clamp(-4, 4).to(torch.float8_e4m3fn)
    for i, kind in enumerate(LAYER_TYPES):
        p = f"{lm}layers.{i}."
        mixer(p + "attn_hyper_connection.")
        mixer(p + "mlp_hyper_connection.")
        if kind == "linear_attention":
            gdn(p)
        else:
            attention(p)
        moe(p, _nvfp4_expert)
        if i + 1 in text["ple_layer_ids"]:
            e = p + "ple."
            emb = e + "ple_embedding."
            half = constants["rows"] // 2
            tensors[emb + "ngram_embedding.shard_0.weight"] = table[:half].clone()
            tensors[emb + "ngram_embedding.shard_1.weight"] = table[half:].clone()
            tensors[emb + "ngram_embedding.weight_scale"] = torch.tensor([0.015625], dtype=torch.bfloat16)
            tensors[emb + "layer_multipliers"] = torch.tensor(
                [m - (1 << 64) if m >= 1 << 63 else m for m in constants["multipliers"]],
                dtype=torch.int64,
            )
            tensors[emb + "ngram_heads_vocab_sizes"] = torch.tensor(constants["moduli"])
            tensors[emb + "ngram_heads_offsets"] = torch.tensor(constants["offsets"])
            bf16(e + "key_proj.weight", (wide, PLE_DIM))
            bf16(e + "value_proj.weight", (H, PLE_DIM))
            for norm in ("norm_key", "norm_query", "norm_conv"):
                bf16(e + norm + ".weight", (wide,))
            bf16(e + "conv1d.weight", (wide, 1, 4), 0.2)
    bf16("mtp.pre_fc_norm_embedding.weight", (H,))
    bf16("mtp.pre_fc_norm_hidden.weight", (wide,))
    bf16("mtp.fc_embedding.weight", (H, H))
    bf16("mtp.fc_hidden.weight", (H, H))
    mixer("mtp.hyper_connection_mixer.", inject=False)
    mixer("mtp.layers.0.attn_hyper_connection.")
    mixer("mtp.layers.0.mlp_hyper_connection.")
    attention("mtp.layers.0.")
    moe("mtp.layers.0.", _block_fp8)

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg, indent=1))
    # Two shards with an index, like the real checkpoint (MTP and table in their own file).
    second = {k for k in tensors if k.startswith("mtp.") or ".ple_embedding." in k}
    shards = {
        "model-00001-of-00002.safetensors": {k: v for k, v in tensors.items() if k not in second},
        "model-00002-of-00002.safetensors": {k: tensors[k] for k in second},
    }
    weight_map = {}
    for name, shard in shards.items():
        save_file({k: v.contiguous() for k, v in shard.items()}, str(path / name))
        weight_map.update({k: name for k in shard})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    write_tokenizer(path)
    return tensors


_SPECIALS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def write_tokenizer(path: Path) -> None:
    """A byte-level BPE vocabulary of 256 byte tokens plus Qwen's specials (265 public IDs)."""
    vocab, extra = {}, 256
    for byte in range(256):
        visible = 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174
        code = byte if visible else extra
        extra += 0 if visible else 1
        vocab[chr(code)] = byte

    def added(index, content):
        return {
            "id": index,
            "content": content,
            "special": True,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }

    tokens = [added(256 + i, content) for i, content in enumerate(_SPECIALS)]
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": tokens[:8]})
    )
    (path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {str(t["id"]): t for t in tokens}}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [256, 258]}))
    (path / "chat_template.jinja").write_text("{{ messages }}")


def divisor_word(multiplier: torch.Tensor) -> bytes:
    """The v3 divisor word of a ModelOpt FP32 multiplier: 1 / multiplier rounded once to FP32."""
    return struct.pack("<f", 1.0 / float(multiplier))
