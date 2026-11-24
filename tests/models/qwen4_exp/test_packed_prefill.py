"""Generate a small definition for the CPU-only packed Session test."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def fixture(path):
    h, vocab = 128, 257
    config = dict(
        architectures=["Qwen4ExpForCausalLM"], model_type="qwen4_exp_text",
        hidden_act="silu", output_gate_type="sigmoid", tie_word_embeddings=False,
        attention_bias=False, mamba_ssm_dtype="float32", norm_topk_prob=True,
        hidden_size=h, vocab_size=vocab, max_position_embeddings=128, rms_norm_eps=1e-6,
        hc_count=1, hc_lowrank=8, linear_num_key_heads=1, linear_num_value_heads=1,
        linear_key_head_dim=2, linear_value_head_dim=2, linear_conv_kernel_dim=4,
        num_attention_heads=1, num_key_value_heads=1, head_dim=2, indexer_n_heads=1,
        indexer_kv_heads=1, indexer_head_dim=2, indexer_compress_ratio=4, indexer_budget=16,
        num_experts=1, num_experts_per_tok=1, moe_intermediate_size=h,
        shared_expert_intermediate_size=h, num_hidden_layers=1,
        layer_types=["indexed_attention"], ple_layer_ids=[0], ple_shard_shape=[16, 160],
        eos_token_id=256, split_ngram_parts=1, ple_embed_dim=2560, ngram_size=3,
        heads_per_ngram=8, ple_conv_kernel_size=4, ple_conv_dilation=1,
        rope_parameters=dict(rope_type="default", rope_theta=10000, partial_rotary_factor=1,
                             mrope_interleaved=True, mrope_section=[1, 0, 0]),
    )
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    other = iter(range(256, 512))
    alphabet = {chr(i if i in visible else next(other)): i for i in range(256)}
    byte_level = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False,
                      use_regex=False)
    split = dict(
        type="Split",
        pattern=dict(Regex=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        behavior="Isolated",
        invert=False,
    )
    tokenizer = dict(
        version="1.0",
        model=dict(type="BPE", vocab=alphabet, merges=[]),
        pre_tokenizer=dict(type="Sequence", pretokenizers=[split, byte_level]),
        decoder=byte_level,
        added_tokens=[dict(id=256, content="<eos>", single_word=False, lstrip=False,
                           rstrip=False, normalized=False, special=True)],
    )
    resources = {
        "tokenizer.json": json.dumps(tokenizer).encode(),
        "tokenizer_config.json": b"{}",
        "generation_config.json": b'{"eos_token_id": 256}',
        "chat_template.jinja": b"{{ messages }}",
        "ple_hash_controls": struct.pack("<35q", 1, 3, 5, *([1] * 16), *range(16)),
    }
    specs = [ResourceSpec(name, len(value)) for name, value in resources.items()]
    bindings, uses = {}, []

    def weight(name, shape, input_name=None):
        specs.append(TensorSpec(name, shape, "bf16", "contiguous_le_v1"))
        bindings[name] = {"object": name}
        if input_name:
            use(name, input_name)

    def use(name, input_name):
        uses.append(dict(parameter=name, input=input_name, activation_policy="A16Only"))

    def hc(name, final=False):
        weight(name + "norm", (h,))
        weight(name + "down", (8, h), name + "normalized_streams")
        weight(name + "up", (h, 8), name + "lowrank_activation")
        if not final:
            weight(name + "inject", (1, h), name + "normalized_streams")

    layer = "text/layers/0/"
    weight("text/token_embedding", (vocab, h))
    weight("text/output_head", (vocab, h), "text/final_hidden")
    hc("text/final_hc/", final=True)
    hc(layer + "attention_hc/")
    hc(layer + "moe_hc/")
    attn = layer + "attention/"
    for part in ("query", "key", "value", "output_gate", "index_query", "index_key"):
        weight(attn + part, (2, h), layer + "attention_input")
    for part in ("query_norm", "key_norm", "index_query_norm", "index_key_norm"):
        weight(attn + part, (2,))
    weight(attn + "output", (h, 2), attn + "gated_output")
    moe = layer + "moe/"
    for part in ("router", "shared_score", "shared/gate", "shared/up", "shared/down"):
        width = 1 if part in ("router", "shared_score") else h
        edge = moe + "shared/product" if part == "shared/down" else layer + "moe_input"
        weight(moe + part, (width, h), edge)
    for name, width in (("gu", 2 * h), ("down", h)):
        specs.append(TensorSpec(name, (1, width, h), "nvfp4", "batched_block_scale_k16_m128x4_v1"))
    for part, name, first in (("gate", "gu", 0), ("up", "gu", h * h), ("down", "down", 0)):
        key = moe + "experts/" + part
        bindings[key] = {"parts": [dict(object=name, range=[first, first + h * h])]}
        use(key, moe + "experts/product" if part == "down" else layer + "moe_input")
    ple = layer + "ple/"
    for name in ("key", "value"):
        weight(ple + name, (h, 2560), ple + "embedding")
    for name in ("norm_key", "norm_query", "norm_conv"):
        weight(ple + name, (h,))
    weight(ple + "convolution", (4, h))
    specs.append(TensorSpec("table", (16, 160), "nf4_g16_e4m3fn", "row_paged_g16_4k_v1"))
    bindings[ple + "table"] = {"object": "table"}
    components = {"text": dict(config=config, resources={name: name for name in resources})}
    with ArtifactWriter(path, specs, components=components, bindings=bindings, uses=uses) as writer:
        for name, obj in writer.by_id.items():
            data = resources.get(name)
            if data is None:
                # Control tests bind geometry without materializing or executing weights.
                data = bytes(obj.bytes - 4) + struct.pack("<f", 1) if name == "table" else bytes(obj.bytes)
            writer.write_object(name, data)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ninfer-qwen4-prefill-") as directory:
        path = Path(directory) / "model.ninfer"
        fixture(path)
        subprocess.run([sys.argv[1], str(path)], check=True)
