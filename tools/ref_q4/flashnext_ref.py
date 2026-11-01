# Copied from sybrix/qwen3.8-flash-next-ninfer @a2fd69bb, flash-next-reference/flashnext_ref.py
# Copyright sybrix, Apache License 2.0 (see the repository LICENSE / NOTICE).
# capy-flash changes (marked "capy-flash"): a dense checkpoint (text config
# without num_experts; `dense_ffn` set by ref_dump.py) gets the official Qwen4ExpTextMLP
# (config.intermediate_size, mlp.{gate,up,down}_proj, BF16) in place of the sparse MoE block, which
# transformers' Qwen4ExpTextDecoderLayer always builds. Everything else is unchanged.
# Requires PyTorch, safetensors and transformers with `qwen4_exp` (versions in tools/README.md).
"""Layer-streaming numerical reference for Qwen3.8-Flash-Next (text only).

Runs the official transformers `qwen4_exp` modules one decoder layer at a time,
loading each layer's weights from the NVIDIA ModelOpt NVFP4 checkpoint, so the
whole model never has to be resident. Routed experts are dequantized to BF16
(weight-only semantics, BF16 activations); the n-gram PLE table is gathered
row-by-row from the memory-mapped FP8 shards.

Output (torch.save): input ids, the 4x2560 hyper-connection residual stream
after every decoder layer, the mixed final hidden state, and logits at every
position. NInfer prefill/decode outputs are compared against these with
teacher forcing: run the prompt plus a continuation through the reference once,
then check NInfer's per-step logits against the matching positions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextMLP,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRotaryEmbedding,
)

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
TEXT_PREFIX = "model.language_model."


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self._handles: dict[str, object] = {}

    def _handle(self, name: str):
        file = self.weight_map[name]
        if file not in self._handles:
            self._handles[file] = safe_open(str(self.root / file), framework="pt", device="cpu")
        return self._handles[file]

    def get(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def slice(self, name: str):
        return self._handle(name).get_slice(name)


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """ModelOpt NVFP4: two E2M1 codes per byte (low nibble = even column),
    one E4M3 scale per 16 columns, one FP32 global scale."""
    rows, half_k = packed.shape
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, half_k * 2)
    values = E2M1_VALUES.to(packed.device)[codes]
    block = scale.to(torch.float32).repeat_interleave(16, dim=1)
    return values * block * scale_2.to(torch.float32)


class NGramTable:
    """Row gather over the 128 FP8 shards of the PLE n-gram embedding."""

    def __init__(self, ckpt: Checkpoint, prefix: str):
        self.ckpt = ckpt
        self.names = []
        self.starts = []
        start = 0
        index = 0
        while f"{prefix}.shard_{index}.weight" in ckpt.weight_map:
            name = f"{prefix}.shard_{index}.weight"
            rows = ckpt.slice(name).get_shape()[0]
            self.names.append(name)
            self.starts.append(start)
            start += rows
            index += 1
        self.rows = start
        self.scale = ckpt.get(f"{prefix}.weight_scale").to(torch.float32)
        self.shard_rows = ckpt.slice(self.names[0]).get_shape()[0]

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).tolist()
        out = []
        for row in flat:
            shard = row // self.shard_rows
            local = row - self.starts[shard]
            out.append(self.ckpt.slice(self.names[shard])[local : local + 1])
        rows = torch.cat(out).to(torch.float32) * self.scale
        return rows.reshape(*ids.shape, -1)


def load_module(module: torch.nn.Module, ckpt: Checkpoint, prefix: str, device, dtype=torch.bfloat16):
    """Materialize `module` (built on meta) from checkpoint tensors under `prefix`."""
    state = {}
    for name, param in list(module.named_parameters()) + list(module.named_buffers()):
        key = f"{prefix}.{name}"
        if name in ("experts.gate_up_proj", "mlp.experts.gate_up_proj", "experts.down_proj", "mlp.experts.down_proj"):
            continue
        if "ple_embedding.ngram_embedding" in name:
            continue
        if key not in ckpt.weight_map:
            if isinstance(param, torch.Tensor) and name.endswith(
                ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
            ):
                continue
            raise KeyError(key)
        tensor = ckpt.get(key)
        state[name] = tensor if tensor.dtype in (torch.int64, torch.int32) else tensor.to(dtype)
    return state


def dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 E4M3 weight with one BF16 multiplier per 128x128 block (FP8_PB_WO)."""
    rows, cols = weight.shape
    scale = scale_inv.float().repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]
    return weight.float() * scale


def materialize_layer(
    config, layer_idx: int, ckpt: Checkpoint, device, *, text=None, prefix=None, fp8_experts=False
) -> Qwen4ExpTextDecoderLayer:
    text = text or config.text_config
    dense = getattr(text, "dense_ffn", False)  # capy-flash
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(text, layer_idx)
        if dense:
            layer.mlp = Qwen4ExpTextMLP(text)
    prefix = prefix or f"{TEXT_PREFIX}layers.{layer_idx}"
    state = load_module(layer, ckpt, prefix, device)
    if dense:
        return _finish_layer(layer, state, layer_idx, prefix, ckpt, device)

    experts = text.num_experts
    inter = text.moe_intermediate_size
    hidden = text.hidden_size
    gate_up = torch.empty(experts, 2 * inter, hidden, dtype=torch.bfloat16, device=device)
    down = torch.empty(experts, hidden, inter, dtype=torch.bfloat16, device=device)
    for e in range(experts):
        base = f"{prefix}.mlp.experts.{e}"
        parts = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if fp8_experts:
                parts.append(
                    dequant_fp8_block(
                        ckpt.get(f"{base}.{proj}.weight").to(device),
                        ckpt.get(f"{base}.{proj}.weight_scale_inv").to(device),
                    ).to(torch.bfloat16)
                )
                continue
            parts.append(
                dequant_nvfp4(
                    ckpt.get(f"{base}.{proj}.weight").to(device),
                    ckpt.get(f"{base}.{proj}.weight_scale").to(device),
                    ckpt.get(f"{base}.{proj}.weight_scale_2").to(device),
                ).to(torch.bfloat16)
            )
        gate_up[e, :inter] = parts[0]
        gate_up[e, inter:] = parts[1]
        down[e] = parts[2]
    state["mlp.experts.gate_up_proj"] = gate_up
    state["mlp.experts.down_proj"] = down
    return _finish_layer(layer, state, layer_idx, prefix, ckpt, device)


def _finish_layer(layer, state, layer_idx, prefix, ckpt, device):
    if layer.ple is not None:
        # Swap the 51B-parameter nn.Embedding for a host-side gather before materializing.
        table = NGramTable(ckpt, f"{prefix}.ple.ple_embedding.ngram_embedding")
        layer.ple.ple_embedding.ngram_embedding = _HostEmbedding(table, device)
    layer.to_empty(device=device)
    layer.to(torch.bfloat16)  # floating parameters only; int64 hash buffers keep their dtype
    missing, unexpected = layer.load_state_dict(state, strict=False)
    allowed_missing = {
        n
        for n in missing
        if "ngram_embedding" in n or n.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"))
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"layer {layer_idx}: missing={sorted(set(missing) - allowed_missing)} unexpected={unexpected}"
        )
    if layer.ple is not None:
        emb = layer.ple.ple_embedding
        # Hash constants are derived from config in __init__; to_empty() wiped them.
        emb.layer_multipliers.copy_(ckpt.get(f"{prefix}.ple.ple_embedding.layer_multipliers"))
        emb.ngram_heads_vocab_sizes.copy_(ckpt.get(f"{prefix}.ple.ple_embedding.ngram_heads_vocab_sizes"))
        emb.ngram_heads_offsets.copy_(ckpt.get(f"{prefix}.ple.ple_embedding.ngram_heads_offsets"))
    layer.eval()
    return layer


class _HostEmbedding(torch.nn.Module):
    def __init__(self, table: NGramTable, device):
        super().__init__()
        self.table = table
        self.device_ = device
        # CPU placeholder: the PLE module moves ids to weight.device before calling us.
        self.weight = torch.empty(0)

    def forward(self, ids):
        return self.table.gather(ids.cpu()).to(self.device_, torch.bfloat16)


def _norm(dim, weight, eps, device):
    norm = Qwen4ExpTextRMSNorm(dim, eps=eps).to(device)
    norm.weight.data.copy_(weight.to(device, torch.float32))
    return norm


@torch.no_grad()
def run_mtp(text, ckpt, device, ids, embed, stream, position_embeddings, causal, lm_head):
    """Teacher-forced MTP draft (SGLang Qwen4ExpForCausalLMMTP semantics).

    Column t fuses the main model's pre-mixer streams at t with the embedding of
    token t+1 and predicts token t+2. Positions stay those of the main columns.
    """
    hidden, streams = text.hidden_size, text.hc_count
    eps = text.rms_norm_eps
    seq = ids.shape[1]
    norm_e = _norm(hidden, ckpt.get("mtp.pre_fc_norm_embedding.weight"), eps, device)
    norm_h = _norm(hidden * streams, ckpt.get("mtp.pre_fc_norm_hidden.weight"), eps, device)
    fc_e = ckpt.get("mtp.fc_embedding.weight").to(device, torch.bfloat16)
    fc_h = ckpt.get("mtp.fc_hidden.weight").to(device, torch.bfloat16)
    emb_next = embed[ids[0, 1:]][None].to(torch.bfloat16)  # [1,T-1,H]
    e = torch.nn.functional.linear(norm_e(emb_next), fc_e)  # [1,T-1,H]
    h = norm_h(stream[:, :-1]).view(1, seq - 1, streams, hidden)
    fused = (e.unsqueeze(-2) + torch.nn.functional.linear(h, fc_h)).view(1, seq - 1, streams * hidden)

    mtp_text = copy.deepcopy(text)
    mtp_text.num_hidden_layers = 1
    mtp_text.layer_types = ["full_attention"]
    mtp_text.ple_layer_ids = []
    layer = materialize_layer(None, 0, ckpt, device, text=mtp_text, prefix="mtp.layers.0", fp8_experts=True)
    cos, sin = position_embeddings
    out_streams = layer(
        fused,
        position_embeddings=(cos[:, : seq - 1], sin[:, : seq - 1]),
        attention_mask=causal[..., : seq - 1, : seq - 1],
        conv_mask=None,
        past_key_values=None,
        ple_input_ids=None,
    )
    del layer
    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(text, use_combine=False)
    mixer.to_empty(device=device)
    mixer.to(torch.bfloat16)
    mixer.load_state_dict(load_module(mixer, ckpt, "mtp.hyper_connection_mixer", device))
    final = mixer(out_streams)
    logits = torch.nn.functional.linear(final, lm_head).float()[0]  # [T-1,V]
    argmax = logits.argmax(-1)
    targets = ids[0, 2:]
    hits = (argmax[: seq - 2] == targets).float()
    print(f"mtp: columns={seq - 2} teacher-forced top1 agreement with next-next token {hits.mean().item():.4f}")
    return {
        "mtp_streams": out_streams[0].cpu(),
        "mtp_final": final[0].cpu(),
        "mtp_logits": logits.cpu(),
        "mtp_argmax": argmax.cpu(),
    }


def restore_buffers(module, cls, config):
    """to_empty() leaves non-persistent buffers (e.g. vision RoPE inv_freq) uninitialized; copy them
    from a normally constructed instance."""
    fresh = cls._from_config(config)
    live = dict(module.named_buffers())
    for name, buffer in fresh.named_buffers():
        live[name].copy_(buffer.to(live[name].device, live[name].dtype))


def multimodal_inputs(args, root, config, ckpt, device):
    """Messages (CLI JSON, images included) -> ids, vision embeddings, mask, 3-axis positions.

    The HF processor renders the checkpoint's chat template (thinking off, generation prompt) and
    expands image placeholders; the checkpoint's own vision tower produces the embeddings and
    Qwen4ExpModel.get_rope_index the Text MRoPE positions. `--generated` ids are appended.
    """
    from transformers import AutoProcessor
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel, Qwen4ExpVisionModel

    processor = AutoProcessor.from_pretrained(root)
    messages = json.loads(Path(args.messages).read_text())
    for message in messages:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image" and "image" in part:
                    part["path"] = part.pop("image")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    prompt_ids = inputs["input_ids"]
    generated = [int(x) for x in args.generated.split()] if args.generated else []
    ids = torch.cat([prompt_ids, torch.tensor([generated], dtype=torch.long)], dim=1)
    mm_types = torch.cat(
        [inputs["mm_token_type_ids"], torch.zeros(1, len(generated), dtype=inputs["mm_token_type_ids"].dtype)], dim=1
    )
    print(f"multimodal prompt: {prompt_ids.shape[1]} tokens, image grid {inputs['image_grid_thw'].tolist()}")

    with torch.device("meta"):
        vision = Qwen4ExpVisionModel._from_config(config.vision_config)
    vision.to_empty(device=device)
    vision.to(torch.bfloat16)
    restore_buffers(vision, Qwen4ExpVisionModel, config.vision_config)
    state = {name: ckpt.get("model.visual." + name).to(device, torch.bfloat16) for name, _ in vision.named_parameters()}
    missing, unexpected = vision.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"vision: missing={missing} unexpected={unexpected}")
    vision.eval()
    grid = inputs["image_grid_thw"].to(device)
    out = vision(inputs["pixel_values"].to(device, torch.bfloat16), grid_thw=grid)
    image_embeds = out.pooler_output if hasattr(out, "pooler_output") else out
    if isinstance(image_embeds, (list, tuple)):
        image_embeds = torch.cat(list(image_embeds), dim=0)
    del vision
    torch.cuda.empty_cache()

    with torch.device("meta"):
        shell = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(shell)
    shell.config = config
    shell.spatial_merge_size = config.vision_config.spatial_merge_size
    position_ids, _ = Qwen4ExpModel.get_rope_index(
        shell, ids, mm_token_type_ids=mm_types, image_grid_thw=inputs["image_grid_thw"]
    )
    image_mask = ids[0] == config.image_token_id
    if int(image_mask.sum()) != image_embeds.shape[0]:
        raise RuntimeError(f"{int(image_mask.sum())} image tokens but {image_embeds.shape[0]} embeddings")
    return ids[0], image_embeds, image_mask.to(device), position_ids, prompt_ids.shape[1]


def report_generated(logits, ids, prompt_len):
    """Teacher-forced check of the appended generation (same rows as compare_decode.py)."""
    rows, agree = [], 0
    for i, token in enumerate(ids[prompt_len:].tolist()):
        row = logits[prompt_len - 1 + i]
        top2 = row.topk(2)
        best = int(top2.indices[0])
        agree += best == token
        rows.append(
            {
                "step": i,
                "ninfer": token,
                "reference": best,
                "margin": round(float(top2.values[0] - top2.values[1]), 4),
                "ninfer_logit_gap": round(float(top2.values[0] - row[token]), 4),
            }
        )
    print(json.dumps(rows, indent=1))
    print(f"prompt_tokens={prompt_len} generated={len(rows)} argmax_agreement={agree}/{len(rows)}")


@torch.no_grad()
def run(args):
    root = Path(args.checkpoint)
    config = Qwen4ExpConfig.from_pretrained(root)
    text = config.text_config
    # Boolean masks: the QSA indexer ANDs its selection into them, which eager attention would mis-add.
    text._attn_implementation = "sdpa"
    device = torch.device(args.device)
    ckpt = Checkpoint(root)

    image_embeds = image_mask = position_ids_3d = None
    prompt_len = None
    if args.messages:
        ids, image_embeds, image_mask, position_ids_3d, prompt_len = multimodal_inputs(args, root, config, ckpt, device)
    elif args.ids:
        ids = torch.tensor([int(x) for x in args.ids.split(",")], dtype=torch.long)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(root)
        ids = torch.tensor(tok(Path(args.prompt_file).read_text() if args.prompt_file else args.prompt)["input_ids"])
    if args.max_tokens:
        ids = ids[: args.max_tokens]
    ids = ids[None, :].to(device)
    seq = ids.shape[1]
    if seq > text.indexer_budget + text.indexer_compress_ratio - 1:
        print(f"warning: {seq} tokens exceeds the dense-equivalent QSA window; indexer selection is active")

    embed = ckpt.get(f"{TEXT_PREFIX}embed_tokens.weight").to(device)
    hidden = embed[ids[0]][None].to(torch.bfloat16)
    if image_embeds is not None:
        # Vision embeddings replace their placeholder tokens before the stream broadcast.
        hidden[0][image_mask] = image_embeds.to(device, torch.bfloat16)
    if not args.mtp:
        del embed

    rotary = Qwen4ExpTextRotaryEmbedding(config=text).to(device)
    position_ids = (
        position_ids_3d.to(device)
        if position_ids_3d is not None
        else torch.arange(seq, device=device).view(1, 1, -1).expand(3, 1, -1)
    )
    position_embeddings = rotary(hidden, position_ids)

    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))[None, None]
    stream = hidden.repeat(1, 1, text.hc_count)
    record = {"ids": ids.cpu(), "layers": []}
    for idx in range(text.num_hidden_layers):
        t0 = time.time()
        layer = materialize_layer(config, idx, ckpt, device)
        t1 = time.time()
        stream = layer(
            stream,
            position_embeddings=position_embeddings,
            attention_mask=causal,
            conv_mask=None,
            past_key_values=None,
            ple_input_ids=ids,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not args.nll_only:
            record["layers"].append(stream[0].cpu())
        print(
            f"layer {idx:2d} {text.layer_types[idx]:17s} load {t1 - t0:5.1f}s run {time.time() - t1:5.2f}s "
            f"|h| {stream.float().norm().item():.4e}",
            flush=True,
        )
        del layer
        torch.cuda.empty_cache()

    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(text, use_combine=False)
    mixer.to_empty(device=device)
    mixer.to(torch.bfloat16)
    mixer.load_state_dict(load_module(mixer, ckpt, f"{TEXT_PREFIX}hyper_connection_mixer", device))
    final = mixer(stream)
    lm_head = ckpt.get("lm_head.weight").to(device)
    if args.mtp:
        record.update(run_mtp(text, ckpt, device, ids, embed, stream, position_embeddings, causal, lm_head))
        del embed
    record["final"] = final[0].cpu()
    if args.nll_only:
        # Chunked log-probs keep full [T,V] logits off the device and host.
        targets = ids[0, 1:]
        token_logprobs, argmax = [], []
        for begin in range(0, seq, 512):
            chunk = torch.nn.functional.linear(final[0, begin : begin + 512], lm_head).float()
            argmax.append(chunk.argmax(-1).cpu())
            lp = torch.log_softmax(chunk, dim=-1)
            rows = torch.arange(begin, min(begin + 512, seq - 1), device=device)
            if len(rows):
                token_logprobs.append(lp[rows - begin].gather(1, targets[rows][:, None]).squeeze(1).cpu())
            del chunk, lp
        record["token_logprobs"] = torch.cat(token_logprobs)
        record["argmax"] = torch.cat(argmax)
        record["layers"] = []
        nll = -record["token_logprobs"].mean().item()
        print(f"tokens={seq} scored={seq - 1} mean_nll={nll:.6f} ppl={math.exp(nll):.4f}")
        torch.save(record, args.out)
        print(f"saved {args.out}")
        return
    logits = torch.nn.functional.linear(final, lm_head).float()
    record["logits"] = logits[0].cpu()
    if prompt_len is not None:
        report_generated(logits[0], ids[0], prompt_len)
    logprobs = torch.log_softmax(logits[0], dim=-1)
    targets = ids[0, 1:]
    token_logprobs = logprobs[:-1].gather(1, targets[:, None]).squeeze(1)
    record["token_logprobs"] = token_logprobs.cpu()
    record["argmax"] = logits[0].argmax(-1).cpu()
    nll = -token_logprobs.mean().item()
    print(f"tokens={seq} scored={seq - 1} mean_nll={nll:.6f} ppl={math.exp(nll):.4f}")
    torch.save(record, args.out)
    top = logits[0, -1].topk(5)
    print("last-position top5:", list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])))
    print(f"saved {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="NVIDIA ModelOpt NVFP4 snapshot directory")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--prompt-file")
    p.add_argument("--ids", help="comma-separated token ids (overrides --prompt)")
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--nll-only", action="store_true", help="save only per-token log-probs")
    p.add_argument("--mtp", action="store_true", help="also run the MTP draft head teacher-forced")
    p.add_argument("--messages", help="CLI-style messages JSON (images allowed); multimodal input")
    p.add_argument("--generated", default="", help="space-separated NInfer ids appended to --messages")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="ref.pt")
    run(p.parse_args())


if __name__ == "__main__":
    main()
