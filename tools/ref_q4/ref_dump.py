# Copyright 2026 Valerio Dolci. Apache License 2.0.
# The reference runner of the Qwen4 validation (synthetic.sh): this directory's FP64 oracle
# (qwen4_reference.py) and HF reference (flashnext_ref.py), both with the dense FFN and PLE-less
# configs; dense checkpoints are plain BF16.
"""Run a Qwen4Exp (Qwen3.8-Flash-Next) reference on a token sequence and write a tfdump/1 (see tfdump.py).

Two independent references read the same NVIDIA ModelOpt NVFP4 checkpoint (real or synthetic):

  --impl hf    sybrix's layer-streaming PyTorch reference (flashnext_ref.py): the official
               transformers `qwen4_exp` decoder layers, one at a time, BF16 weights and activations, experts
               dequantized NVFP4 -> BF16, PLE rows gathered from the FP8 shards, MTP after SGLang. Deviation from
               sybrix: lm_head is applied in FP32 (sybrix rounds logits to BF16, ~0.06 at |logit| 16).
  --impl fp64  ninfer-ext's FP64 NumPy oracle (qwen4_reference.py) fed by a lazy view of the
               checkpoint: every weight is decoded exactly into FP64 (NVFP4 code*E4M3*FP32, FP8*BF16), layer by
               layer, experts on demand (LRU). CPU only. QSA ties broken by lower block index.

Both stream the checkpoint one decoder layer at a time, so the real 130 GB checkpoint never has to be resident.
Rows = all positions; hidden.L<i> = wide residual [T, hc_count*hidden] after decoder layer i; hidden.final = output mixer.

  ref_dump.py --checkpoint DIR --impl fp64 --tokens toks.txt --out ref64.tfdump [--mtp] [--layers all|none|0,3]
  ref_dump.py --checkpoint DIR --impl hf --text-file pg19_4096.txt --max-tokens 64 --out hf.tfdump --device cpu

Requires NumPy and safetensors; --impl hf also PyTorch and transformers with `qwen4_exp`; --text-file also
tokenizers (versions in tools/README.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tfdump  # noqa: E402

TEXT_PREFIX = "model.language_model."


def rss_gib(field="VmHWM"):
    """VmHWM includes clean, reclaimable pages of the mmapped shards; RssAnon is the real memory demand."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith(field):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("nan")


ANON_PEAK = [0.0]


def anon():
    a = rss_gib("RssAnon")
    ANON_PEAK[0] = max(ANON_PEAK[0], a)
    return a


def read_tokens(args, root):
    if args.tokens:
        p = Path(args.tokens)
        if p.exists():
            ids = np.load(p) if p.suffix == ".npy" else np.asarray(p.read_text().replace(",", " ").split(), np.int64)
        else:
            ids = np.asarray(args.tokens.replace(",", " ").split(), np.int64)
    else:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        ids = np.asarray(
            tok.encode(Path(args.text_file).read_text(encoding="utf-8"), add_special_tokens=False).ids, np.int64
        )
    if args.max_tokens:
        ids = ids[: args.max_tokens]
    return ids.astype(np.int64)


def want_layer(spec, i):
    return spec == "all" or (spec != "none" and i in {int(x) for x in spec.split(",")})


# ---------------------------------------------------------------------------------------------------------------
# FP64: lazy exact decoding of the checkpoint into the oracle's logical weight names

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class Store:
    def __init__(self, root: Path):
        from safetensors import safe_open

        self._open = safe_open
        self.root = root
        self.map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def has(self, name):
        return name in self.map

    def _h(self, name):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = self._open(str(self.root / f), framework="pt", device="cpu")
        return self.handles[f]

    def f64(self, name):
        import torch

        t = self._h(name).get_tensor(name)
        return t.numpy() if t.dtype in (torch.int64, torch.int32) else t.to(torch.float64).numpy()

    def raw(self, name):
        return self._h(name).get_tensor(name)

    def slice(self, name):
        return self._h(name).get_slice(name)


def nvfp4_f64(store, prefix):
    import torch

    packed = store.raw(prefix + ".weight").numpy()
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scale = store.raw(prefix + ".weight_scale").to(torch.float64).numpy()
    s2 = float(store.raw(prefix + ".weight_scale_2").to(torch.float64))
    return E2M1[codes] * np.repeat(scale, 16, axis=1) * s2


def fp8_block_f64(store, prefix, block=128):
    import torch

    w = store.raw(prefix + ".weight").to(torch.float64).numpy()
    s = store.raw(prefix + ".weight_scale_inv").to(torch.float64).numpy()
    r, c = w.shape
    return w * np.repeat(np.repeat(s, block, 0)[:r], block, 1)[:, :c]


class ExpertBank:
    """``bank[e]`` = logical [2I, H] gate_up (gate rows first) or [H, I] down of expert e, FP64, LRU-cached."""

    def __init__(self, store, prefix, kind, cap):
        self.store, self.prefix, self.kind, self.cap = store, prefix, kind, cap
        self.cache = OrderedDict()

    def _dq(self, leaf):
        return (
            nvfp4_f64(self.store, leaf) if self.store.has(leaf + ".weight_scale_2") else fp8_block_f64(self.store, leaf)
        )

    def __getitem__(self, e):
        e = int(e)
        if e in self.cache:
            self.cache.move_to_end(e)
            return self.cache[e]
        base = f"{self.prefix}.{e}"
        if self.kind == "gate_up":
            m = np.concatenate([self._dq(base + ".gate_proj"), self._dq(base + ".up_proj")], axis=0)
        else:
            m = self._dq(base + ".down_proj")
        self.cache[e] = m
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return m


class NgramTable:
    """``table[rows]`` over the FP8 shards ``...ngram_embedding.shard_<i>.weight`` times the BF16 weight_scale."""

    def __init__(self, store, prefix):
        import torch

        self.store, self.names, self.starts = store, [], []
        start = i = 0
        while store.has(f"{prefix}.shard_{i}.weight"):
            n = f"{prefix}.shard_{i}.weight"
            self.names.append(n)
            self.starts.append(start)
            start += store.slice(n).get_shape()[0]
            i += 1
        self.rows = start
        self.shard_rows = store.slice(self.names[0]).get_shape()[0]
        self.scale = float(store.raw(prefix + ".weight_scale").to(torch.float64).reshape(-1)[0])

    def __getitem__(self, ids):
        import torch

        ids = np.asarray(ids)
        out = []
        for row in ids.reshape(-1).tolist():
            s = row // self.shard_rows
            local = row - self.starts[s]
            out.append(self.store.slice(self.names[s])[local : local + 1].to(torch.float64).numpy())
        return (np.concatenate(out) * self.scale).reshape(*ids.shape, -1)


class RowGather:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def __getitem__(self, ids):
        import torch

        sl = self.store.slice(self.name)
        ids = np.asarray(ids).reshape(-1)
        return np.concatenate([sl[int(i) : int(i) + 1].to(torch.float64).numpy() for i in ids]) if len(ids) else None


class _HeadT:
    __array_ufunc__ = None  # make numpy defer `x @ head.T` to __rmatmul__

    def __init__(self, store, name, chunk=16384):
        self.store, self.name, self.chunk = store, name, chunk

    def __rmatmul__(self, x):
        import torch

        sl = self.store.slice(self.name)
        rows = sl.get_shape()[0]
        out = np.empty((x.shape[0], rows))
        for b in range(0, rows, self.chunk):
            out[:, b : b + self.chunk] = x @ sl[b : b + self.chunk].to(torch.float64).numpy().T
        return out


class LazyHead:
    def __init__(self, store, name):
        self.T = _HeadT(store, name)


class LazyWeights:
    """Oracle names (``layers.N.*``, ``embed_tokens.weight``, ``mtp.*``) -> FP64 arrays, one layer group resident."""

    def __init__(self, store: Store, expert_cache=96):
        self.store, self.expert_cache = store, expert_cache
        self.group, self.cache = None, {}

    def _src(self, name):
        if name.startswith("mtp.") or name == "lm_head.weight":
            return name
        return TEXT_PREFIX + name

    def __getitem__(self, name):
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] in ("layers",) else parts[0] if parts[0] == "mtp" else "top"
        if group != self.group and group != "top":
            self.cache = {k: v for k, v in self.cache.items() if k.startswith("@top:")}
            self.group = group
        key = f"@{'top' if group == 'top' else group}:{name}"
        if key in self.cache:
            return self.cache[key]
        src = self._src(name)
        if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            v = ExpertBank(
                self.store,
                src.rsplit(".", 1)[0],
                "gate_up" if name.endswith("gate_up_proj") else "down",
                self.expert_cache,
            )
        elif name.endswith("ple_embedding.ngram_embedding.weight"):
            v = NgramTable(self.store, src.rsplit(".", 1)[0])
        elif name == "embed_tokens.weight":
            v = RowGather(self.store, src)
        elif name == "lm_head.weight":
            v = LazyHead(self.store, src)
        else:
            v = self.store.f64(src)
        self.cache[key] = v
        return v


def run_fp64(root, ids, args):
    import qwen4_reference as ref

    cfg_json = json.loads((root / "config.json").read_text())
    cfg = ref.TextConfig.from_hf(cfg_json["text_config"])
    store = Store(root)
    W = LazyWeights(store, args.expert_cache)
    toks = [int(t) for t in ids]
    T = len(toks)
    dump = tfdump.Dump(ids)
    state = ref.TextState(cfg)
    positions = ref.text_positions(0, T)
    x = np.tile(W["embed_tokens.weight"][np.asarray(toks)], (1, cfg.hc_count))
    for i, kind in enumerate(cfg.layer_types[: min(cfg.num_hidden_layers, args.max_layers or 10**9)]):
        t0 = time.time()
        ple = (toks, state.ple[i], cfg.ple_layer_ids.index(i + 1)) if i in state.ple else None
        x = ref.decoder_layer(
            cfg, W, f"layers.{i}", kind, x, positions, state.layers[i], ple, sparse=not args.dense_qsa
        )
        if want_layer(args.layers, i):
            dump.put(f"hidden.L{i:02d}", x.astype(np.float32))
        print(
            f"fp64 layer {i:2d} {kind:17s} {time.time() - t0:6.1f}s |x| {np.linalg.norm(x):.6e} "
            f"peakRSS {rss_gib():.1f} GiB anon {anon():.1f} GiB",
            flush=True,
        )
    final = ref.hyper_connection_mix(cfg, W, "hyper_connection_mixer", x, combine=False)
    logits = final @ W["lm_head.weight"].T
    if args.layers != "none":
        dump.put("hidden.final", final.astype(np.float32))
    if args.mtp and T > 1:
        mlog, _ = ref.mtp_forward(cfg, W, state, toks[1:], x[:-1], ref.text_positions(0, T - 1))
        dump.put("mtp_logits", mlog.astype(np.float32), np.arange(T - 1))
    return dump, logits


# ---------------------------------------------------------------------------------------------------------------
# HF BF16 (sybrix)


def run_hf(root, ids, args):
    import torch
    import flashnext_ref as fr

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg_json = json.loads((root / "config.json").read_text())
    config = fr.Qwen4ExpConfig.from_pretrained(root)
    text = config.text_config
    raw = cfg_json["text_config"]
    text.dense_ffn = not raw.get("num_experts")  # capy-flash: dense SwiGLU FFN
    if text.dense_ffn:
        text.intermediate_size = raw["intermediate_size"]
    text._attn_implementation = "sdpa"
    device = torch.device(args.device)
    ckpt = fr.Checkpoint(root)
    dump = tfdump.Dump(ids)
    with torch.no_grad():
        idt = torch.tensor(ids, dtype=torch.long)[None].to(device)
        seq = idt.shape[1]
        embed = ckpt.get(f"{TEXT_PREFIX}embed_tokens.weight").to(device)
        hidden = embed[idt[0]][None].to(torch.bfloat16)
        rotary = fr.Qwen4ExpTextRotaryEmbedding(config=text).to(device)
        position_ids = torch.arange(seq, device=device).view(1, 1, -1).expand(3, 1, -1)
        pe = rotary(hidden, position_ids)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))[None, None]
        stream = hidden.repeat(1, 1, text.hc_count)
        for idx in range(min(text.num_hidden_layers, args.max_layers or 10**9)):
            t0 = time.time()
            layer = fr.materialize_layer(config, idx, ckpt, device)
            t1 = time.time()
            stream = layer(
                stream,
                position_embeddings=pe,
                attention_mask=causal,
                conv_mask=None,
                past_key_values=None,
                ple_input_ids=idt,
            )
            if want_layer(args.layers, idx):
                dump.put(f"hidden.L{idx:02d}", stream[0].float().cpu().numpy())
            print(
                f"hf layer {idx:2d} {text.layer_types[idx]:17s} load {t1 - t0:5.1f}s run {time.time() - t1:6.2f}s "
                f"|h| {stream.float().norm().item():.6e} peakRSS {rss_gib():.1f} GiB anon {anon():.1f} GiB",
                flush=True,
            )
            del layer
        with torch.device("meta"):
            mixer = fr.Qwen4ExpTextGatedResidual(text, use_combine=False)
        mixer.to_empty(device=device)
        mixer.to(torch.bfloat16)
        mixer.load_state_dict(fr.load_module(mixer, ckpt, f"{TEXT_PREFIX}hyper_connection_mixer", device))
        final = mixer(stream)
        lm_head = ckpt.get("lm_head.weight").to(device)
        logits = torch.nn.functional.linear(final[0].float(), lm_head.float()).double().cpu().numpy()
        if args.layers != "none":
            dump.put("hidden.final", final[0].float().cpu().numpy())
        if args.mtp and seq > 2:
            rec = fr.run_mtp(text, ckpt, device, idt, embed, stream, pe, causal, lm_head)
            mlog = torch.nn.functional.linear(rec["mtp_final"].to(device).float(), lm_head.float()).cpu().numpy()
            dump.put("mtp_logits", mlog, np.arange(seq - 1))
    return dump, logits


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--impl", choices=("hf", "fp64"), required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", help=".npy, text file of ids, or a comma/space list")
    src.add_argument("--text-file", help="UTF-8 text, tokenized with the checkpoint tokenizer.json, no specials")
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--out", required=True, help="output dir (tfdump) or .safetensors")
    p.add_argument("--layers", default="all", help="all | none | comma list of layer indices to dump")
    p.add_argument("--mtp", action="store_true", help="also the MTP head, teacher-forced")
    p.add_argument("--no-logits", action="store_true", help="keep only token_logprobs (long sequences)")
    p.add_argument("--device", default="cpu", help="hf only: cpu or cuda:N")
    p.add_argument("--threads", type=int, default=0, help="hf only: torch CPU threads")
    p.add_argument("--expert-cache", type=int, default=96, help="fp64 only: dequantized experts kept per bank")
    p.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="smoke only: stop after N decoder layers (logits are then NOT the model's)",
    )
    p.add_argument("--dense-qsa", action="store_true", help="fp64 only: dense causal attention (no indexer)")
    args = p.parse_args()
    root = Path(args.checkpoint)
    ids = read_tokens(args, root)
    t0 = time.time()
    dump, logits = (run_fp64 if args.impl == "fp64" else run_hf)(root, ids, args)
    # per-token log-probs in FP64 from the full-precision logits
    lse = np.empty(logits.shape[0])
    for b in range(0, logits.shape[0], 256):  # chunked: no [T,V] FP64 temporaries
        c = logits[b : b + 256]
        m = c.max(-1, keepdims=True)
        lse[b : b + 256] = (m + np.log(np.exp(c - m).sum(-1, keepdims=True)))[:, 0]
    anon()
    T = len(ids)
    lp = logits[np.arange(T - 1), ids[1:]] - lse[: T - 1]
    dump.token_logprobs = lp.astype(np.float32)
    if not args.no_logits:
        dump.put("logits", logits.astype(np.float32))
    nll = -float(lp.mean()) if T > 1 else float("nan")
    dump.meta.update(
        {
            "producer": f"tools/ref_q4/ref_dump.py --impl {args.impl}",
            "checkpoint": root.name,
            "impl": args.impl,
            "device": args.device if args.impl == "hf" else "cpu",
            "n_tokens": int(T),
            "mean_nll": nll,
            "ppl": math.exp(nll) if T > 1 else None,
            "seconds": round(time.time() - t0, 1),
            "peak_rss_gib": round(rss_gib(), 2),
            "peak_anon_gib_sampled": round(ANON_PEAK[0], 2),
            "argmax": logits.argmax(-1).tolist(),
            "dense_qsa": bool(args.dense_qsa),
            "truncated_layers": args.max_layers or None,
        }
    )
    dump.save(args.out)
    print(
        f"{args.impl}: tokens={T} scored={T - 1} mean_nll={nll:.6f} ppl={math.exp(nll):.4f} "
        f"time {time.time() - t0:.0f}s peakRSS {rss_gib():.1f} GiB anon-peak(sampled) {ANON_PEAK[0]:.1f} GiB -> {args.out}"
    )


if __name__ == "__main__":
    main()
