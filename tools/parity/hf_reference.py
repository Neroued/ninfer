#!/usr/bin/env python3
"""Trusted bf16 reference for Qwen3.6-27B (text core), driven manually.

Why not model.generate(): the upstream Qwen3_5ForConditionalGeneration generate
path manages multimodal mRoPE / rope_deltas + GDN cache state during incremental
decode; with our text-only prompt that path produced a correct first token then
degraded into garbage. To get a trustworthy ground truth we control the compute
graph ourselves:

  * greedy decode is a no-cache full re-forward each step, so every step is a
    fresh, correct prefill with no cache/position-advance corruption.

The model is ~54 GB in bf16 (> one 32 GB GPU) so we use accelerate device_map
"auto" with a CPU spill budget. This is slow and report-only; the Task 5 gate is
the approved q5090 snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from safetensors import safe_open

from tools.q5090_convert import tensor_plan as tp


class HfShardReader:
    """Lazy safetensors reader for local HF bf16 shards."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as f:
            index = json.load(f)
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}
        self._cache_name: Optional[str] = None
        self._cache_tensor: Optional[torch.Tensor] = None

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.model_dir, shard), framework="pt", device="cpu")
            self._handles[shard] = h
        return h

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        if name == self._cache_name:
            assert self._cache_tensor is not None
            return self._cache_tensor
        shard = self.weight_map[name]
        t = self._handle(shard).get_tensor(name)
        self._cache_name = name
        self._cache_tensor = t
        return t

    def rows(self, name: str, row0: int, row1: int) -> torch.Tensor:
        shard = self.weight_map[name]
        return self._handle(shard).get_slice(name)[row0:row1]

    def shape(self, name: str) -> list[int]:
        shard = self.weight_map[name]
        return list(self._handle(shard).get_slice(name).get_shape())

    def close(self) -> None:
        for h in self._handles.values():
            close = getattr(h, "close", None)
            if close is not None:
                close()
        self._handles.clear()
        self._cache_name = None
        self._cache_tensor = None


def parse_id_text(text: str) -> list[int]:
    return [int(t) for t in text.replace(",", " ").split()]


def read_ids(path: str | Path) -> list[int]:
    return parse_id_text(Path(path).read_text(encoding="utf-8"))


def prepare_source_tensor(reader: HfShardReader, spec: tp.TensorSpec) -> torch.Tensor:
    t = reader.get(spec.src_name)
    if spec.row_slice is not None:
        a, b = spec.row_slice
        t = t[a:b]
    if spec.reshape is not None:
        t = t.reshape(spec.reshape)
    if spec.transform is not None:
        if spec.transform == tp.TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE:
            t = tp.runtime_native_gdn_conv1d(t)
        elif spec.transform == tp.TRANSFORM_ATTN_QPROJ_QUERY:
            t = tp.attn_qproj_split(t, take_gate=False)
        elif spec.transform == tp.TRANSFORM_ATTN_QPROJ_GATE:
            t = tp.attn_qproj_split(t, take_gate=True)
        else:
            raise ValueError(f"{spec.name}: unknown tensor transform {spec.transform!r}")
    return t.contiguous()


def prepare_source_rows(
    reader: HfShardReader,
    spec: tp.TensorSpec,
    row0: int,
    row1: int,
) -> torch.Tensor:
    if spec.transform is not None or spec.reshape is not None:
        return prepare_source_tensor(reader, spec)[row0:row1].contiguous()
    if spec.row_slice is not None:
        base0, _ = spec.row_slice
        row0 += base0
        row1 += base0
    return reader.rows(spec.src_name, row0, row1).contiguous()


def load_hf_model(
    model_dir: str | Path,
    *,
    gpu_mem: str = "26GiB",
    cpu_mem: str = "80GiB",
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "HF full-model reference requires CUDA. Run with a CUDA environment such as "
            "/home/neroued/miniconda3/envs/vllm-bench/bin/python."
        )
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "HF full-model reference requires accelerate for device_map='auto'. "
            "Run with /home/neroued/miniconda3/envs/vllm-bench/bin/python."
        ) from exc

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: gpu_mem, "cpu": cpu_mem},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def first_parameter_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hf_greedy_tokens(
    model,
    ids: Iterable[int],
    max_new: int,
    *,
    stop_token_ids: Optional[set[int]] = None,
) -> list[int]:
    seq = list(ids)
    gen: list[int] = []
    device = first_parameter_device(model)
    for _ in range(max_new):
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=inp, use_cache=False)
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        seq.append(nxt)
        if stop_token_ids and nxt in stop_token_ids:
            break
    return gen


def parse_ids(args) -> list[int]:
    if args.ids:
        text = Path(args.ids).read_text(encoding="utf-8")
    elif args.prompt:
        text = args.prompt
    else:
        raise SystemExit("pass --ids <file> or --prompt '<ids>'")
    return parse_id_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16")
    ap.add_argument("--ids", default=None, help="path to a .ids file of prompt token ids")
    ap.add_argument("--prompt", default=None, help="comma/space separated prompt token ids")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--gpu-mem", default="26GiB")
    ap.add_argument("--cpu-mem", default="80GiB")
    ap.add_argument("--stop-token-ids", default="248046,248044")
    args = ap.parse_args()

    ids = parse_ids(args)
    stop = {int(x) for x in args.stop_token_ids.split(",") if x.strip()}
    print(f"prompt_len={len(ids)}", flush=True)

    t0 = time.perf_counter()
    tok, model = load_hf_model(args.model, gpu_mem=args.gpu_mem, cpu_mem=args.cpu_mem)
    print(f"load_s={time.perf_counter() - t0:.1f}", flush=True)

    t1 = time.perf_counter()
    gen = hf_greedy_tokens(model, ids, args.max_new, stop_token_ids=stop)
    print(f"gen_s={time.perf_counter() - t1:.1f} steps={len(gen)}", flush=True)
    print("HF_REF_TOKENS:", gen, flush=True)
    print("HF_REF_TEXT:", repr(tok.decode(gen)), flush=True)


if __name__ == "__main__":
    main()
