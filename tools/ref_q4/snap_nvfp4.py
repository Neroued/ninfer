# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Snap a synthetic dense Qwen4 checkpoint's projections onto an exact NVFP4 grid.

The FP64 oracle (ref_dump.py --impl fp64) reads a checkpoint's BF16 weights; an artifact converted with
the qwen4_nvfp4 recipe stores the attention, GDN and MLP projections as NVFP4 (nvfp4_mse). On random
weights the quantization error alone puts the engine far outside the BF16 tolerance profiles, so an
oracle on the original checkpoint cannot tell a kernel bug from rounding. This tool rewrites those
projections so that the converter's quantization is lossless: the oracle on the snapped checkpoint
then sees exactly the weights the engine decodes, and engine_check.sh compares like with like.

Every snapped matrix W[N,K] is w = e2m1(c) * s / d with one power-of-two divisor d for the whole
checkpoint, E4M3FN group scales s (16 values along K) and E2M1 codes c, built so that
tools/convert/methods.py nvfp4_mse reproduces it bit for bit for any row subset or row concatenation
of these matrices (the parents the recipe forms):
  - the parent divisor is 448 * 4 / amax (the smallest nvfp4_mse target is 4): the first group of
    every row holds the parent amax A = 1792 / d (scale 448, largest code 4), every other value is at
    most A, so each parent's divisor is exactly d;
  - every other group has its largest code at 6 with a scale <= 288, so the first target (6)
    reconstructs it with zero error.
The tool checks the claim with the converter's own quantize_nvfp4 on every snapped matrix.

  snap_nvfp4.py --checkpoint IN_DIR --out OUT_DIR

OUT_DIR gets every non-safetensors file of IN_DIR copied and the shards rewritten (BF16, same index).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.convert.methods import NVFP4_MSE_TARGETS  # noqa: E402
from tools.convert.quantization.nvfp4 import GROUP, quantize_nvfp4, weight_divisor  # noqa: E402

# HF tensors behind the qwen4_nvfp4 recipe's NVFP4 parents (official_recipes._QWEN4_NVFP4_DENSE),
# Text and MTP layers.
PROJECTIONS = re.compile(
    r"\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)\.weight$"
)
E2M1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
MIDPOINTS = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
GROUP_SCALE_CAP = 288.0  # largest E4M3FN value with 6 * s <= 448 * 4
ROW_SCALE = 448.0  # E4M3FN maximum: the first group of each row, largest code 4


def e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def codes(magnitude: torch.Tensor) -> torch.Tensor:
    """E2M1 magnitude codes of values in [0, 6] (ties to the even code: only exactness matters here)."""
    return torch.bucketize(magnitude, MIDPOINTS, right=False)


def snap_groups(groups: torch.Tensor, d: float, scale: torch.Tensor, top: int) -> torch.Tensor:
    """groups [R, G, 16] -> values on the grid e2m1 * scale / d, the largest magnitude at code `top`
    (an index of E2M1: 6 is magnitude 4, 7 is magnitude 6)."""
    step = scale[..., None] / d
    magnitude = groups.abs()
    code = codes(torch.clamp(magnitude / step, max=E2M1[top].item()))
    code = torch.minimum(code, torch.tensor(top))
    lead = magnitude.argmax(dim=-1, keepdim=True)
    code.scatter_(-1, lead, top)
    sign = torch.where(groups < 0, -1.0, 1.0)
    return sign * E2M1[code] * step


def snap(w: torch.Tensor, d: float) -> torch.Tensor:
    n, k = w.shape
    if k % GROUP or k < 2 * GROUP:
        raise ValueError(f"K = {k} is not a multiple of {GROUP} with at least two groups")
    groups = w.to(torch.float32).reshape(n, k // GROUP, GROUP)
    out = torch.empty_like(groups)
    first = torch.full((n, 1), ROW_SCALE)
    out[:, :1] = snap_groups(groups[:, :1], d, first, 6)
    rest = groups[:, 1:]
    scale = e4m3(torch.clamp(rest.abs().amax(dim=-1) * d / 6.0, min=2.0**-6, max=GROUP_SCALE_CAP))
    out[:, 1:] = snap_groups(rest, d, scale, 7)
    return out.reshape(n, k)


def dequantize(words, divisor: float) -> torch.Tensor:
    low, high = words.codes & 0x0F, words.codes >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(words.codes.shape[0], -1).to(torch.int64)
    value = E2M1[nibbles & 7] * torch.where((nibbles & 8) != 0, -1.0, 1.0)
    scale = words.scales.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(GROUP, dim=1)
    return value * scale / divisor


def check(name: str, snapped: torch.Tensor, d: float) -> None:
    amax = float(snapped.abs().max())
    divisor = weight_divisor(amax, NVFP4_MSE_TARGETS)
    if divisor != d:
        raise RuntimeError(f"{name}: parent divisor {divisor} != {d}")
    # Row halves too: the recipe's parents can be row subsets of one HF matrix.
    for rows in (slice(None), slice(0, snapped.shape[0] // 2)):
        part = snapped[rows]
        words = quantize_nvfp4(part, weight_divisor(float(part.abs().max()), NVFP4_MSE_TARGETS), NVFP4_MSE_TARGETS)
        if not torch.equal(dequantize(words, d), part):
            raise RuntimeError(f"{name}: nvfp4_mse does not reproduce the snapped values")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    shards = sorted(args.checkpoint.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{args.checkpoint}: no safetensors shards")
    # One divisor: 99.9 % of the groups below the scale cap in the widest projection.
    q = 0.0
    for shard in shards:
        for name, w in load_file(shard).items():
            if PROJECTIONS.search(name):
                amax = w.to(torch.float32).reshape(w.shape[0], -1, GROUP).abs().amax(dim=-1).flatten()
                q = max(q, float(torch.quantile(amax[: 1 << 24], 0.999)))
    if q == 0.0:
        raise SystemExit("no projection matched")
    d = 2.0 ** math.floor(math.log2(6.0 * GROUP_SCALE_CAP / q))
    args.out.mkdir(parents=True, exist_ok=True)
    for item in args.checkpoint.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, args.out / item.name)
    count = 0
    for shard in shards:
        tensors = load_file(shard)
        for name, w in tensors.items():
            if PROJECTIONS.search(name):
                snapped = snap(w, d)
                check(name, snapped, d)
                tensors[name] = snapped.to(torch.bfloat16)
                if not torch.equal(tensors[name].to(torch.float32), snapped):
                    raise RuntimeError(f"{name}: snapped values are not BF16-exact")
                count += 1
        save_file(tensors, args.out / shard.name, metadata={"format": "pt"})
    (args.out / "snap_nvfp4.json").write_text(
        json.dumps({"source": str(args.checkpoint), "divisor": d, "matrices": count}) + "\n"
    )
    print(f"snapped {count} projections, divisor {d}, -> {args.out}")


if __name__ == "__main__":
    main()
