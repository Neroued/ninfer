#!/usr/bin/env python3
"""L2 weight-level sanity gate for q5090 v2 text weights."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.parity import hf_reference as hfref  # noqa: E402
from tools.parity.ref_model import RefModel  # noqa: E402
from tools.q5090_convert import qtypes as qt  # noqa: E402
from tools.q5090_convert import tensor_plan as tp  # noqa: E402
from tools.q5090_convert.convert import _layer_types, load_config  # noqa: E402

L2_ROW_COS_WARN_THRESHOLD = 0.98
L2_ROW_COS_HARD_MIN = 0.97
L2_TENSOR_MEAN_COS_THRESHOLD = 0.98


@dataclass
class Metrics:
    max_abs: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    row_cos_sum: float = 0.0
    row_cos_count: int = 0
    row_cos_min: float = 1.0
    warn_rows: int = 0

    def add(self, other: "Metrics") -> None:
        self.max_abs = max(self.max_abs, other.max_abs)
        self.sum_sq += other.sum_sq
        self.count += other.count
        self.row_cos_sum += other.row_cos_sum
        self.row_cos_count += other.row_cos_count
        self.row_cos_min = min(self.row_cos_min, other.row_cos_min)
        self.warn_rows += other.warn_rows

    @property
    def rms(self) -> float:
        return math.sqrt(self.sum_sq / max(self.count, 1))

    @property
    def row_cos_mean(self) -> float:
        return self.row_cos_sum / max(self.row_cos_count, 1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    if float(denom) == 0.0:
        return 1.0 if torch.count_nonzero(af - bf).item() == 0 else 0.0
    return float(torch.dot(af, bf) / denom)


def require_selected_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda":
        raise RuntimeError("q5090 parity verification requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for q5090 parity, but this Python has CPU-only PyTorch. "
            "Run with a CUDA environment such as "
            "/home/neroued/miniconda3/envs/py311/bin/python or "
            "/home/neroued/miniconda3/envs/vllm-bench/bin/python."
        )
    return selected


def compare_chunk(q5090: torch.Tensor, hf: torch.Tensor) -> Metrics:
    if q5090.device != hf.device:
        raise ValueError(f"device mismatch: q5090 {q5090.device} vs HF {hf.device}")
    q = q5090.float()
    h = hf.float()
    if q.shape != h.shape:
        raise ValueError(f"shape mismatch: q5090 {tuple(q.shape)} vs HF {tuple(h.shape)}")
    qrows = q.reshape(q.shape[0] if q.dim() else 1, -1)
    hrows = h.reshape(h.shape[0] if h.dim() else 1, -1)
    diff = qrows - hrows

    m = Metrics()
    if diff.numel():
        m.max_abs = float(diff.abs().max().item())
        m.sum_sq = float(torch.sum(diff * diff).item())
        m.count = int(diff.numel())

    dots = torch.sum(qrows * hrows, dim=1)
    qn = torch.linalg.vector_norm(qrows, dim=1)
    hn = torch.linalg.vector_norm(hrows, dim=1)
    denom = qn * hn
    same_zero = (denom == 0) & (torch.count_nonzero(diff, dim=1) == 0)
    cos = torch.where(denom == 0, torch.where(same_zero, 1.0, 0.0), dots / denom)
    m.row_cos_count = int(cos.numel())
    if m.row_cos_count:
        m.row_cos_sum = float(cos.sum().item())
        m.row_cos_min = float(cos.min().item())
        m.warn_rows = int(torch.count_nonzero(cos < L2_ROW_COS_WARN_THRESHOLD).item())
    return m


def compare_full_tensor(a: torch.Tensor, b: torch.Tensor) -> Metrics:
    return compare_chunk(a, b)


def segment_specs(hf_dir: Path) -> list[tp.TensorSegmentSpec]:
    manifest = tp.build_text_manifest(_layer_types(load_config(str(hf_dir))))
    return list(manifest.segments)


def passes_l2_sanity(metrics: Metrics, qtype: int) -> bool:
    if qtype in {qt.QT_BF16, qt.QT_FP32}:
        return metrics.max_abs == 0.0
    return (
        metrics.row_cos_mean >= L2_TENSOR_MEAN_COS_THRESHOLD
        and metrics.row_cos_min >= L2_ROW_COS_HARD_MIN
    )


def l2_status(metrics: Metrics, qtype: int) -> str:
    if not passes_l2_sanity(metrics, qtype):
        return "FAIL"
    return "WARN" if metrics.warn_rows else "PASS"


def run_l2(
    weights: Path,
    hf_dir: Path,
    device: torch.device,
    rows_per_chunk: int,
) -> bool:
    reader = hfref.HfShardReader(hf_dir)
    model = RefModel(weights, device=str(device), cache_globals=False, resident="stream")
    failed = False
    by_qtype: dict[int, Metrics] = {}

    try:
        for seg in segment_specs(hf_dir):
            if seg.name not in model.q5090.views:
                print(f"L2 FAIL {seg.name}: missing q5090 logical view", file=sys.stderr)
                failed = True
                continue
            view = model.q5090.views[seg.name]
            qtype = view.block.qtype
            qtype_metrics = by_qtype.setdefault(qtype, Metrics())

            tensor_metrics = Metrics()
            if view.block.layout == qt.LAYOUT_ROW_SPLIT:
                for row0, row1, q_chunk in model.q5090.row_split_row_chunks(
                    seg.name, device, rows_per_chunk=rows_per_chunk
                ):
                    h_chunk = hfref.prepare_source_rows(reader, seg.source, row0, row1).to(
                        device=device, non_blocking=True
                    )
                    chunk_metrics = compare_chunk(q_chunk, h_chunk)
                    tensor_metrics.add(chunk_metrics)
                    qtype_metrics.add(chunk_metrics)
                    del q_chunk, h_chunk
            else:
                q_tensor = model.q5090.tensor(seg.name, device)
                h_tensor = hfref.prepare_source_tensor(reader, seg.source).to(
                    device=device, non_blocking=True
                )
                tensor_metrics = compare_full_tensor(q_tensor, h_tensor)
                qtype_metrics.add(tensor_metrics)
                del q_tensor, h_tensor

            status = l2_status(tensor_metrics, qtype)
            qname = qt.QTYPE_NAME.get(qtype, str(qtype))
            print(
                f"L2 {status} {seg.name} qtype={qname} "
                f"max_abs={tensor_metrics.max_abs:.6g} rms={tensor_metrics.rms:.6g} "
                f"row_cos_mean={tensor_metrics.row_cos_mean:.8f} "
                f"row_cos_min={tensor_metrics.row_cos_min:.8f} "
                f"rows_below_{L2_ROW_COS_WARN_THRESHOLD:.2f}="
                f"{tensor_metrics.warn_rows}/{tensor_metrics.row_cos_count}"
            )
            if not passes_l2_sanity(tensor_metrics, qtype):
                failed = True

        for qtype, metrics in sorted(by_qtype.items()):
            qname = qt.QTYPE_NAME.get(qtype, str(qtype))
            print(
                f"L2_QTYPE status={l2_status(metrics, qtype)} qtype={qname} "
                f"max_abs={metrics.max_abs:.6g} "
                f"rms={metrics.rms:.6g} row_cos_mean={metrics.row_cos_mean:.8f} "
                f"row_cos_min={metrics.row_cos_min:.8f} "
                f"rows_below_{L2_ROW_COS_WARN_THRESHOLD:.2f}="
                f"{metrics.warn_rows}/{metrics.row_cos_count}"
            )
    finally:
        reader.close()
        model.q5090.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return not failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--hf", required=True, help="local HF bf16 model directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows-per-chunk", type=int, default=2048)
    args = ap.parse_args()

    weights = Path(args.weights)
    hf_dir = Path(args.hf)
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not hf_dir.exists():
        raise FileNotFoundError(hf_dir)
    device = require_selected_device(args.device)

    ok_l2 = run_l2(weights, hf_dir, device, args.rows_per_chunk)
    if not ok_l2:
        raise SystemExit(1)
    print("PASS L2 weight sanity gate")


if __name__ == "__main__":
    main()
