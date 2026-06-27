#!/usr/bin/env python3
"""Random-weight block parity gate for the M2 model wiring."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.parity.ref_model import D, RefModel  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    layer: int
    helper: str
    tolerance: str


CASES = [
    Case("prefill_t4_attn", 3, "attn_mix", "attention_bf16"),
    Case("prefill_t4_gdn", 0, "gdn_mix", "linear_bf16"),
    Case("prefill_t4_full_mlp", 3, "mlp_tail", "linear_bf16"),
    Case("prefill_t4_gdn_mlp", 0, "mlp_tail", "linear_bf16"),
]


def ensure_fixture(path: Path, regenerate: bool) -> None:
    if path.exists() and not regenerate:
        return
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tests/fixtures/make_q5090_fixture.py"),
            "--profile",
            "model-blocks-random",
            "--out",
            str(path),
        ],
        cwd=ROOT,
    )


def read_case_tensor(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size % D != 0:
        raise ValueError(f"{path} has {values.size} floats, not a multiple of hidden dim {D}")
    return values.reshape(values.size // D, D)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    if float(denom) == 0.0:
        return 1.0 if torch.count_nonzero(af - bf).item() == 0 else 0.0
    return float(torch.dot(af, bf) / denom)


def run_oracle(model: RefModel, case: Case, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    model.reset_state()
    if case.helper == "attn_mix":
        return model.attn_mix(case.layer, x, "prefill", positions)
    if case.helper == "gdn_mix":
        return model.gdn_mix(case.layer, x, "prefill")
    if case.helper == "mlp_tail":
        return model.mlp_tail(case.layer, x)
    raise ValueError(case.helper)


def tolerance_limits(name: str) -> tuple[float, float, float]:
    if name == "attention_bf16":
        return 2.0e-3, 1.6e-2, 2.0e-3
    if name == "linear_bf16":
        return 2.0e-3, 1.6e-2, 2.0e-3
    raise ValueError(name)


def compare(case: Case, got_np: np.ndarray, ref: torch.Tensor) -> tuple[bool, str]:
    got = torch.from_numpy(got_np).to(ref.device)
    ref = ref.float()
    diff = got.float() - ref
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    rms = math.sqrt(float(torch.mean(diff.float() * diff.float()).item()))
    ref_rms = math.sqrt(float(torch.mean(ref.float() * ref.float()).item()))
    rms_rel = rms / max(ref_rms, 1.0e-12)
    cos = cosine(got, ref)
    atol, rtol, tail_frac = tolerance_limits(case.tolerance)
    bound = atol + rtol * ref.float().abs()
    violating = diff.abs() > bound
    n_violating = int(torch.count_nonzero(violating).item())
    frac = n_violating / max(int(diff.numel()), 1)
    if n_violating:
        worst_ratio = float((diff.abs()[violating] / bound[violating]).max().item())
    else:
        worst_ratio = 0.0
    ok = (
        bool(torch.isfinite(got).all().item())
        and bool(torch.isfinite(ref).all().item())
        and (
            n_violating == 0
            or (frac <= tail_frac and worst_ratio <= 5.0 and rms_rel <= 8.0e-3)
        )
    )
    status = "PASS" if ok else "FAIL"
    return (
        ok,
        f"{status} {case.name}: cosine={cos:.8f} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} rms_rel={rms_rel:.6g} viol={n_violating} "
        f"frac={frac:.3g} worst={worst_ratio:.3g} tol={case.tolerance}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", default="/tmp/qus_q5090_model_blocks_random.qus")
    ap.add_argument("--dump-exe", default=str(ROOT / "build/tests/qus_block_dump"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    fixture = Path(args.fixture)
    ensure_fixture(fixture, args.regenerate)

    dump_exe = Path(args.dump_exe)
    if not dump_exe.exists():
        raise FileNotFoundError(f"missing dump executable: {dump_exe}")

    with tempfile.TemporaryDirectory(prefix="qus-block-parity-") as tmp:
        out_dir = Path(tmp)
        subprocess.check_call([str(dump_exe), str(fixture), str(out_dir)], cwd=ROOT)

        model = RefModel(fixture, device=args.device, cache_globals=False)
        failed = False
        for case in CASES:
            inp = read_case_tensor(out_dir / f"{case.name}.input.f32")
            got = read_case_tensor(out_dir / f"{case.name}.out.f32")
            x = torch.from_numpy(inp).to(model.device)
            positions = torch.arange(inp.shape[0], device=model.device, dtype=torch.int32)
            ref = run_oracle(model, case, x, positions)
            ok, line = compare(case, got, ref)
            print(line)
            if not ok:
                print(
                    f"first divergent block helper: {case.name} layer={case.layer} "
                    f"helper={case.helper}",
                    file=sys.stderr,
                )
                failed = True
                break
        model.q5090.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
