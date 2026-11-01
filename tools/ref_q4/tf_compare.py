# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Teacher-forcing comparison of an engine dump against a reference dump (both tfdump/1, see tfdump.py).

  tf_compare.py REF ENG [--profile bf16|w4a16|w4a4|none [--floor FLOOR.json]] [--tol kl_mean=1e-3,top1_min=0.98,...] [--json out.json]

Rows are matched by sequence position; the token sequences must agree on every position a compared row depends
on (0..max_row, plus the next token for NLL). Per quantity:
  logits / mtp_logits  KL(ref||eng) per row in FP64 (mean, p99, max), top-1 agreement, decisive disagreements
                       (reference top-1/top-2 margin > --tie), max/mean |dlogit|, NLL of the actual next token
                       (mean delta, both PPLs over the common rows)
  hidden.L<i>, final   per layer: rel L2 = ||d||_F/||ref||_F, worst row rel L2, min row cosine, max/mean |d|
  token_logprobs       used when a side has no logits: mean/max |dNLL| and both PPLs
Exit 0 = every threshold of the profile/--tol holds (or none set), 1 = a threshold failed, 2 = an input that cannot
be read or compared (missing or malformed dump, unknown profile or tolerance, no common rows).
Requires NumPy (and safetensors for .safetensors dumps).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tfdump  # noqa: E402

# Engine classes. Thresholds are a factor K times the FLOOR = the distance of the BF16 reference (--impl hf) from
# the FP64 oracle on the same model and tokens (`tf_compare.py ref.fp64 ref.hf --json floor.json`), so they follow
# the model (synthetic logits are near-uniform, real ones peaked) and the sequence (QSA selection, depth).
#   bf16   same math as the BF16 reference (BF16 weights/activations, FP32 accumulation): K = 2
#   w4a16  NVFP4 experts decoded exactly, 16-bit activations (the decode route): K = 3
#   w4a4   NVFP4 experts with FP4 activations (the W4A4 expert prefill) and Q8/FP8 dense: K = 10 (4.7x the KL
#          floor was measured on the synthetic fn4 model of tools/capy_flash/make_test_artifact.sh), top-1
#          decisive mismatches allowed up to 2 % of rows
# The MTP head (mtp_logits) is held to its own floor under the same K: its error is that of the main hidden it
# reads plus the MTP layer's, and its floor KL differs from the main head's (about half of it on the synthetic
# Qwen4 presets), so the main head's numbers would mis-scale it. Its keys carry the `mtp_logits.` prefix
# (`mtp_logits.kl_max`, ...); the top-1 criterion of the profile applies to both heads.
# Without --floor, the absolute numbers below apply: they are 2x/3x/10x the floor measured on fn4 at 160 tokens
# (KL mean 9.1e-5, max 3.9e-4, worst layer rel 2.9e-2) and are only meaningful on that synthetic model; no MTP
# floor was measured there, so the MTP head then has the top-1 criterion only.
FACTORS = {"bf16": 2.0, "w4a16": 3.0, "w4a4": 10.0}
FN4_FLOOR = {"kl_mean": 9.1e-5, "kl_max": 3.9e-4, "hidden_rel": 2.9e-2}
# The compared heads and the prefix of their threshold keys.
HEADS = (("logits", ""), ("mtp_logits", "mtp_logits."))


def profile_tol(profile, floor):
    if profile == "none":
        return {}
    k = FACTORS[profile]
    tol = {}
    if floor:
        for name, prefix in HEADS:
            lg = floor.get(name) or {}
            if lg:
                tol[prefix + "kl_mean"] = k * lg["kl_mean"]
                tol[prefix + "kl_max"] = k * lg["kl_max"]
        for name, h in floor.get("hidden", {}).items():
            tol[f"hidden_rel.{name}"] = k * h["rel"]
    else:
        tol = {
            "kl_mean": k * FN4_FLOOR["kl_mean"],
            "kl_max": k * FN4_FLOOR["kl_max"],
            "hidden_rel_max": k * FN4_FLOOR["hidden_rel"],
        }
    for _, prefix in HEADS:
        if profile == "w4a4":
            tol[prefix + "top1_decisive_mismatch_frac"] = 0.02
        else:
            tol[prefix + "top1_decisive_mismatch_max"] = 0
    return tol


def _logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


def _common(rows_a, rows_b):
    common, ia, ib = np.intersect1d(rows_a, rows_b, assume_unique=False, return_indices=True)
    return common, ia, ib


def compare_logits(ref, eng, name, tie, offset):
    """offset = 1 for logits (row p predicts token p+1), 2 for mtp_logits (predicts p+2)."""
    (ra, A), (rb, B) = ref.get(name), eng.get(name)
    rows, ia, ib = _common(ra, rb)
    if len(rows) == 0:
        return None
    if A.shape[1] != B.shape[1]:
        raise InputError(f"{name}: vocab {A.shape[1]} vs {B.shape[1]}")
    T = min(len(ref.tokens), len(eng.tokens))
    kl, top1, decisive, maxabs, meanabs, margins, dnll, nll_r, nll_e = [], [], [], [], [], [], [], [], []
    worst = []
    for r, a, b in zip(rows, A[ia], B[ib]):
        lp, lq = _logsoftmax(a), _logsoftmax(b)
        p = np.exp(lp)
        k = float(np.sum(p * (lp - lq)))
        kl.append(max(k, 0.0))
        o = np.argsort(a)[-2:][::-1]
        ia1, eb1 = int(o[0]), int(np.argmax(b))
        margin = float(a[o[0]] - a[o[1]])
        margins.append(margin)
        top1.append(ia1 == eb1)
        if ia1 != eb1 and margin > tie:
            decisive.append({"pos": int(r), "ref": ia1, "eng": eb1, "ref_margin": round(margin, 4)})
        d = np.abs(a.astype(np.float64) - b)
        maxabs.append(float(d.max()))
        meanabs.append(float(d.mean()))
        if r + offset < T:
            t = int(ref.tokens[r + offset])
            nll_r.append(-lp[t])
            nll_e.append(-lq[t])
            dnll.append(lp[t] - lq[t])
        worst.append((k, int(r)))
    kl = np.asarray(kl)
    out = {
        "rows": int(len(rows)),
        "first_row": int(rows[0]),
        "last_row": int(rows[-1]),
        "kl_mean": float(kl.mean()),
        "kl_p99": float(np.quantile(kl, 0.99)),
        "kl_max": float(kl.max()),
        "top1": float(np.mean(top1)),
        "top1_mismatch": int(len(top1) - sum(top1)),
        "top1_decisive_mismatch": len(decisive),
        "decisive": decisive[:20],
        "near_ties": int(sum(m <= tie for m in margins)),
        "dlogit_max": float(max(maxabs)),
        "dlogit_mean": float(np.mean(meanabs)),
        "worst_kl_rows": [{"pos": p, "kl": round(k, 6)} for k, p in sorted(worst, reverse=True)[:5]],
    }
    if dnll:
        out.update(
            {
                "scored": len(dnll),
                "dnll_mean": float(np.mean(dnll)),
                "dnll_absmax": float(np.max(np.abs(dnll))),
                "ppl_ref": math.exp(float(np.mean(nll_r))),
                "ppl_eng": math.exp(float(np.mean(nll_e))),
            }
        )
    return out


def compare_hidden(ref, eng, name):
    (ra, A), (rb, B) = ref.get(name), eng.get(name)
    rows, ia, ib = _common(ra, rb)
    if len(rows) == 0:
        return None
    A, B = A[ia].astype(np.float64), B[ib].astype(np.float64)
    if A.shape != B.shape:
        raise InputError(f"{name}: shape {A.shape} vs {B.shape}")
    D = A - B
    row_rel = np.linalg.norm(D, axis=1) / np.maximum(np.linalg.norm(A, axis=1), 1e-30)
    cos = np.sum(A * B, 1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-30)
    return {
        "rows": int(len(rows)),
        "rel": float(np.linalg.norm(D) / max(np.linalg.norm(A), 1e-30)),
        "row_rel_max": float(row_rel.max()),
        "row_rel_argmax_pos": int(rows[int(row_rel.argmax())]),
        "cos_min": float(cos.min()),
        "abs_max": float(np.abs(D).max()),
        "abs_mean": float(np.abs(D).mean()),
        "ref_rms": float(np.sqrt(np.mean(A * A))),
    }


def compare_logprobs(ref, eng):
    a, b = ref.token_logprobs, eng.token_logprobs
    n = min(len(a), len(b))
    if n == 0:
        return None
    d = a[:n].astype(np.float64) - b[:n]
    return {
        "scored": n,
        "dnll_mean": float(-d.mean()),
        "dnll_absmax": float(np.abs(d).max()),
        "ppl_ref": math.exp(-float(a[:n].mean())),
        "ppl_eng": math.exp(-float(b[:n].mean())),
    }


class InputError(Exception):
    """Dumps that cannot be compared (exit 2, as unreadable input)."""


def check_tokens(ref, eng, last_needed):
    n = min(len(ref.tokens), len(eng.tokens), last_needed + 1)
    bad = np.nonzero(ref.tokens[:n] != eng.tokens[:n])[0]
    if len(bad):
        raise InputError(
            f"token sequences differ first at position {int(bad[0])}: "
            f"ref {int(ref.tokens[bad[0]])} eng {int(eng.tokens[bad[0]])} — not the same teacher forcing"
        )


def verdict(report, tol):
    fails = []
    for name, prefix in HEADS:
        lg = report.get(name)
        if lg is None:
            continue
        for key in ("kl_mean", "kl_max"):
            t = tol.get(prefix + key)
            if t is not None and lg[key] > t:
                fails.append(f"{name} {key}: {lg[key]:.4g} vs {t:.4g}")
        t = tol.get(prefix + "top1_min")
        if t is not None and lg["top1"] < t:
            fails.append(f"{name} top1_min: {lg['top1']:.4g} vs {t:.4g}")
        t = tol.get(prefix + "top1_decisive_mismatch_max")
        if t is not None and lg["top1_decisive_mismatch"] > t:
            fails.append(f"{name} top1_decisive_mismatch_max: {lg['top1_decisive_mismatch']:.4g} vs {t:.4g}")
        t = tol.get(prefix + "top1_decisive_mismatch_frac")
        if t is not None:
            frac = lg["top1_decisive_mismatch"] / max(lg["rows"], 1)
            if frac > t:
                fails.append(f"{name} decisive top-1 mismatches {frac:.3f} of rows > {t}")
        t = tol.get(prefix + "dnll_absmean_max")
        if t is not None and "dnll_mean" in lg and abs(lg["dnll_mean"]) > t:
            fails.append(f"{name} dnll_mean {lg['dnll_mean']:.4g}")
    for name, h in report.get("hidden", {}).items():
        t = tol.get(f"hidden_rel.{name}", tol.get("hidden_rel_max"))
        if t is not None and h["rel"] > t:
            fails.append(f"{name} rel {h['rel']:.4g} > {t:.4g}")
    return fails


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("eng")
    p.add_argument("--profile", default="none", choices=["none", *FACTORS])
    p.add_argument("--floor", help="tf_compare JSON of the hf reference vs the fp64 reference on the same tokens")
    p.add_argument(
        "--tol",
        default="",
        help="overrides, e.g. kl_mean=1e-3,top1_min=0.99,mtp_logits.kl_max=1e-3,hidden_rel_max=0.02",
    )
    p.add_argument("--tie", type=float, default=0.1, help="reference top-1/top-2 logit margin counted as a near-tie")
    p.add_argument("--json", help="write the full report here")
    a = p.parse_args()
    try:
        compare(a)
    except (InputError, OSError, ValueError, KeyError) as error:
        print(f"tf_compare: input error: {error}", file=sys.stderr)
        sys.exit(2)


def compare(a):
    ref, eng = tfdump.load(a.ref), tfdump.load(a.eng)
    floor = json.loads(Path(a.floor).read_text()) if a.floor else None
    tol = profile_tol(a.profile, floor)
    for kv in filter(None, a.tol.split(",")):
        k, v = kv.split("=")
        tol[k.strip()] = float(v)
    report = {
        "ref": a.ref,
        "eng": a.eng,
        "ref_meta": {k: v for k, v in ref.meta.items() if k != "argmax"},
        "eng_meta": {k: v for k, v in eng.meta.items() if k != "argmax"},
        "profile": a.profile,
        "tol": tol,
        "tie": a.tie,
        "hidden": {},
    }
    last = 0
    for name in ("logits", "mtp_logits"):
        if ref.get(name) and eng.get(name):
            last = max(last, int(max(ref.get(name)[0].max(), eng.get(name)[0].max())) + (1 if name == "logits" else 2))
    for name in set(ref.layer_names()) & set(eng.layer_names()):
        last = max(last, int(eng.get(name)[0].max()))
    check_tokens(ref, eng, last)
    for name, off in (("logits", 1), ("mtp_logits", 2)):
        if ref.get(name) and eng.get(name):
            report[name] = compare_logits(ref, eng, name, a.tie, off)
    for name in [n for n in ref.layer_names() if n in set(eng.layer_names())]:
        h = compare_hidden(ref, eng, name)
        if h:
            report["hidden"][name] = h
    if "logits" not in report and ref.token_logprobs is not None and eng.token_logprobs is not None:
        check_tokens(ref, eng, min(len(ref.tokens), len(eng.tokens)) - 1)
        report["token_logprobs"] = compare_logprobs(ref, eng)
    if not any(report.get(k) for k in ("logits", "mtp_logits", "hidden", "token_logprobs")):
        print("nothing comparable (no common rows/quantities)")
        sys.exit(2)
    for name in ("logits", "mtp_logits"):
        r = report.get(name)
        if r:
            line = (
                f"{name:10s} rows {r['rows']} [{r['first_row']}..{r['last_row']}]  KL mean {r['kl_mean']:.3e} "
                f"p99 {r['kl_p99']:.3e} max {r['kl_max']:.3e}  top1 {r['top1']:.4f} "
                f"({r['top1_mismatch']} mismatch, {r['top1_decisive_mismatch']} decisive >{a.tie}, "
                f"{r['near_ties']} near-ties)  |dlogit| max {r['dlogit_max']:.3e} mean {r['dlogit_mean']:.3e}"
            )
            if "dnll_mean" in r:
                line += f"  dNLL {r['dnll_mean']:+.3e} PPL ref {r['ppl_ref']:.5f} eng {r['ppl_eng']:.5f}"
            print(line)
            for d in r["decisive"][:5]:
                print(
                    f"    decisive mismatch at pos {d['pos']}: ref {d['ref']} eng {d['eng']} margin {d['ref_margin']}"
                )
    if report.get("token_logprobs"):
        r = report["token_logprobs"]
        print(
            f"token_logprobs scored {r['scored']} dNLL mean {r['dnll_mean']:+.3e} |max| {r['dnll_absmax']:.3e} "
            f"PPL ref {r['ppl_ref']:.5f} eng {r['ppl_eng']:.5f}"
        )
    for name, h in report["hidden"].items():
        print(
            f"{name:13s} rows {h['rows']:5d} rel {h['rel']:.3e} worst-row {h['row_rel_max']:.3e} "
            f"(pos {h['row_rel_argmax_pos']}) cos_min {h['cos_min']:.6f} |d| max {h['abs_max']:.3e} "
            f"mean {h['abs_mean']:.3e} ref_rms {h['ref_rms']:.3e}"
        )
    fails = verdict(report, tol)
    report["fails"] = fails
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    if tol:
        print(("FAIL " + "; ".join(fails)) if fails else f"PASS profile {a.profile} {tol}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
