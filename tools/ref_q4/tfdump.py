# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""tfdump/1 — the teacher-forcing dump format shared by the references and the engine (qwen4_tfdump.h).

A dump is either a DIRECTORY of .npy files or ONE .safetensors file with the same keys:

  meta.json                 {"format": "tfdump/1", "producer": ..., ...}   (safetensors: __metadata__["tfdump"])
  tokens.npy      int64 [T] the whole teacher-forced sequence (prompt + continuation)
  logits.npy      f32 [R,V] row r = logits after consuming tokens[rows[r]] (predicts tokens[rows[r]+1])
  rows.npy        int64 [R] sequence positions of the rows (absent = 0..R-1)
  hidden/L07.npy  f32 [R,hc*H] wide residual (4 streams, stream-major) after decoder layer 7, same rows
  hidden/final.npy f32 [R,H]  output hyper-connection mixer (input of lm_head), same rows
  mtp_logits.npy  f32 [R',V] MTP column at position p predicts tokens[p+2] from main hidden at p and tokens[p+1]
  mtp_rows.npy    int64 [R']
  token_logprobs.npy f32 [T-1] log p(tokens[t+1] | tokens[..t]) (lets long runs skip full logits)

Per-step files are accepted in place of a stacked array, which is what a decode loop emits naturally:
  logits/p000123.npy f32 [V]   hidden/L07/p000123.npy f32 [hc*H]   mtp_logits/p000123.npy f32 [V]
(the number is the sequence position). safetensors keys: "tokens", "logits", "rows", "hidden.L07",
"hidden.final", "mtp_logits", "mtp_rows", "token_logprobs". float16/bfloat16 arrays are accepted and widened.
Requires NumPy (and safetensors for .safetensors dumps).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

FORMAT = "tfdump/1"
_STEP = re.compile(r"^p(\d+)\.npy$")


class Dump:
    """In-memory view: tokens, meta, and per-quantity (rows, array) pairs."""

    def __init__(self, tokens, meta=None):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.meta = dict(meta or {})
        self.meta.setdefault("format", FORMAT)
        self.items: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name -> (rows, data)
        self.token_logprobs: np.ndarray | None = None

    def put(self, name, data, rows=None):
        data = np.asarray(data)
        rows = np.arange(data.shape[0], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        if rows.shape[0] != data.shape[0]:
            raise ValueError(f"{name}: {rows.shape[0]} rows for {data.shape[0]} entries")
        self.items[name] = (rows, data)

    def get(self, name):
        return self.items.get(name)

    def layer_names(self):
        def key(n):
            m = re.match(r"hidden\.L(\d+)$", n)
            return (0, int(m.group(1))) if m else (1, n)

        return sorted((n for n in self.items if n.startswith("hidden.")), key=key)

    # -- writing -------------------------------------------------------------------------------------------------
    def save(self, path, dtype=np.float32):
        path = Path(path)
        if str(path).endswith(".safetensors"):
            from safetensors.numpy import save_file

            tensors = {"tokens": self.tokens}
            if self.token_logprobs is not None:
                tensors["token_logprobs"] = self.token_logprobs.astype(np.float32)
            for name, (rows, data) in self.items.items():
                tensors[name] = np.ascontiguousarray(data.astype(dtype))
                if name == "logits":
                    tensors["rows"] = rows
                elif name == "mtp_logits":
                    tensors["mtp_rows"] = rows
                else:
                    tensors[name + ".rows"] = rows
            save_file(tensors, str(path), metadata={"tfdump": json.dumps(self.meta)})
            return
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps(self.meta, indent=1))
        np.save(path / "tokens.npy", self.tokens)
        if self.token_logprobs is not None:
            np.save(path / "token_logprobs.npy", self.token_logprobs.astype(np.float32))
        for name, (rows, data) in self.items.items():
            if name.startswith("hidden."):
                (path / "hidden").mkdir(exist_ok=True)
                stem = path / "hidden" / name.split(".", 1)[1]
                np.save(f"{stem}.npy", data.astype(dtype))
                np.save(f"{stem}.rows.npy", rows)
            else:
                np.save(path / f"{name}.npy", data.astype(dtype))
                np.save(path / ("rows.npy" if name == "logits" else f"{name.replace('_logits', '')}_rows.npy"), rows)


def _widen(a):
    a = np.asarray(a)
    return a.astype(np.float32) if a.dtype.kind == "f" and a.itemsize < 4 else a


def _steps(directory: Path):
    files = sorted((int(m.group(1)), directory / f) for f in os.listdir(directory) if (m := _STEP.match(f)))
    if not files:
        return None
    rows = np.asarray([p for p, _ in files], dtype=np.int64)
    return rows, np.stack([_widen(np.load(f)) for _, f in files])


def load(path) -> Dump:
    path = Path(path)
    if str(path).endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = json.loads((f.metadata() or {}).get("tfdump", "{}"))
            keys = set(f.keys())
            d = Dump(f.get_tensor("tokens"), meta)
            if "token_logprobs" in keys:
                d.token_logprobs = _widen(f.get_tensor("token_logprobs"))
            for k in keys:
                if k in ("tokens", "token_logprobs", "rows", "mtp_rows") or k.endswith(".rows"):
                    continue
                rows_key = "rows" if k == "logits" else "mtp_rows" if k == "mtp_logits" else k + ".rows"
                rows = f.get_tensor(rows_key) if rows_key in keys else None
                d.put(k, _widen(f.get_tensor(k)), rows)
        return d
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    d = Dump(np.load(path / "tokens.npy"), meta)
    if (path / "token_logprobs.npy").exists():
        d.token_logprobs = _widen(np.load(path / "token_logprobs.npy"))
    for name, rows_file in (("logits", "rows.npy"), ("mtp_logits", "mtp_rows.npy")):
        if (path / f"{name}.npy").exists():
            rows = np.load(path / rows_file) if (path / rows_file).exists() else None
            d.put(name, _widen(np.load(path / f"{name}.npy")), rows)
        elif (path / name).is_dir() and (s := _steps(path / name)):
            d.put(name, s[1], s[0])
    hd = path / "hidden"
    if hd.is_dir():
        for entry in sorted(os.listdir(hd)):
            p = hd / entry
            if p.is_dir():
                if s := _steps(p):
                    d.put("hidden." + entry, s[1], s[0])
            elif entry.endswith(".npy") and not entry.endswith(".rows.npy"):
                stem = entry[:-4]
                rows_file = hd / f"{stem}.rows.npy"
                d.put("hidden." + stem, _widen(np.load(p)), np.load(rows_file) if rows_file.exists() else None)
    return d
