#!/usr/bin/env python3
"""Self-contained PyTorch oracle for the qwen3.6-ultraspeed M2 schedule.

The oracle reads our q5090 file directly and dequantizes weights through
tools.q5090_convert.layouts.decode_tensor. It is deliberately correctness-first:
weights stream layer-by-layer, activations are rounded to bf16 after each
L1-equivalent op, and greedy decode is implemented without tokenizer dependencies.
"""

from __future__ import annotations

import argparse
import mmap
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.layouts import decode_tensor

torch.set_grad_enabled(False)


D = 5120
L = 64
I = 17408
V = 248320
H_Q = 24
H_KV = 4
DH = 256
Q_SIZE = H_Q * DH
KV_SIZE = H_KV * DH
GDN_HK = 16
GDN_HV = 48
GDN_DK = 128
GDN_DV = 128
GDN_KEY = GDN_HK * GDN_DK
GDN_VALUE = GDN_HV * GDN_DV
GDN_CONV = 2 * GDN_KEY + GDN_VALUE
GDN_GROUP = GDN_HV // GDN_HK
EPS = 1.0e-6
ROPE_THETA = 1.0e7
ROTARY_DIM = 64
ATTN_SCALE = 1.0 / (DH**0.5)
GDN_SCALE = 1.0 / (GDN_DK**0.5)


def is_full(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def full_idx(layer: int) -> int:
    return (layer + 1) // 4 - 1


def gdn_idx(layer: int) -> int:
    return layer - full_idx(layer) - 1 if is_full(layer) else layer - ((layer + 1) // 4)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return bf16(x.float() @ w.float().t())


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    unit_offset: bool = True,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + EPS)
    wf = weight.float()
    if unit_offset:
        wf = wf + 1.0
    out = xf * inv * wf
    if z is not None:
        out = out * F.silu(z.float())
    return bf16(out)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.sum(xf * xf, dim=-1, keepdim=True) + EPS)
    return bf16(xf * inv)


def residual_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return bf16(x.float() + y.float())


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return bf16(F.silu(gate.float()) * up.float())


def sigmoid_gate_mul(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return bf16(torch.sigmoid(gate.float()) * x.float())


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q [T,24,256], k [T,4,256], partial NeoX split over dims [0:32] and [32:64].
    device = q.device
    half = ROTARY_DIM // 2
    pair = torch.arange(half, device=device, dtype=torch.float32)
    freq = torch.pow(torch.tensor(ROPE_THETA, device=device, dtype=torch.float32), -2.0 * pair / ROTARY_DIM)
    angle = positions.to(device=device, dtype=torch.float32)[:, None] * freq[None, :]
    cos = torch.cos(angle)[:, None, :]
    sin = torch.sin(angle)[:, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        x1 = x[:, :, :half].float()
        x2 = x[:, :, half:ROTARY_DIM].float()
        y[:, :, :half] = bf16(x1 * cos - x2 * sin)
        y[:, :, half:ROTARY_DIM] = bf16(x2 * cos + x1 * sin)
        return y

    return rotate(q), rotate(k)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    # x [T,C], weight decoded as [C,1,4] or [C,4], state [C,3] oldest -> newest.
    T, C = x.shape
    w = weight.reshape(C, 4).float()
    old = state.clone()
    outs = []
    for t in range(T):
        x0 = x[t - 3] if t >= 3 else old[:, t]
        x1 = x[t - 2] if t >= 2 else old[:, t + 1]
        x2 = x[t - 1] if t >= 1 else old[:, t + 2]
        x3 = x[t]
        acc = w[:, 0] * x0.float() + w[:, 1] * x1.float() + w[:, 2] * x2.float() + w[:, 3] * x3.float()
        outs.append(bf16(F.silu(acc)))
    for s in range(3):
        seq_pos = T + s
        if seq_pos == 0:
            state[:, s] = old[:, 0]
        elif seq_pos == 1:
            state[:, s] = old[:, 1]
        elif seq_pos == 2:
            state[:, s] = old[:, 2]
        else:
            state[:, s] = x[seq_pos - 3]
    return torch.stack(outs, dim=0)


def gdn_gating(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sp = torch.where(a.float() + dt_bias.float() > 20.0, a.float() + dt_bias.float(), F.softplus(a.float() + dt_bias.float()))
    g = -torch.exp(a_log.float()) * sp
    beta = torch.sigmoid(b.float())
    return g.float(), beta.float()


@dataclass
class Entry:
    name: str
    qtype: int
    layout: int
    shape: List[int]
    padded_shape: List[int]
    payload_offset: int
    payload_bytes: int


class Q5090File:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        hdr = fmt.unpack_header(self._mm[: fmt.HEADER_SIZE])
        if hdr["magic"] != fmt.MAGIC:
            raise ValueError(f"bad q5090 magic in {self.path}")
        entries = []
        off = hdr["tensor_index_offset"]
        for _ in range(hdr["tensor_count"]):
            entries.append(fmt.unpack_tensor_entry(self._mm[off : off + fmt.TENSOR_ENTRY_SIZE]))
            off += fmt.TENSOR_ENTRY_SIZE
        table = self._mm[hdr["string_table_offset"] : hdr["string_table_offset"] + hdr["string_table_bytes"]]
        self.entries: Dict[str, Entry] = {}
        for e in entries:
            name = table[e["name_offset"] : e["name_offset"] + e["name_len"]].decode("utf-8")
            self.entries[name] = Entry(
                name=name,
                qtype=e["qtype"],
                layout=e["layout"],
                shape=e["shape"],
                padded_shape=e["padded_shape"],
                payload_offset=e["payload_offset"],
                payload_bytes=e["payload_bytes"],
            )

    def tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        e = self.entries[name]
        payload = self._mm[e.payload_offset : e.payload_offset + e.payload_bytes]
        return decode_tensor(payload, e.qtype, e.layout, e.shape, e.padded_shape, device=device).float()

    def row_grouped_rows(self, name: str, rows: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        e = self.entries[name]
        if e.layout != qt.LAYOUT_ROW_GROUPED_G64:
            raise ValueError(f"{name} is not ROW_GROUPED_G64")
        if e.qtype not in qt.QUANT_SPECS:
            raise ValueError(f"{name} is not a low-bit row-grouped tensor")
        spec = qt.QUANT_SPECS[e.qtype]
        n, k = e.shape
        _, kp = e.padded_shape
        kg = kp // spec.group_size
        roww = 2 + spec.bytes_per_group
        out = []
        for row in rows.detach().cpu().tolist():
            if row < 0 or row >= n:
                raise IndexError(f"{name} row out of range: {row}")
            offset = e.payload_offset + row * kg * roww
            payload = self._mm[offset : offset + kg * roww]
            out.append(decode_tensor(payload, e.qtype, e.layout, [1, k], [1, kp], device=device))
        return torch.cat(out, dim=0).float()

    def tiled_row_chunks(
        self,
        name: str,
        device: torch.device | str,
        rows_per_chunk: int = 8192,
    ):
        e = self.entries[name]
        if e.layout not in (qt.LAYOUT_TILE_N64_K64, qt.LAYOUT_TILE_N64_K128):
            raise ValueError(f"{name} is not a tiled tensor")
        n, k = e.shape
        np_, kp = e.padded_shape
        if rows_per_chunk % 64 != 0:
            raise ValueError("rows_per_chunk must be a multiple of 64")
        spec = qt.QUANT_SPECS[e.qtype]
        kg = kp // spec.group_size
        tilew = 64 * 2 + 64 * spec.bytes_per_group
        for row0 in range(0, n, rows_per_chunk):
            row1 = min(n, row0 + rows_per_chunk)
            tile0 = row0 // 64
            tile_rows = ((row1 - row0 + 63) // 64) * 64
            tiles = tile_rows // 64
            offset = e.payload_offset + tile0 * kg * tilew
            payload = self._mm[offset : offset + tiles * kg * tilew]
            chunk = decode_tensor(
                payload,
                e.qtype,
                e.layout,
                [row1 - row0, k],
                [tile_rows, kp],
                device=device,
            ).float()
            yield row0, row1, chunk

    def close(self) -> None:
        self._mm.close()
        self._fh.close()


class RefModel:
    def __init__(self, weights: str | Path, device: str = "cuda", cache_globals: bool = False):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python has CPU-only PyTorch. "
                "Run with a CUDA torch environment, for example "
                "/home/neroued/miniconda3/envs/vllm-bench/bin/python, "
                "or pass --device cpu explicitly."
            )
        self.device = torch.device(device)
        self.q5090 = Q5090File(weights)
        self.cache_globals = cache_globals
        self._globals: Dict[str, torch.Tensor] = {}
        self.reset_state()

    def weight(self, name: str) -> torch.Tensor:
        if self.cache_globals and (name.startswith("model.language_model.embed_tokens") or name in {"lm_head.weight", "model.language_model.norm.weight"}):
            if name not in self._globals:
                self._globals[name] = self.q5090.tensor(name, self.device)
            return self._globals[name]
        return self.q5090.tensor(name, self.device)

    def reset_state(self) -> None:
        self.kv: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.conv: Dict[int, torch.Tensor] = {
            i: torch.zeros(GDN_CONV, 3, device=self.device, dtype=torch.float32) for i in range(48)
        }
        self.ssm: Dict[int, torch.Tensor] = {
            i: torch.zeros(GDN_HV, GDN_DV, GDN_DK, device=self.device, dtype=torch.float32) for i in range(48)
        }
        self.pos = 0

    def embed(self, ids: Iterable[int]) -> torch.Tensor:
        idx = torch.tensor(list(ids), device=self.device, dtype=torch.long)
        if not self.cache_globals:
            return bf16(
                self.q5090.row_grouped_rows(
                    "model.language_model.embed_tokens.weight", idx, self.device
                )
            )
        return bf16(self.weight("model.language_model.embed_tokens.weight").index_select(0, idx))

    def gqa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        fidx: int,
        phase: str,
    ) -> torch.Tensor:
        T = q.shape[0]
        if phase == "prefill":
            self.kv[fidx] = (k.clone(), v.clone())
            k_all, v_all = self.kv[fidx]
            k_rep = k_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            v_rep = v_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            scores = torch.einsum("thd,shd->ths", q.float(), k_rep) * ATTN_SCALE
            t_idx = torch.arange(T, device=self.device)
            s_idx = torch.arange(k_all.shape[0], device=self.device)
            scores = scores.masked_fill(s_idx[None, None, :] > t_idx[:, None, None], -torch.inf)
            probs = torch.softmax(scores, dim=-1)
            return bf16(torch.einsum("ths,shd->thd", probs, v_rep))
        else:
            old_k, old_v = self.kv.get(
                fidx,
                (
                    torch.empty(0, H_KV, DH, device=self.device),
                    torch.empty(0, H_KV, DH, device=self.device),
                ),
            )
            self.kv[fidx] = (torch.cat([old_k, k], dim=0), torch.cat([old_v, v], dim=0))
            k_all, v_all = self.kv[fidx]
            k_rep = k_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            v_rep = v_all.repeat_interleave(H_Q // H_KV, dim=1).float()
            scores = torch.einsum("thd,shd->ths", q.float(), k_rep) * ATTN_SCALE
            probs = torch.softmax(scores, dim=-1)
            return bf16(torch.einsum("ths,shd->thd", probs, v_rep))

    def gdn_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        gidx: int,
    ) -> torch.Tensor:
        T = q.shape[0]
        state = self.ssm[gidx]
        out = torch.empty(T, GDN_HV, GDN_DV, device=self.device, dtype=torch.float32)
        hv_to_hk = torch.arange(GDN_HV, device=self.device, dtype=torch.long) // GDN_GROUP
        for t in range(T):
            kt = k[t].float().index_select(0, hv_to_hk)
            qt = q[t].float().index_select(0, hv_to_hk)
            state.mul_(torch.exp(g[t].float()).view(GDN_HV, 1, 1))
            sk = torch.bmm(state, kt.unsqueeze(-1)).squeeze(-1)
            delta = beta[t].float().unsqueeze(-1) * (v[t].float() - sk)
            state.add_(delta.unsqueeze(-1) * kt.unsqueeze(1))
            out[t] = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1) * GDN_SCALE
        return bf16(out)

    def attn_mix(self, layer: int, x: torch.Tensor, phase: str, positions: torch.Tensor) -> torch.Tensor:
        p = f"model.language_model.layers.{layer}."
        h = rmsnorm(x, self.weight(p + "input_layernorm.weight"), unit_offset=True)
        q = linear(h, self.weight(p + "self_attn.q_proj.q")).reshape(-1, H_Q, DH)
        gate = linear(h, self.weight(p + "self_attn.q_proj.gate")).reshape(-1, H_Q, DH)
        k = linear(h, self.weight(p + "self_attn.k_proj.weight")).reshape(-1, H_KV, DH)
        v = linear(h, self.weight(p + "self_attn.v_proj.weight")).reshape(-1, H_KV, DH)
        qn = rmsnorm(q, self.weight(p + "self_attn.q_norm.weight"), unit_offset=True)
        kn = rmsnorm(k, self.weight(p + "self_attn.k_norm.weight"), unit_offset=True)
        qn, kn = apply_rope(qn, kn, positions)
        a = self.gqa_attention(qn, kn, v, full_idx(layer), phase)
        a = sigmoid_gate_mul(gate, a).reshape(-1, Q_SIZE)
        o = linear(a, self.weight(p + "self_attn.o_proj.weight"))
        return residual_add(x, o)

    def gdn_mix(self, layer: int, x: torch.Tensor, phase: str) -> torch.Tensor:
        p = f"model.language_model.layers.{layer}."
        gidx = gdn_idx(layer)
        h = rmsnorm(x, self.weight(p + "input_layernorm.weight"), unit_offset=True)
        q = linear(h, self.weight(p + "linear_attn.in_proj_qkv.q"))
        k = linear(h, self.weight(p + "linear_attn.in_proj_qkv.k"))
        v = linear(h, self.weight(p + "linear_attn.in_proj_qkv.v"))
        qkv = torch.cat([q, k, v], dim=-1)
        a = linear(h, self.weight(p + "linear_attn.in_proj_a.weight"))
        b = linear(h, self.weight(p + "linear_attn.in_proj_b.weight"))
        qkv_c = causal_conv1d(qkv, self.weight(p + "linear_attn.conv1d.weight"), self.conv[gidx])
        g, beta = gdn_gating(a, b, self.weight(p + "linear_attn.A_log"), self.weight(p + "linear_attn.dt_bias"))
        qc = qkv_c[:, :GDN_KEY].reshape(-1, GDN_HK, GDN_DK)
        kc = qkv_c[:, GDN_KEY : 2 * GDN_KEY].reshape(-1, GDN_HK, GDN_DK)
        vc = qkv_c[:, 2 * GDN_KEY :].reshape(-1, GDN_HV, GDN_DV)
        qn = l2norm(qc)
        kn = l2norm(kc)
        o = self.gdn_recurrent(qn, kn, vc, g, beta, gidx)
        z = linear(h, self.weight(p + "linear_attn.in_proj_z.weight")).reshape(-1, GDN_HV, GDN_DV)
        on = rmsnorm(o, self.weight(p + "linear_attn.norm.weight"), unit_offset=False, z=z)
        out = linear(on.reshape(-1, GDN_VALUE), self.weight(p + "linear_attn.out_proj.weight"))
        return residual_add(x, out)

    def mlp_tail(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"model.language_model.layers.{layer}."
        h = rmsnorm(x, self.weight(p + "post_attention_layernorm.weight"), unit_offset=True)
        gate = linear(h, self.weight(p + "mlp.gate_proj.weight"))
        up = linear(h, self.weight(p + "mlp.up_proj.weight"))
        a = silu_and_mul(gate, up)
        d = linear(a, self.weight(p + "mlp.down_proj.weight"))
        return residual_add(x, d)

    def block(self, layer: int, x: torch.Tensor, phase: str, positions: torch.Tensor) -> torch.Tensor:
        if is_full(layer):
            x = self.attn_mix(layer, x, phase, positions)
        else:
            x = self.gdn_mix(layer, x, phase)
        return self.mlp_tail(layer, x)

    def run_layers(
        self,
        x: torch.Tensor,
        phase: str,
        positions: torch.Tensor,
        dumps: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        for layer in range(L):
            x = self.block(layer, x, phase, positions)
            if dumps is not None:
                dumps[f"layer_{layer}"] = x.detach().float().cpu()
        return x

    def logits_last(self, x: torch.Tensor) -> torch.Tensor:
        xf = rmsnorm(x, self.weight("model.language_model.norm.weight"), unit_offset=True)
        if self.cache_globals:
            return linear(xf[-1:, :], self.weight("lm_head.weight"))[0]

        xlast = xf[-1:, :].float()
        logits = torch.empty(V, device=self.device, dtype=torch.float32)
        for row0, row1, weight in self.q5090.tiled_row_chunks("lm_head.weight", self.device):
            logits[row0:row1] = (xlast @ weight.float().t())[0]
            del weight
        return bf16(logits)

    def forward(
        self,
        prompt: Iterable[int],
        n_decode: int,
        *,
        dumps: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[int]:
        prompt_ids = list(prompt)
        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        if n_decode < 0:
            raise ValueError("n_decode must be nonnegative")
        self.reset_state()
        if n_decode == 0:
            return []

        x = self.embed(prompt_ids)
        pos = torch.arange(len(prompt_ids), device=self.device, dtype=torch.int32)
        x = self.run_layers(x, "prefill", pos, dumps)
        token = int(torch.argmax(self.logits_last(x)).item())
        out = [token]
        self.pos = len(prompt_ids)

        while len(out) < n_decode:
            x = self.embed([token])
            pos = torch.tensor([self.pos], device=self.device, dtype=torch.int32)
            x = self.run_layers(x, "decode", pos, dumps)
            token = int(torch.argmax(self.logits_last(x)).item())
            out.append(token)
            self.pos += 1
        return out


def parse_prompt(text: str) -> List[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def main() -> None:
    ap = argparse.ArgumentParser(description="q5090 PyTorch oracle")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", required=True, help="comma or space separated token ids")
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-cache-globals", action="store_true")
    ap.add_argument("--cache-globals", action="store_true")
    args = ap.parse_args()

    model = RefModel(
        args.weights,
        device=args.device,
        cache_globals=args.cache_globals and not args.no_cache_globals,
    )
    with torch.inference_mode():
        tokens = model.forward(parse_prompt(args.prompt), args.decode)
    print(" ".join(str(t) for t in tokens))


if __name__ == "__main__":
    main()
