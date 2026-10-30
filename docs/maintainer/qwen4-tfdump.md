# qwen4_exp teacher-forcing dump (tfdump)

Status: **implemented (v1)** — per-position final-layer hidden state + token + position
capture, default-OFF, env-gated.

## Summary

NInfer computes the final-layer residual (the hidden state fed to the output head) on every
prefill, but discards it after sampling. There is **no way to export it**. This blocks
fine-tuning the native MTP head of a Qwen3.8-Flash-Next (qwen4_exp) artifact against its
served target: training needs per-position final-layer hidden states + tokens + positions
extracted from the *served* (NVFP4) weights.

## Implementation

- **`src/models/qwen4_exp/execution/tf_dump.{h,cpp}`** — a model-agnostic dump writer
  (ported from the qwen3_5 DFlash2 branch's `dflash_tfdump`, simplified: no logits, no
  top-k). Writes the tfdump/1 layout: `hidden_L00.bin` (compact records), `positions.npy`,
  `tokens.npy`, `meta.json`.
- **Wired into `RankProgramImpl::after_main()`**: when the dump is enabled and the frame is
  a prefill with MTP, `f.conditioning` (the `{channels, t, 1}` final-layer residual buffer)
  is D2H-copied to disk; tokens come from `row.ids`; positions are derived from the on-disk
  `.bin` (guaranteed alignment).
- **Default-OFF**: `TfDump::from_environment` returns null unless `NINFER_TFDUMP=<dir>` is
  set; the hooks are null-checked, so a prod lane carries zero overhead and the flag is
  never needed.

## Env

| Variable | Effect |
|---|---|
| `NINFER_TFDUMP=<dir>` | Enable the dump; `<dir>` is the output directory (created if missing). Unset = off. |

## File format (tfdump/1)

| File | Type | Description |
|---|---|---|
| `hidden_L00.bin` | compact binary | Per-record: `{i32 first_position, i32 columns, i32 rows, bf16[rows][columns]}`. Appended per prefill chunk. |
| `positions.npy` | numpy int32 `[T]` | Token positions 0..T-1 (derived from the on-disk `.bin` total column count). |
| `tokens.npy` | numpy int64 `[T]` | Token ids for the prefilled sequence. |
| `meta.json` | JSON | `{"format":"tfdump/1","producer":"ninfer","artifact":"...","model":"qwen4-exp-flashnext"}` |

## Tap point

The tap is the **final-layer residual** (`f.conditioning`), which is the per-position hidden
state after the last transformer layer (before the output head projection). Shape:
`{residual_width, t, 1}` BF16 on device.

## Zero-overhead-when-off guarantee

`TfDump::from_environment()` returns `nullptr` when `NINFER_TFDUMP` is unset. The hook in
`after_main()` is `if (tf_dump && ...)` — a single null pointer check per frame. No
allocations, no CUDA calls, no I/O.

## Non-goals (v1)

- No logits / top-k capture (CE against ground-truth labels is the training plan).
- No multi-layer taps (only the final layer; the qwen4_exp arch has no DFlash2 tap points).
- No per-sequence subdirectories (single flat output; one capture run = one directory).
