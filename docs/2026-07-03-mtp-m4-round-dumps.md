# MTP M4 Round State Dumps

M4 adds an opt-in round dump entry for debugging target/GDN/MTP state at eager MTP round
boundaries. The dump is disabled by default and is enabled from the text CLI with:

```bash
./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --tokenizer /path/to/tokenizer \
  --prompt "hello" \
  --mtp-draft-tokens 5 \
  --mtp-round-dump-dir out/mtp_round_dumps
```

The runtime writes one manifest per completed MTP round:

```text
round_000000_manifest.json
round_000000_gdn_00_conv_slot0.f32
round_000000_gdn_00_ssm_slot0.f32
round_000000_mtp_kv_00_k.f32
round_000000_mtp_kv_00_v.f32
```

For real Qwen3.6-27B runs there are 48 GDN layer pairs and one MTP KV layer pair per round. Files
are contiguous f32 binary payloads. BF16 runtime tensors are converted to f32 during dump so they can
be inspected with NumPy directly. The manifest records the schema version, round index, committed
length, MTP KV host cursor, padded context, layer counts, and file list.

MTP KV tensors are dumped in full `[head_dim,padded_context,num_kv_heads]` layout. The manifest
marks `committed_mtp_kv_positions = [0, committed_length)`,
`active_draft_mtp_kv_positions = [committed_length, mtp_kv_position)`, and
`dumped_mtp_kv_positions = [0, padded_context)`. Parity consumers should mask to the committed
range when comparing committed state with the reference model; draft rows are proposal state and
positions after `mtp_kv_position` are padding/stale rows.

Use strict sequential mode when isolating state-machine bugs from batched-kernel numerical
differences:

```bash
./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --tokenizer /path/to/tokenizer \
  --prompt "hello" \
  --mtp-draft-tokens 5 \
  --mtp-strict-sequential \
  --mtp-round-dump-dir out/mtp_round_dumps_strict
```

`GDN slot 0` in these dumps is the committed state between rounds. Compare GDN slot 0 directly and
compare MTP KV only over the committed range unless intentionally inspecting the active proposal
rows.
