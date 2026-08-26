# Serve TTFT corpus

This directory contains the frozen inputs for the external Serve TTFT benchmark. The timed runner
only reads these files, checks their hashes, converts committed media to data URLs, and serializes
requests. It never loads a tokenizer, generates images, resizes media, or calls an input-token
endpoint.

`manifest.json` is the authority for each file path, SHA-256, exact prompt token count, output
limit, shared-prefix frontier, and media envelope fact. Text shapes use the common Qwen3.6-family
frontend resources embedded in every registered artifact.

`long_8k_independent.json` is the cold long arm of `mixed-four`. It has the same 7680-token
prompt size as the seed but diverges at the first system-content token, so the completed seed
cannot turn that arm into a stable-prefix hit.

The generated media consists of 56 deterministic, byte-distinct 1024×1024 PNGs. The two legal
heavy inputs use disjoint sets of 28 images each. Every image expands to a 12 MiB Vision
preprocessing patch tensor (this is media payload, not KV storage), so the files exercise media
live-byte and preprocess-cache policy without creating large HTTP bodies. The 33-image input is an
intentional aggregate Vision-envelope rejection.

Regenerate only when a case contract changes, using a local tokenizer checkout:

```bash
python3 tools/bench/ttft/build_fixtures.py \
  --tokenizer /path/to/local/Qwen3.8-27B/tokenizer
```

Verify committed files without loading a tokenizer:

```bash
python3 tools/bench/ttft/build_fixtures.py --check
```

See [the benchmark contract](../../../tools/bench/ttft/README.md) for formulas, Serve profiles,
and the audited case catalog.
