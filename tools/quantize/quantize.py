"""Offline quantizer: bf16 safetensors -> stable 4-bit format.

Main tensors -> 4-bit; sensitive tensors -> high precision. Structure stub; no
implementation yet.

See ../../docs/design.md (Weight pipeline) and
../../docs/qwen3.6-27b-architecture.md (offline transforms).
"""

# TODO(impl): load safetensors; quantize main weights to 4-bit; keep sensitive tensors
# (e.g. in_proj_a/b, norms, possibly lm_head/embed) high-precision; write the stable format.


def main() -> None:
    raise NotImplementedError("quantize: structure stub")


if __name__ == "__main__":
    main()
