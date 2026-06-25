"""Offline packer: stable 4-bit -> kernel-optimal fixed weight file.

Emits the one file the C++ runtime loads. Structure stub; no implementation yet.

Bakes the offline transforms (see ../../docs/qwen3.6-27b-architecture.md §11/§13):
  - +1 into all RMSNorm weights EXCEPT linear_attn.norm
  - precompute A = -exp(A_log)
  - GDN V-head relayout; squeeze conv1d weight
  - optional projection fusions (in_proj_qkv|z, in_proj_a|b, gate|up)
and writes the header (magic/version/dims/tensor table) + packed tensor blobs.
File format spec: ../weight_format.md (the Python <-> C++ contract).
"""

# TODO(impl): write header + kernel-optimal packed tensor blobs.


def main() -> None:
    raise NotImplementedError("pack: structure stub")


if __name__ == "__main__":
    main()
