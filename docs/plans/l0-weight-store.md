# L0 WeightStore Plan

This plan is superseded by
[`l0-weight-store-q5090.md`](l0-weight-store-q5090.md).

The canonical weight ABI is
[`../q5090_packed_file_format_v1.md`](../q5090_packed_file_format_v1.md), produced by
[`../../tools/q5090_convert`](../../tools/q5090_convert). The C++ `WeightStore` consumes q5090
directly, validates model metadata and payload CRCs, and supports module-selective loading for
TEXT, MTP, and vision payloads.
