"""Offline converter: Qwen3.6-27B bf16 safetensors -> q5090_w4g64_mixed_v1 packed file.

Binary contract: ../../docs/q5090_packed_file_format_v1.md
Quant policy:    ../../docs/qwen3_6_27b_q5090_final_quant_format_v1.md
"""

__all__ = ["qtypes", "format", "quantize", "packing", "layouts", "tensor_plan"]
