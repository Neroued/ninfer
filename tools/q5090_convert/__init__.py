"""Offline converter: Qwen3.6-27B bf16 safetensors -> q5090_w4g64_mixed_v2 packed file.

Binary and quantization contract: ../../docs/q5090_packed_file_format_v4.md
"""

__all__ = ["qtypes", "format", "quantize", "packing", "layouts", "tensor_plan"]
