"""Exercise the production Python writer through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter


def main() -> int:
    executable = sys.argv[1]
    data = bytes((index * 37 + 11) % 251 for index in range(130 * 130 * 2))
    nf4 = bytearray(8192)
    for row in range(46):
        offset = row // 45 * 4096 + row % 45 * 90
        nf4[offset : offset + 80] = b"\x77" * 80
    nf4.extend(struct.pack("<f", 2.0))
    nvfp4 = b"\x11" * 49152 + b"\x38" * 6144 + struct.pack("<fff", 2.0, 3.0, 5.0)
    fp8 = b"\x38" * 98304 + struct.pack(
        "<6H", 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0
    )
    with tempfile.TemporaryDirectory(prefix="ninfer-writer-interop-") as temporary:
        for label, limit in (("single", 1_000_000), ("sharded", 12288)):
            path = Path(temporary) / f"{label}.ninfer"
            with ArtifactWriter(
                path,
                [
                    TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1"),
                    TensorSpec(
                        "nf4", (46, 160), "nf4_g16_e4m3fn", "row_paged_g16_4k_v1"
                    ),
                    TensorSpec(
                        "nvfp4",
                        (3, 256, 128),
                        "nvfp4",
                        "batched_block_scale_k16_m128x4_v1",
                    ),
                    TensorSpec(
                        "fp8",
                        (3, 128, 256),
                        "fp8_e4m3fn_block128x128_bf16",
                        "block_scale_m128_k128_v1",
                    ),
                ],
                components={"text": {"config": {}}},
                bindings={
                    "whole": {"object": "matrix"},
                    "paged_table": {"object": "nf4"},
                    "nvfp4_bank": {"object": "nvfp4"},
                    "fp8_bank": {"object": "fp8"},
                    "reordered_rows": {
                        "parts": [
                            {"object": "matrix", "range": [16770, 16900]},
                            {"object": "matrix", "range": [130, 260]},
                        ]
                    },
                },
                max_file_bytes=limit,
            ) as writer:
                writer.write_object("matrix", data)
                writer.write_object("nf4", nf4)
                writer.write_object("nvfp4", nvfp4)
                writer.write_object("fp8", fp8)
            result = subprocess.run([executable, "--writer-fixture", str(path)])
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
