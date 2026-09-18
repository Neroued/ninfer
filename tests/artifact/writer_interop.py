"""Exercise the production Python writer through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter


def main() -> int:
    executable = sys.argv[1]
    data = bytes((index * 37 + 11) % 251 for index in range(130 * 130 * 2))
    with tempfile.TemporaryDirectory(prefix="ninfer-writer-interop-") as temporary:
        for label, limit in (("single", 1_000_000), ("sharded", 12288)):
            path = Path(temporary) / f"{label}.ninfer"
            with ArtifactWriter(
                path,
                [TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1")],
                components={"text": {"config": {}}},
                bindings={
                    "whole": {"object": "matrix"},
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
            result = subprocess.run([executable, "--writer-fixture", str(path)])
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
