from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact.layouts import row_split_geometry
from tools.convert.common.quantize import quantize_and_encode


N = 70
K = 130
FORMATS = {
    "q4": "Q4G64_F16S",
    "q5": "Q5G64_F16S",
    "q6": "Q6G64_F16S",
    "w8g32": "W8G32_F16S",
}


def known_matrix() -> torch.Tensor:
    return torch.tensor(
        [
            [
                (-1.0 if (row + col) & 1 else 1.0)
                * ((((row * 37 + col * 17) % 211) - 105) / 37.0)
                + ((row % 5) - 2) * 0.03125
                for col in range(K)
            ]
            for row in range(N)
        ],
        dtype=torch.float32,
    )


def main() -> None:
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    source = known_matrix()
    for name, format_name in FORMATS.items():
        geometry = row_split_geometry(format_name, (N, K))
        expected_group = 32 if name == "w8g32" else 64
        if (
            geometry.n != N
            or geometry.k != K
            or geometry.k_pad != 256
            or geometry.k_pad // geometry.groups_per_row != expected_group
        ):
            raise RuntimeError(f"unexpected row-split geometry: {geometry}")
        (output_dir / f"{name}.bin").write_bytes(
            quantize_and_encode(source, format_name, device="cpu")
        )


if __name__ == "__main__":
    main()
