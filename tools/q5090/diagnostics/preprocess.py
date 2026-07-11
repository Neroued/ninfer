"""Compare the native C++ preprocessor against the Hugging Face oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tools.q5090.ref.multimodal import Processor, load_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", default="build/src/qus-preprocess")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thinking = not args.no_thinking
    with tempfile.TemporaryDirectory(prefix="qus-preprocess-parity-") as directory:
        metadata = Path(directory) / "cpp.json"
        patches = Path(directory) / "cpp.f32"
        command = [args.cpp, args.weights, args.messages, str(metadata), str(patches)]
        if not thinking:
            command.append("--no-thinking")
        subprocess.run(command, check=True)
        cpp = json.loads(metadata.read_text(encoding="utf-8"))
        cpp_patches = np.fromfile(patches, dtype=np.float32).reshape(cpp["patch_shape"])

    hf = Processor(args.processor).process(load_messages(args.messages), thinking=thinking)
    hf_patches = []
    if hf.pixel_values is not None:
        hf_patches.append(hf.pixel_values.numpy())
    if hf.pixel_values_videos is not None:
        hf_patches.append(hf.pixel_values_videos.numpy())
    expected = np.concatenate(hf_patches) if hf_patches else np.empty((0, 1536), np.float32)
    if cpp_patches.shape != expected.shape:
        raise SystemExit(f"patch shape mismatch: C++ {cpp_patches.shape}, HF {expected.shape}")
    difference = np.abs(cpp_patches - expected)
    report = {
        "input_ids_exact": cpp["input_ids"] == hf.input_ids.tolist(),
        "token_types_exact": cpp["token_types"] == hf.mm_token_type_ids.tolist(),
        "positions_exact": cpp["positions"] == hf.position_ids.tolist(),
        "rope_delta_exact": cpp["rope_delta"] == hf.rope_delta,
        "cpp_rope_delta": cpp["rope_delta"],
        "hf_rope_delta": hf.rope_delta,
        "patch_shape": list(cpp_patches.shape),
        "patch_max_abs": float(difference.max(initial=0.0)),
        "patch_mean_abs": float(difference.mean()) if difference.size else 0.0,
        "patch_rmse": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
        "cpp": cpp,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    exact = all(
        report[key]
        for key in ("input_ids_exact", "token_types_exact", "positions_exact", "rope_delta_exact")
    )
    if not exact:
        raise SystemExit("discrete processor parity failed")


if __name__ == "__main__":
    main()
