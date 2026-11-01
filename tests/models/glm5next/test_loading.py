"""Small production conversion consumed by the native GLM definition and rank loader."""

from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.convert.glm5next_fixture import build_checkpoint
from tools.convert.glm5next import build_model
from tools.convert.official_recipes import glm5next
from tools.convert.pipeline import convert
from tools.convert.proposal import add_official_proposal
from tools.convert.recipe import Recipe


def source_oracles(source: Path, destination: Path) -> None:
    tensors = load_file(source / "model.safetensors")

    def save(name: str, value: torch.Tensor) -> None:
        (destination / name).write_bytes(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        )

    prefix = "model.language_model.layers.0.self_attn."
    save(
        "convolution.bin",
        torch.cat(
            [
                tensors[prefix + role + "_conv1d.weight"].squeeze(1)
                for role in ("q", "k", "v")
            ]
        ).T,
    )
    save("embedding.bin", tensors["model.language_model.embed_tokens.weight"])
    save("head.bin", tensors["lm_head.weight"])
    prefix = "model.language_model.layers.1.self_attn."
    for logical, source_name in (
        ("latent", "kv_a_proj_with_mqa.weight"),
        ("index_key", "indexer.wk.weight"),
        ("pool_gate", "indexer.index_kpool_compress_gate"),
        ("query_a", "q_a_proj.weight"),
        ("index_weight", "indexer.weights_proj.weight"),
        ("query_b", "q_b_proj.weight"),
        ("index_query", "indexer.wq_b.weight"),
    ):
        save(logical + ".bin", tensors[prefix + source_name])
    factors = tensors[prefix + "kv_b_proj.weight"].reshape(2, 128, 128)
    save("key_up.bin", factors[:, :64, :])
    save("value_up.bin", factors[:, 64:, :])


def main() -> int:
    torch.set_num_threads(1)
    temporary_root = ROOT / ".agent-tmp/glm5next-loading-test"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        directory = Path(temporary)
        source_path = directory / "source"
        with build_checkpoint(source_path) as source:
            model = build_model(source, components=("text", "mtp", "vision"))
            recipe = Recipe(model)
            glm5next(model, recipe, {"base": source})
            add_official_proposal(recipe)
            convert(
                model,
                recipe,
                directory / "model.ninfer",
                device="cpu",
                rows_per_chunk=128,
            )
        source_oracles(source_path, directory)
        return subprocess.run([sys.argv[1], str(directory), *sys.argv[2:]]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
