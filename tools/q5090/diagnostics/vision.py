"""Compare q5090 Vision activations with the source BF16 Hugging Face vision tower."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.q5090.ref import RefModel
from tools.q5090.ref.multimodal import Processor, load_messages


CAPTURE_LAYERS = {0, 13, 26}


def load_hf_vision(model_dir: Path):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_config(config.vision_config).to(dtype=torch.bfloat16)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith("model.visual."):
            shards.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(shards.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as source:
            for name in names:
                state[name.removeprefix("model.visual.")] = source.get_tensor(name)
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"HF vision state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    af, ef = actual.float(), expected.float()
    diff = af - ef
    rmse = diff.square().mean().sqrt()
    reference_rms = ef.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(af.flatten(), ef.flatten(), dim=0)
    return {
        "shape": list(actual.shape),
        "rmse": float(rmse),
        "relative_rmse": float(rmse / reference_rms),
        "cosine": float(cosine),
        "actual_norm": float(af.norm()),
        "reference_norm": float(ef.norm()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--q5090-dump")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    processor = Processor(model_dir)
    batch = processor.process(load_messages(args.messages), thinking=args.thinking)
    q5090_captures: dict[str, torch.Tensor] = {}
    with RefModel(args.weights, device=device, compile_codec=True) as model, torch.inference_mode():
        q5090_output = model.encode_vision(
            batch,
            compile_codec=True,
            tap=lambda name, value: q5090_captures.__setitem__(
                name, value.detach().to(device="cpu", dtype=torch.bfloat16)
            ),
        )
        q5090_captures["merger"] = torch.cat(
            [
                value
                for value in (q5090_output.image_embeddings, q5090_output.video_embeddings)
                if value is not None
            ]
        ).detach().to(device="cpu", dtype=torch.bfloat16)
        vision_stats = q5090_output.stats
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_captures: dict[str, torch.Tensor] = {}
    hf = load_hf_vision(model_dir).to(device=device, dtype=torch.bfloat16).eval()
    handles = []
    for layer in sorted(CAPTURE_LAYERS):
        name = f"block_{layer:02d}"
        handles.append(
            hf.blocks[layer].register_forward_hook(
                lambda _module, _inputs, output, name=name: hf_captures.__setitem__(
                    name, output.detach().to(device="cpu", dtype=torch.bfloat16)
                )
            )
        )
    pixels = [value for value in (batch.pixel_values, batch.pixel_values_videos) if value is not None]
    grids = [value for value in (batch.image_grid_thw, batch.video_grid_thw) if value is not None]
    with torch.inference_mode():
        output = hf(
            torch.cat(pixels).to(device=device, dtype=torch.bfloat16),
            grid_thw=torch.cat(grids).to(device=device),
            return_dict=True,
        )
        hf_captures["merger"] = output.pooler_output.detach().to(
            device="cpu", dtype=torch.bfloat16
        )
    for handle in handles:
        handle.remove()

    comparisons = {
        name: metrics(q5090_captures[name], hf_captures[name])
        for name in ("block_00", "block_13", "block_26", "merger")
    }
    report = {
        "format": "q5090_vision_bf16_comparison_v1",
        "weights": str(Path(args.weights).resolve()),
        "model_dir": str(model_dir.resolve()),
        "image_grid_thw": None if batch.image_grid_thw is None else batch.image_grid_thw.tolist(),
        "video_grid_thw": None if batch.video_grid_thw is None else batch.video_grid_thw.tolist(),
        "vision": {
            "images": vision_stats.images,
            "videos": vision_stats.videos,
            "raw_patches": vision_stats.raw_patches,
            "llm_tokens": vision_stats.llm_tokens,
            "attention_pairs": vision_stats.attention_pairs,
        },
        "comparisons": comparisons,
    }
    if args.q5090_dump:
        root = Path(args.q5090_dump)
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for short_name in ("patch_embed", "block_00", "block_13", "block_26", "merger"):
            value = q5090_captures[short_name].float().contiguous()
            file_name = f"{short_name}.f32"
            value.numpy().tofile(root / file_name)
            records.append(
                {
                    "name": f"vision/{short_name}",
                    "file": file_name,
                    "shape": list(value.shape),
                    "source_dtype": "bf16",
                    "phase": "prefill",
                    "step": 0,
                    "chunk": 0,
                    "position_begin": 0,
                    "position_end": int(value.shape[0]),
                }
            )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "ninfer_activation_dump_v1",
                    "runtime": "python-q5090",
                    "tensors": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
