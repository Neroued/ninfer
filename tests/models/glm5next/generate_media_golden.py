"""Independent GLM media reference. Requires the explicit vLLM processor source and PyAV.

Uses the source geometry/sampling functions, torchvision RGB8 resize, PyTorch patchify,
and HF Jinja/tokenizers. NInfer is not loaded by this generator.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import av

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.v2 import functional as F
from transformers.utils.chat_template_utils import _compile_jinja_template

FIXTURES = Path(__file__).parent / "fixtures"


def references(path):
    tree = ast.parse(path.read_text())
    names = {
        "glm_sample_frame_indices",
        "_ceil_to_factor",
        "_fit_aligned_size_within_budget",
        "smart_resize",
        "_get_pad_content_size",
    }
    selected = ast.Module(
        body=[
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        type_ignores=[],
    )
    scope = dict(
        math=math, np=np, GLM_VIDEO_DEFAULT_FPS=2.0, GLM_VIDEO_DEFAULT_MAX_FRAMES=2048
    )
    exec(compile(selected, str(path), "exec"), scope)
    return scope


def patchify(frames, cfg, ref):
    p, m, tp = cfg.get("patch_size", 14), cfg.get("merge_size", 2), 2
    while len(frames) % tp:
        frames.append(frames[-1])
    t, h, w, _ = np.asarray(frames).shape
    lo, hi = [
        cfg[k] * tp * (p * m) ** 2 for k in ("min_image_tokens", "max_image_tokens")
    ]
    ch, cw = ref["smart_resize"](t, h, w, tp, p * m, p * m, lo, hi)
    rh, rw = ref["_get_pad_content_size"](h, w, ch, cw, t * h * w < lo)
    rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    if (rh, rw) != (h, w):
        rgb = F.resize(
            rgb, [rh, rw], interpolation=F.InterpolationMode.BICUBIC, antialias=True
        )
    rgb = F.pad(rgb, [0, 0, cw - rw, ch - rh])
    mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1) * 255.0
    x = (rgb.float() - mean) / std
    gt, gh, gw = t // tp, ch // p, cw // p
    x = x.reshape(gt, tp, 3, gh // m, m, p, gw // m, m, p).permute(
        0, 3, 6, 4, 7, 2, 1, 5, 8
    )
    words = (
        x.contiguous()
        .reshape(-1, 3 * tp * p * p)
        .to(torch.bfloat16)
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )
    return [gt, gh, gw], words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor-source", type=Path, required=True)
    parser.add_argument("--processor-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=FIXTURES / "media-golden.json")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dump-patches", type=Path)
    args = parser.parse_args()
    root = args.output.parent
    root.mkdir(parents=True, exist_ok=True)
    ref = references(args.processor_source)
    config = json.loads(args.processor_config.read_text())
    if args.tokenizer:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    else:
        tok = json.loads((FIXTURES / "prompt-tokenizer.json").read_text())
        next_id = (
            max(
                [
                    *tok["model"]["vocab"].values(),
                    *[t["id"] for t in tok["added_tokens"]],
                ]
            )
            + 1
        )
        for text in (
            "<|begin_of_image|>",
            "<|end_of_image|>",
            "<|image|>",
            "<|begin_of_video|>",
            "<|end_of_video|>",
            "<|video|>",
        ):
            tok["added_tokens"].append(
                dict(
                    id=next_id,
                    content=text,
                    single_word=False,
                    lstrip=False,
                    rstrip=False,
                    normalized=False,
                    special=True,
                )
            )
            next_id += 1
        (root / "media-tokenizer.json").write_text(
            json.dumps(tok, ensure_ascii=False, indent=2) + "\n"
        )
        tokenizer = Tokenizer.from_str(json.dumps(tok))
    template = _compile_jinja_template((FIXTURES / "chat_template.jinja").read_text())
    records = []

    def rgb(h, w, frame=0):
        y, x = np.indices((h, w))
        return np.stack(
            [
                (x * 7 + y * 11 + frame * 37) % 256,
                (x * 3 + y * 17 + frame * 61) % 256,
                (x * 19 + y * 5 + frame * 13) % 256,
            ],
            -1,
        ).astype(np.uint8)

    def add(name, arrays, fps=None, maximum=8000, jpeg=False, rotation=0):
        cfg = copy.deepcopy(config)
        kind = "image" if fps is None else "video"
        cfg[kind + "_processor"]["max_image_tokens"] = maximum
        path = root / (
            name + (".jpg" if jpeg else ".png")
            if fps is None
            else name + (".mov" if rotation else ".mkv")
        )
        if fps is None:
            image = Image.fromarray(arrays[0])
            image.save(path, **(dict(quality=95, subsampling=0) if jpeg else {}))
            decoded = np.array(Image.open(path).convert("RGB"))
            # Exact JPEG decode is covered separately by the shared decoder; this case is constant RGB.
            selected, stamps = [decoded], []
        else:
            t, h, w, _ = np.asarray(arrays).shape
            with av.open(str(path), "w") as container:
                stream = container.add_stream("png" if rotation else "ffv1", rate=fps)
                stream.width, stream.height, stream.pix_fmt = (
                    w,
                    h,
                    "rgb24" if rotation else "bgr0",
                )
                if rotation:
                    stream.set_display_rotation(rotation)
                for array in arrays:
                    for packet in stream.encode(
                        av.VideoFrame.from_ndarray(array, format="rgb24")
                    ):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            indices = ref["glm_sample_frame_indices"](t, fps, t / fps)
            if not indices:
                indices = [0, 0]  # NInfer's documented nonempty short-video rule.
            selected = [
                np.rot90(arrays[i], rotation // 90).copy() if rotation else arrays[i]
                for i in indices
            ]
            stamps = [int(i / fps) for i in indices][::2]
        grid, words = patchify(selected, cfg[kind + "_processor"], ref)
        messages = [
            dict(
                role="user",
                content=[
                    dict(type="text", text="Describe: "),
                    dict(type=kind),
                    dict(type="text", text=" Done."),
                ],
            )
        ]
        rendered = template.render(
            messages=messages, tools=[], add_generation_prompt=True
        )
        length = grid[1] * grid[2] // 4

        def image_block():
            return "<|begin_of_image|>" + "<|image|>" * length + "<|end_of_image|>"

        replacement = (
            image_block()
            if kind == "image"
            else "<|begin_of_video|>"
            + "".join(image_block() + f"{s:.1f} seconds" for s in stamps)
            + "<|end_of_video|>"
        )
        wrapper = f"<|begin_of_{kind}|><|{kind}|><|end_of_{kind}|>"
        rendered = rendered.replace(wrapper, replacement)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        image_id = tokenizer.token_to_id("<|image|>")
        spans = []
        for i, token in enumerate(ids):
            if token != image_id:
                continue
            if spans and spans[-1][0] + spans[-1][1] == i:
                spans[-1][1] += 1
            else:
                spans.append([i, 1])
        records.append(
            dict(
                name=name,
                config=cfg,
                input=dict(messages=messages, options={}),
                media=[dict(message=0, part=1, file=path.name)],
                ids=ids,
                expected=[
                    dict(
                        grid=grid,
                        timestamps=stamps,
                        spans=spans,
                        patch_sha256=hashlib.sha256(words).hexdigest(),
                    )
                ],
            )
        )
        if args.dump_patches:
            args.dump_patches.mkdir(parents=True, exist_ok=True)
            (args.dump_patches / (name + "-reference.bf16")).write_bytes(words)

    add("image-aligned", [rgb(112, 112)])
    add("image-pad", [rgb(57, 83)])
    add("image-upscale", [rgb(9, 13)])
    add("image-downscale", [rgb(127, 193)], maximum=16)
    add("image-jpeg", [np.full((112, 112, 3), 127, np.uint8)], jpeg=True)
    add("video-short", [rgb(56, 84)], fps=30)
    add("video-odd", [rgb(56, 84, i) for i in range(3)], fps=1)
    add("video-dense", [rgb(56, 84, i) for i in range(20)], fps=10)
    add("video-rotated", [rgb(56, 84, i) for i in range(3)], fps=1, rotation=90)
    add("video-linspace", [rgb(28, 28, i) for i in range(31)], fps=Fraction(62, 23))
    # Source template orders tool results by their call IDs, independent of input message order.
    a, b = copy.deepcopy(records[0]), copy.deepcopy(records[1])
    messages = [
        dict(role="user", content="Look"),
        dict(
            role="assistant",
            content="",
            reasoning_content="r",
            tool_calls=[
                dict(id=n, type="function", function=dict(name=n, arguments={}))
                for n in ("A", "B")
            ],
        ),
        dict(role="tool", content=[dict(type="image")], tool_call_id="B"),
        dict(role="tool", content=[dict(type="image")], tool_call_id="A"),
    ]
    rendered = template.render(messages=messages, tools=[], add_generation_prompt=True)
    expected = []
    for original in (a, b):
        item = original["expected"][0]
        length = item["grid"][1] * item["grid"][2] // 4
        wrapper = "<|begin_of_image|><|image|><|end_of_image|>"
        offset = rendered.index(wrapper)
        replacement = "<|begin_of_image|>" + "<|image|>" * length + "<|end_of_image|>"
        item["spans"] = [
            [
                len(
                    tokenizer.encode(
                        rendered[:offset] + "<|begin_of_image|>",
                        add_special_tokens=False,
                    ).ids
                ),
                length,
            ]
        ]
        rendered = rendered[:offset] + replacement + rendered[offset + len(wrapper) :]
        expected.append(item)
    records.append(
        dict(
            name="tool-media-order",
            config=config,
            input=dict(messages=messages, options={}),
            media=[
                dict(message=2, part=0, file=b["media"][0]["file"]),
                dict(message=3, part=0, file=a["media"][0]["file"]),
            ],
            ids=tokenizer.encode(rendered, add_special_tokens=False).ids,
            expected=expected,
        )
    )
    args.output.write_text(
        json.dumps(
            dict(
                reference=dict(
                    processor_sha256=hashlib.sha256(
                        args.processor_source.read_bytes()
                    ).hexdigest(),
                    torch=torch.__version__,
                ),
                cases=records,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
