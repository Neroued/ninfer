"""Regenerate chat goldens with HF: --resources DIR --output GOLDEN.json."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).parent / "fixtures"
    source = (fixtures / "chat_template.jinja").read_text()
    data = json.loads((fixtures / "chat-golden.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.resources, local_files_only=True)
    tokenizer.chat_template = source
    for case in data["cases"]:
        reference = case["reference"]
        text = tokenizer.apply_chat_template(
            reference["messages"], tokenize=False, **reference["options"]
        )
        case["expected"] = dict(
            text=text, ids=tokenizer.encode(text, add_special_tokens=False)
        )
    data["source"] = dict(
        url="https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/main/chat_template.jinja",
        sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    data["resources"] = {
        name: hashlib.sha256((args.resources / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    data["reference"] = {
        name: importlib.metadata.version(name)
        for name in ("transformers", "jinja2", "tokenizers")
    }
    header = json.dumps(
        {key: value for key, value in data.items() if key != "cases"}, indent=2
    )
    rows = [
        json.dumps(case, ensure_ascii=False, separators=(",", ":"))
        for case in data["cases"]
    ]
    args.output.write_text(
        header[:-2] + ',\n  "cases": [\n    ' + ",\n    ".join(rows) + "\n  ]\n}\n"
    )
    print(f"Wrote {len(data['cases'])} HF cases to {args.output}")


if __name__ == "__main__":
    main()
