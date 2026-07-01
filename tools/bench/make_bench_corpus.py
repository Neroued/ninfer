#!/usr/bin/env python3
"""Bake the qus_bench meaningful-token corpus.

qus_bench slices the first P token ids of this corpus to build a prefill of an exact
length P, so the corpus must be real, in-distribution text (not random/dummy tokens) and at
least as long as the largest prefill you want to benchmark.

Two content sources:
  * built-in curated multi-domain prose (Chinese / English / code / math), the default; or
  * one or more `--source-text` files you provide (e.g. a downloaded public-domain book or a
    concatenated document set) for genuinely diverse, very long content.

The chosen text is encoded with a local Hugging Face Qwen3.6 tokenizer WITHOUT the chat template
and WITHOUT special tokens. If it is shorter than `--tokens`, it is tiled (built-in paragraphs are
rotated each cycle to avoid a trivial period-1 loop) and then truncated to exactly `--tokens`
tokens.

`--check` verifies the committed `.ids` against its manifest hash and token count only. It does not
re-tokenize, so a corpus baked from a large downloaded source stays reproducible/CI-verifiable
without committing the source text.

Outputs:
  bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
  bench/fixtures/bench_corpus.manifest.json  provenance (tokenizer id, token count, sha256, source)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

TOKENIZER_MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_TOKENS = 65536

# Curated, in-distribution prose spanning several domains and languages. Order and content are part
# of the corpus contract: changing them changes the committed ids. Repetition to reach --tokens does
# not bias prefill/decode throughput (both are token-count/bandwidth bound, not content dependent),
# so this bank fills long corpora by rotated tiling. For genuinely diverse very long content, pass
# --source-text instead.
PARAGRAPHS: tuple[str, ...] = (
    "周末旅行规划资料：目的地是一个适合亲子散步的湖边小城，市中心到湖区有直达公交，车程大约"
    "四十分钟。周六下午两点以后人流会增加，建议提前购买返程车票，并把儿童水杯、薄外套、充电宝"
    "和少量现金放在随身包里。",
    "天气预报显示周六白天多云，傍晚可能有小雨，湖边风会比城区更明显。行程中可以安排室内书店、"
    "湖边步道和家庭餐厅三个停留点，这样下雨时也能快速调整，不需要临时寻找避雨地点。",
    "餐饮方面，湖区东门附近有一家面馆和一家简餐店，面馆上菜快，适合孩子饿的时候优先选择；简餐店"
    "座位宽敞，但晚餐高峰可能排队二十分钟。预算按交通、饮料、晚餐和临时雨具计算，一家三口控制在"
    "三百元以内比较稳妥。",
    "光合作用是绿色植物把二氧化碳和水，在光照和叶绿素的参与下转化为葡萄糖并释放氧气的过程。它分为"
    "依赖光的反应和不依赖光的碳同化两个阶段，前者在类囊体膜上进行，后者主要在叶绿体基质中完成，"
    "共同支撑了地球上绝大多数生态系统的能量来源。",
    "丝绸之路并不是一条单独的道路，而是横贯欧亚大陆的商路网络。沿途的绿洲城市既是补给点，也是不同"
    "语言、宗教和技术交流的枢纽；除了丝绸，还流通着香料、纸张、玻璃和数学天文知识，深刻影响了沿线"
    "文明的发展节奏。",
    "A single-stream language model runtime spends almost all of its decode time moving weights "
    "from memory, so throughput at batch size one is bounded by memory bandwidth rather than raw "
    "arithmetic. Prefill, by contrast, processes the whole prompt at once and is usually compute "
    "bound, which is why the two phases must be measured separately.",
    "When you profile an inference engine, keep the prompt content fixed and vary only the lengths "
    "you care about. Report prefill tokens per second and decode tokens per second as independent "
    "numbers, because a change that speeds up one phase can easily slow down the other, and an "
    "averaged single figure hides that trade-off.",
    "Distributed systems replace the certainty of a single machine with probabilities. Messages are "
    "delayed, reordered, or lost; clocks drift; and any node can fail at the least convenient moment. "
    "Good designs therefore assume partial failure from the start, favouring idempotent operations, "
    "explicit timeouts, and state that can be reconstructed rather than trusted blindly.",
    "A vegetable garden rewards patience more than effort. Seeds need warmth and consistent moisture "
    "to germinate, seedlings need light and room to breathe, and mature plants need steady feeding "
    "and a watchful eye for pests. The gardener who checks a little every day usually harvests far "
    "more than the one who works furiously once a week.",
    "In economics, opportunity cost is the value of the best alternative you give up when you make a "
    "choice. It reminds us that resources spent on one project are unavailable for another, so a "
    "decision that looks free in cash terms can still be expensive once the road not taken is priced "
    "in honestly.",
    "def moving_average(values, window):\n"
    "    if window <= 0:\n"
    "        raise ValueError(\"window must be positive\")\n"
    "    total = 0.0\n"
    "    result = []\n"
    "    for index, value in enumerate(values):\n"
    "        total += value\n"
    "        if index >= window:\n"
    "            total -= values[index - window]\n"
    "        if index >= window - 1:\n"
    "            result.append(total / window)\n"
    "    return result",
    "template <typename T>\n"
    "T clamp(T value, T low, T high) {\n"
    "    if (value < low) {\n"
    "        return low;\n"
    "    }\n"
    "    if (value > high) {\n"
    "        return high;\n"
    "    }\n"
    "    return value;\n"
    "}",
    "SELECT department_id, AVG(salary) AS avg_salary, COUNT(*) AS headcount\n"
    "FROM employees\n"
    "WHERE hire_date >= '2020-01-01'\n"
    "GROUP BY department_id\n"
    "HAVING COUNT(*) >= 5\n"
    "ORDER BY avg_salary DESC;",
    "Consider a sequence defined by a_1 = 1 and a_{n+1} = a_n + 1 / a_n for n >= 1. Each step adds a "
    "positive amount, so the sequence is strictly increasing. Squaring the recurrence gives "
    "a_{n+1}^2 = a_n^2 + 2 + 1 / a_n^2, hence a_n^2 grows at least linearly and a_n is on the order "
    "of the square root of 2n for large n.",
    "The probability that at least two people in a room of twenty-three share a birthday is greater "
    "than one half. The result feels surprising because we instinctively compare ourselves to one "
    "other person, but there are two hundred fifty-three distinct pairs among twenty-three people, "
    "and it is the number of pairs, not the number of people, that drives the collision.",
    "睡眠不是简单的休息，而是大脑进行记忆巩固和代谢清理的重要阶段。深度睡眠有助于把当天的短期记忆"
    "转为长期记忆，而快速眼动睡眠与情绪调节和创造性联想相关。长期睡眠不足会削弱注意力、判断力和"
    "免疫功能，其代价往往在数周后才显现。",
    "会议纪要：本周完成了基准测试工具的重构评审，确认新的吞吐量工具只负责测速，正确性与逐层对齐仍由"
    "独立的校验工具负责。下一步先固化预填充与解码两个阶段的计时边界，再补充机器可读的输出格式，方便"
    "把历史结果归档和比较。",
    "Reproducibility is the quiet backbone of good benchmarking. Fix the model, the input tokens, the "
    "context length, and the measurement boundary, then record the exact command, the commit, and the "
    "hardware. A number without that context is a rumour; a number with it is evidence others can "
    "check and build on.",
)


def build_text(paragraphs: Sequence[str]) -> str:
    return "\n\n".join(paragraphs) + "\n\n"


def rotate(paragraphs: Sequence[str], by: int) -> list[str]:
    if not paragraphs:
        return []
    k = by % len(paragraphs)
    return list(paragraphs[k:]) + list(paragraphs[:k])


def resolve_tokenizer_path(cli_path: str | None) -> Path:
    raw = cli_path or os.environ.get("QUS_TOKENIZER_PATH")
    if not raw:
        raise SystemExit(
            "tokenizer path required; pass --tokenizer-path or set QUS_TOKENIZER_PATH"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"tokenizer path is not a directory: {path}")
    return path


def load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required; install tools/bench/requirements.txt"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    for value in ids:
        if not isinstance(value, int) or value < 0:
            raise SystemExit(f"tokenizer produced an invalid id: {value!r}")
    return ids


def bake_ids(tokenizer: Any, paragraphs: Sequence[str], tokens: int) -> list[int]:
    """Encode paragraphs, tiling (with rotation) and truncating to exactly `tokens` ids."""
    base_ids = encode(tokenizer, build_text(paragraphs))
    if not base_ids:
        raise SystemExit("source text produced no tokens")
    if len(base_ids) >= tokens:
        return base_ids[:tokens]
    repeats = math.ceil(tokens / len(base_ids)) + 1
    full_text = "".join(build_text(rotate(paragraphs, cycle)) for cycle in range(repeats))
    ids = encode(tokenizer, full_text)
    if len(ids) < tokens:
        raise SystemExit(f"tiled corpus has {len(ids)} tokens, below target {tokens}")
    return ids[:tokens]


def load_source_paragraphs(source_files: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
    paragraphs: list[str] = []
    provenance: list[dict[str, str]] = []
    for raw in source_files:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise SystemExit(f"--source-text is not a file: {path}")
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise SystemExit(f"--source-text file is empty: {path}")
        paragraphs.append(content)
        provenance.append({"name": path.name, "sha256": sha256_text(content)})
    return paragraphs, provenance


def format_ids(ids: Sequence[int], per_line: int = 32) -> str:
    lines = [
        " ".join(str(v) for v in ids[i : i + per_line])
        for i in range(0, len(ids), per_line)
    ]
    return "\n".join(lines) + "\n"


def parse_ids_text(text: str) -> list[int]:
    tokens = text.split()
    if not tokens:
        raise SystemExit("committed .ids is empty")
    ids = []
    for tok in tokens:
        if not tok.isdigit():
            raise SystemExit(f"committed .ids has a non-digit token: {tok!r}")
        ids.append(int(tok))
    return ids


def build_manifest(
    ids: Sequence[int], ids_text: str, tokens: int, source_provenance: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "artifact_type": "qus_bench_corpus",
        "schema_version": 1,
        "tokenizer_source": "local_hf",
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "add_special_tokens": False,
        "chat_template": False,
        "tokens": tokens,
        "token_count": len(ids),
        "ids_sha256": sha256_text(ids_text),
        "source": "source_text" if source_provenance else "builtin_curated_bank",
        "source_files": source_provenance,
        "note": (
            "meaningful tokens; tiled (rotated) and truncated to exactly `tokens`. "
            "Repetition fills length only and does not bias throughput."
        ),
    }


def write_outputs(
    out_path: Path, manifest_path: Path, ids: Sequence[int], tokens: int,
    source_provenance: list[dict[str, str]]
) -> None:
    ids_text = format_ids(ids)
    manifest = build_manifest(ids, ids_text, tokens, source_provenance)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ids_text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def check_outputs(out_path: Path, manifest_path: Path) -> int:
    """Verify the committed .ids against its manifest, without re-tokenizing."""
    if not out_path.exists() or not manifest_path.exists():
        print(f"missing corpus artifact: {out_path} or {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids_text = out_path.read_text(encoding="utf-8")
    ids = parse_ids_text(ids_text)
    failures = 0
    actual_sha = sha256_text(ids_text)
    if actual_sha != manifest.get("ids_sha256"):
        print(f"ids_sha256 mismatch: {actual_sha} != {manifest.get('ids_sha256')}", file=sys.stderr)
        failures += 1
    if len(ids) != manifest.get("token_count"):
        print(
            f"token_count mismatch: {len(ids)} != {manifest.get('token_count')}", file=sys.stderr
        )
        failures += 1
    if failures == 0:
        print(f"corpus OK: {len(ids)} tokens, sha256 {actual_sha}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", default=None)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--out", type=Path, default=repo_root / "bench/fixtures/bench_corpus.ids")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"exact corpus token count (default: {DEFAULT_TOKENS})")
    parser.add_argument("--source-text", action="append", default=[],
                        help="meaningful text file(s) to use instead of the built-in bank; repeatable")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed artifacts against the manifest (no tokenizer needed) instead of writing",
    )
    args = parser.parse_args(argv)
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")

    if args.check:
        return check_outputs(args.out, manifest_path)

    if args.tokens < 1:
        raise SystemExit("--tokens must be positive")
    tokenizer = load_tokenizer(resolve_tokenizer_path(args.tokenizer_path))
    if args.source_text:
        paragraphs, source_provenance = load_source_paragraphs(args.source_text)
    else:
        paragraphs, source_provenance = list(PARAGRAPHS), []
    ids = bake_ids(tokenizer, paragraphs, args.tokens)

    write_outputs(args.out, manifest_path, ids, args.tokens, source_provenance)
    print(f"wrote {args.out} ({len(ids)} tokens) and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
