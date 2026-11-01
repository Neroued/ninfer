"""HF-backed tokenizer qualification; real-resource goldens use explicit resource paths."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
QWEN_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
GLM_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def byte_alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [value for value in range(256) if value not in visible]
    return {value: chr(value) for value in visible} | {
        value: chr(256 + index) for index, value in enumerate(missing)
    }


def synthetic_resource(nfc, pattern, ignore_merges):
    vocab = {symbol: value for value, symbol in byte_alphabet().items()}
    for word in ("ab", "bc", "abc", "12", "123", "45", "456"):
        vocab[word] = len(vocab)
    added = [
        dict(
            id=len(vocab) + i,
            content=word,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=special,
        )
        for i, (word, special) in enumerate(
            (
                ("<mark>", True),
                ("<mark>tail", False),
                ("<think>", False),
                ("</think>", False),
            )
        )
    ]
    return dict(
        version="1.0",
        truncation=None,
        padding=None,
        normalizer={"type": "NFC"} if nfc else None,
        pre_tokenizer={
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pattern},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        post_processor=None,
        decoder={
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        added_tokens=added,
        model=dict(
            type="BPE",
            dropout=None,
            unk_token=None,
            continuing_subword_prefix=None,
            end_of_word_suffix=None,
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=ignore_merges,
            vocab=vocab,
            merges=[
                ["a", "b"],
                ["b", "c"],
                ["a", "bc"],
                ["1", "2"],
                ["12", "3"],
                ["4", "5"],
                ["45", "6"],
            ],
        ),
    )


class Reference:
    def __init__(self, source):
        self.backend = Tokenizer.from_str(json.dumps(source, ensure_ascii=False))
        ordinary = dict(source, added_tokens=[])
        self.ordinary = Tokenizer.from_str(json.dumps(ordinary, ensure_ascii=False))
        alphabet = {symbol: value for value, symbol in byte_alphabet().items()}
        self.bytes = {
            index: bytes(alphabet[c] for c in word)
            for word, index in source["model"]["vocab"].items()
        }
        self.added = sorted(
            source["added_tokens"], key=lambda item: -len(item["content"].encode())
        )
        for item in self.added:
            self.bytes[item["id"]] = item["content"].encode()

    def normalize(self, text):
        normalizer = self.backend.normalizer
        return normalizer.normalize_str(text) if normalizer else text

    def segments(self, case):
        data = case["text"].encode()
        literal = case.get("literal_spans", [])
        if not case.get("parse_added_tokens", True):
            return [(0, len(data), None)]
        result = []
        begin = pos = 0
        while pos < len(data):
            match = next(
                (
                    item
                    for item in self.added
                    if data.startswith(item["content"].encode(), pos)
                    and not any(
                        pos < end and start < pos + len(item["content"].encode())
                        for start, end in literal
                    )
                ),
                None,
            )
            if match is None:
                pos += 1
                continue
            result.append((begin, pos, None))
            end = pos + len(match["content"].encode())
            result.append((pos, end, match["id"]))
            begin = pos = end
        result.append((begin, len(data), None))
        return result

    def evaluate(self, case):
        data = case["text"].encode()
        markers = case.get("boundaries", [])
        ids = []
        boundaries = [None] * len(markers)
        for begin, end, added in self.segments(case):
            base = len(ids)
            if added is not None:
                ids.append(added)
                for i, pos in enumerate(markers):
                    if begin < pos < end:
                        boundaries[i] = dict(exact=None, stable=base)
                    elif pos == end:
                        boundaries[i] = dict(exact=len(ids), stable=len(ids))
                continue
            raw = data[begin:end].decode()
            encoded = self.ordinary.encode(raw, add_special_tokens=False)
            ids.extend(encoded.ids)
            normalized = self.normalize(raw)
            token_ends = []
            cursor = 0
            for token in encoded.ids:
                cursor += len(self.bytes[token])
                token_ends.append(cursor)
            assert (
                b"".join(self.bytes[token] for token in encoded.ids)
                == normalized.encode()
            )
            word_ends = [
                len(normalized[:stop].encode())
                for _, (_, stop) in self.backend.pre_tokenizer.pre_tokenize_str(
                    normalized
                )
            ]
            for i, pos in enumerate(markers):
                if not begin <= pos <= end:
                    continue
                local = pos - begin
                if local == 0 or local == end - begin:
                    frontier = base if local == 0 else len(ids)
                    boundaries[i] = dict(exact=frontier, stable=frontier)
                    continue
                offset = local
                if self.backend.normalizer:
                    left, right = data[begin:pos].decode(), data[pos:end].decode()
                    if self.normalize(left) + self.normalize(right) != normalized:
                        boundaries[i] = dict(exact=None, stable=base)
                        continue
                    offset = len(self.normalize(left).encode())
                count = bisect.bisect_right(token_ends, offset)
                exact = (
                    base + count
                    if offset == 0 or (count and token_ends[count - 1] == offset)
                    else None
                )
                words = [stop for stop in word_ends if stop <= offset]
                stable = (
                    base + bisect.bisect_right(token_ends, words[-1]) if words else base
                )
                boundaries[i] = dict(exact=exact, stable=stable)
        if not case.get("literal_spans") and case.get("parse_added_tokens", True):
            assert (
                ids == self.backend.encode(case["text"], add_special_tokens=False).ids
            )
        return dict(
            ids=ids,
            decoded=self.backend.decode(ids, skip_special_tokens=False),
            skip_special=self.backend.decode(ids, skip_special_tokens=True),
            boundaries=boundaries,
            limited=[ids[:n] for n in case.get("limits", [])],
        )


def cases(source):
    texts = [
        "",
        "Hello, world!",
        "你好，世界！Hello, world!",
        "abc ab bc abcabc",
        "1234567",
        "0 12 123 1234 12345678901234567890",
        "１２３４５６７",
        "١٢٣٤٥٦٧",
        "Ⅻ²½",
        "é",
        "e\u0301",
        "a\u0301b",
        "\u0301a",
        "Å Å A\u030a",
        "가 가",
        "\t\r\n   ",
        "  a   b \n\n",
        "\u00a0x\u2028\u2029",
        " \x85a",
        "👨‍👩‍👧‍👦👍🏽",
        "I'm WE'RE we'd can't 'S 'ſ",
        "给主人留下些什么吧",
        "straße Verhältnis",
        "hältnis",
        " bisschen",
        " администрации",
        " берегу",
        " снег",
        "x\x00y",
        "a_b = 123.456\nprint('你好')",
        '<tool_call>f<arg_key>x</arg_key><arg_value>{"nested":[true,2]}</arg_value></tool_call>',
        "Context 1234567: e\u0301 中文.\n" * 128,
    ]
    result = [dict(text=text, limits=[0, 1, 3, 16, 10000]) for text in texts]
    for text in (
        "abc",
        "1234567",
        "hello world!",
        "é",
        "a\u0301b",
        "e\u0301x",
        "'ſx",
        " \x85a",
    ):
        # Separate source markers also check that normalization cannot be split at the marker.
        for i in range(len(text) + 1):
            result.append(dict(text=text, boundaries=[len(text[:i].encode())]))
    result.append(dict(text="abc", boundaries=[2, 3, 0, 1, 2], literal_spans=[[1, 2]]))
    if source.get("normalizer") is None:
        # Byte-BPE may end a token inside a UTF-8 scalar; HF character offsets cannot expose this.
        result.append(dict(text="\U0010ffff", boundaries=[0, 1, 2, 3, 4]))
    for item in source["added_tokens"]:
        text = item["content"]
        result.extend([dict(text=text), dict(text=text, parse_added_tokens=False)])
    if source["added_tokens"]:
        token = source["added_tokens"][0]["content"]
        text = "é " + token + " ordinary " + token
        start = len("é ".encode())
        end = start + len(token.encode())
        result.append(
            dict(
                text=text,
                literal_spans=[[start, end]],
                boundaries=[start, end, len(text.encode())],
            )
        )
        result.append(dict(text=text, literal_spans=[[start + 1, end - 1]]))
    return result


def provenance(root):
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }


def run_probe(probe, root, inputs, expected):
    process = subprocess.run(
        [str(probe), str(root)],
        input="".join(json.dumps(case, ensure_ascii=False) + "\n" for case in inputs),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [json.loads(line) for line in process.stdout.split("\n") if line]
    assert len(actual) == len(expected), process.stderr
    for index, (case, got, want) in enumerate(zip(inputs, actual, expected)):
        if got != want:
            raise AssertionError(
                json.dumps(
                    dict(index=index, case=case, actual=got, expected=want),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    print(f"{root.name}: {len(inputs)} cases passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--write-golden", action="store_true")
    args = parser.parse_args()
    if (args.golden or args.write_golden) and not args.resources:
        parser.error("golden generation and comparison require --resources")
    if not args.probe and not args.write_golden:
        parser.error("comparison requires --probe; generation requires --write-golden")
    if args.resources:
        if args.write_golden:
            if not args.golden:
                parser.error("--write-golden requires --golden")
            from transformers import AutoTokenizer

            wrapper = AutoTokenizer.from_pretrained(
                args.resources, local_files_only=True
            )
            source = json.loads(wrapper.backend_tokenizer.to_str())
            reference = Reference(source)
            inputs = cases(source)
            for case in inputs:
                assert (
                    wrapper.encode(case["text"], add_special_tokens=False)
                    == reference.backend.encode(
                        case["text"], add_special_tokens=False
                    ).ids
                )
            data = dict(
                resources=provenance(args.resources),
                reference={
                    name: importlib.metadata.version(name)
                    for name in ("tokenizers", "transformers")
                },
                cases=[
                    dict(input=case, expected=reference.evaluate(case))
                    for case in inputs
                ],
            )
            args.golden.parent.mkdir(parents=True, exist_ok=True)
            header = json.dumps(
                {k: v for k, v in data.items() if k != "cases"}, indent=2
            )
            rows = [
                json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                for case in data["cases"]
            ]
            args.golden.write_text(
                header[:-2]
                + ',\n  "cases": [\n    '
                + ",\n    ".join(rows)
                + "\n  ]\n}\n"
            )
            print(f"Wrote {len(inputs)} HF golden cases: {args.golden}")
        elif args.golden:
            data = json.loads(args.golden.read_text())
            assert data["resources"] == provenance(
                args.resources
            ), "golden uses different resources"
        else:
            parser.error("--resources requires --golden")
        if args.probe:
            run_probe(
                args.probe,
                args.resources,
                [case["input"] for case in data["cases"]],
                [case["expected"] for case in data["cases"]],
            )
        return
    temp_root = ROOT / ".agent-tmp" / "tokenizer-oracle"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        for nfc, pattern in ((True, QWEN_SPLIT), (False, GLM_SPLIT)):
            for ignore in (False, True):
                root = Path(directory) / f"{'nfc' if nfc else 'none'}-ignore-{ignore}"
                root.mkdir()
                source = synthetic_resource(nfc, pattern, ignore)
                stored = dict(source)
                config = {}
                if nfc:
                    stored["added_tokens"] = source["added_tokens"][:2]
                    config["added_tokens_decoder"] = {
                        str(item["id"]): {k: v for k, v in item.items() if k != "id"}
                        for item in source["added_tokens"][2:]
                    }
                (root / "tokenizer.json").write_text(
                    json.dumps(stored, ensure_ascii=False)
                )
                (root / "tokenizer_config.json").write_text(json.dumps(config))
                reference = Reference(source)
                inputs = cases(source)
                run_probe(
                    args.probe,
                    root,
                    inputs,
                    [reference.evaluate(case) for case in inputs],
                )


if __name__ == "__main__":
    main()
