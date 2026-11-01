"""HF reference for prepared prompts; --tokenizer FILE --output FILE selects explicit artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template


def message(role, content, **fields):
    return dict(role=role, content=content, **fields)


def marker(location, count=0, **fields):
    return dict(location=location, after_message_count=count, **fields)


def cases():
    user = message("user", "question")
    system = message("system", "policy")
    yield dict(
        name="instructions",
        messages=[system, user],
        markers=[marker("message", 1), marker("part", 1, after_message_part_count=1)],
        ends=["<|user|>", "<|user|>"],
    )
    yield dict(
        name="part-merge",
        messages=[
            message("user", [dict(type="text", text="a"), dict(type="text", text="bc")])
        ],
        normalized=[message("user", "abc")],
        markers=[marker("part", 1, after_message_part_count=1)],
        ends=[("<|user|>", 1)],
    )
    yield dict(
        name="instruction-merge",
        messages=[message("system", "abc"), user],
        markers=[marker("leading", leading_instruction_bytes=1)],
        ends=[("<|system|>abc", -2)],
    )
    history = [
        system,
        user,
        message("assistant", "answer", reasoning_content="reasoning"),
        message("user", "next"),
    ]
    yield dict(
        name="cleared-history-boundary",
        messages=history,
        options=dict(preserve_thinking=False),
        kwargs=dict(clear_thinking=True),
        markers=[marker("message", 3)],
        ends=["<|user|>next"],
        rewrite="turn",
    )
    calls = [
        dict(id=n, type="function", function=dict(name=n, arguments={}))
        for n in ("A", "B")
    ]
    history = [
        user,
        message("assistant", "", reasoning_content="r", tool_calls=calls),
        message("tool", "result-B", tool_call_id="B"),
        message("tool", "result-A", tool_call_id="A"),
    ]
    for clear in (False, True):
        yield dict(
            name=f"tool-order-{clear}",
            messages=history,
            options=dict(preserve_thinking=not clear),
            kwargs=dict(clear_thinking=clear),
            markers=[
                marker("message", 3),
                marker("message", 4),
                marker("part", 3, after_message_part_count=1),
            ],
            ends=[None, "generation", None],
            rewrite="turn" if clear else "response",
            first_response=clear,
        )
    tools = [
        dict(
            type="function",
            function=dict(
                name=n,
                description="{ braces }",
                parameters=dict(type="object", properties=dict(x=dict(type="string"))),
            ),
        )
        for n in ("A", "B")
    ]
    ends = [(json.dumps(t["function"], ensure_ascii=False), 0) for t in tools]
    yield dict(
        name="tool-declarations",
        messages=[system, user],
        options=dict(tools=tools),
        kwargs=dict(tools=tools),
        markers=[
            marker("tool", after_tool_count=1),
            marker("tool", after_tool_count=2),
        ],
        ends=ends,
    )
    history = [
        user,
        message("assistant", "answer", reasoning_content="r", tool_calls=calls),
        message("user", "next"),
    ]
    yield dict(
        name="assistant-content",
        messages=history,
        markers=[
            marker(
                "content",
                2,
                after_message_part_count=1,
                after_tool_call_count=1,
                reasoning_bytes=1,
            ),
            marker("content", 2, reasoning_bytes=0),
        ],
        ends=[("</tool_call>", 0), None],
    )
    yield dict(
        name="cleared-assistant-content",
        messages=history,
        options=dict(preserve_thinking=False),
        kwargs=dict(clear_thinking=True),
        markers=[
            marker(
                "content",
                2,
                after_message_part_count=1,
                after_tool_call_count=1,
                reasoning_bytes=1,
            )
        ],
        ends=[("</tool_call>", 0)],
        rewrite="turn",
    )
    yield dict(
        name="continuation",
        messages=[user, message("assistant", " answer ", reasoning_content="r")],
        options=dict(continuation=True),
        markers=[marker("content", 2, after_message_part_count=1, reasoning_bytes=1)],
        ends=["end"],
        rewrite="response",
    )
    yield dict(
        name="empty-message",
        messages=[message("system", ""), user],
        markers=[marker("message", 1)],
        ends=["<|user|>"],
    )
    yield dict(
        name="unicode-boundary",
        messages=[message("system", "éx"), user],
        markers=[marker("leading", leading_instruction_bytes=2)],
        ends=[("<|system|>é", 0)],
    )


def byte_alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [n for n in range(256) if n not in visible]
    return {chr(n): n for n in visible} | {
        chr(256 + i): n for i, n in enumerate(missing)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (Path(__file__).parent / "fixtures/chat_template.jinja").read_text()
    template = _compile_jinja_template(source)
    backend = Tokenizer.from_file(str(args.tokenizer))
    resource = json.loads(args.tokenizer.read_text())
    alphabet = byte_alphabet()
    decoded = {
        index: bytes(alphabet[c] for c in word)
        for word, index in resource["model"]["vocab"].items()
    }
    decoded.update({t["id"]: t["content"].encode() for t in resource["added_tokens"]})
    output = []
    for case in cases():
        opts = case.get("options", {})
        continuation = opts.get("continuation", False)
        normalized = case.get("normalized", case["messages"])
        text = template.render(
            messages=normalized,
            add_generation_prompt=not continuation,
            **case.get("kwargs", {}),
        )
        ids = backend.encode(text, add_special_tokens=False).ids
        ends = {0: 0}
        position = 0
        for i, token in enumerate(ids):
            position += len(decoded[token])
            ends[position] = i + 1
        assert b"".join(decoded[t] for t in ids) == text.encode()

        def frontier(end):
            if end is None:
                return None
            if end == "generation":
                offset = text.rindex("<|assistant|>")
            elif end == "end":
                offset = len(text)
            elif isinstance(end, tuple):
                literal, extra = end
                offset = text.index(literal) + len(literal) + extra
            else:
                offset = text.index(end)
            return ends.get(len(text[:offset].encode()))

        # 'part-merge' names the offset after the first literal content byte.
        expected_markers = [frontier(e) for e in case["ends"]]
        start = (
            text.index("<|assistant|>")
            if case.get("first_response")
            else text.rindex("<|assistant|>")
        )
        output.append(
            dict(
                name=case["name"],
                input=dict(
                    messages=case["messages"], options=opts, markers=case["markers"]
                ),
                expected=dict(
                    ids=ids,
                    markers=expected_markers,
                    rewrite=ends[len(text[:start].encode())],
                    kind=case.get("rewrite", "response"),
                ),
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                tokenizer_sha256=hashlib.sha256(
                    args.tokenizer.read_bytes()
                ).hexdigest(),
                cases=output,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(output)} prepared-prompt cases to {args.output}")


if __name__ == "__main__":
    main()
