"""Freeze GLM output bytes/token IDs using the source template and HF tokenizer."""

import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = _compile_jinja_template(
        (Path(__file__).parent / "fixtures/chat_template.jinja").read_text()
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = []
    for name, content, calls in [
        ("plain", "你好。Answer!", []),
        ("zero", "", [("noop", {})]),
        (
            "typed",
            "Running.",
            [
                (
                    "call",
                    {
                        "s": ' 007 "\\\n你好 ',
                        "n": 7,
                        "b": True,
                        "z": None,
                        "o": {"tag": "</arg_value>", "a": [1, False]},
                        "a": ["x", 2],
                    },
                )
            ],
        ),
        ("multiple", "", [("call", {"s": "true"}), ("call", {"s": "123"})]),
        ("keys", "", [("call", {'a"&键': "line\nend", "": ""})]),
    ]:
        tool_calls = [
            dict(id=str(i), type="function", function=dict(name=n, arguments=v))
            for i, (n, v) in enumerate(calls)
        ]
        messages = [
            dict(role="user", content="question"),
            dict(
                role="assistant",
                content=content,
                reasoning_content="思考 reasoning.",
                tool_calls=tool_calls,
            ),
        ]
        rendered = template.render(
            messages=messages,
            add_generation_prompt=False,
            tools=[],
            clear_thinking=False,
        )
        response = rendered.rsplit("<|assistant|><think>", 1)[1]
        eos = "<|observation|>" if calls else "<|endoftext|>"
        history = messages + [
            dict(role="tool", content="ok", tool_call_id=str(i))
            for i in range(len(calls))
        ]
        history_text = template.render(
            messages=history, add_generation_prompt=True, tools=[], clear_thinking=False
        )
        literal_value_closers = sum(
            (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            ).count("</arg_value>")
            for _, arguments in calls
            for value in arguments.values()
        )
        cases.append(
            dict(
                history_text=history_text,
                literal_value_closers=literal_value_closers,
                history_ids=tokenizer.encode(
                    history_text, add_special_tokens=False
                ).ids,
                name=name,
                wire=response + eos,
                ids=tokenizer.encode(response + eos, add_special_tokens=False).ids,
                reasoning="思考 reasoning.",
                content=content,
                calls=[dict(name=n, arguments=v) for n, v in calls],
            )
        )
    args.output.write_text(
        json.dumps(dict(cases=cases), ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote {len(cases)} HF output cases")


if __name__ == "__main__":
    main()
