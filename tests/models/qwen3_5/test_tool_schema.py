"""Compare generated Qwen arguments with an independent JSON Schema validator."""

import json
import os
from pathlib import Path
import subprocess
import unittest

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
PROBE = Path(
    os.environ.get(
        "NINFER_TOOL_PROBE", ROOT / "build/tests/ninfer_qwen3_5_tool_constraints_test"
    )
)


def schema(value):
    return {
        "type": "object",
        "properties": {"x": value},
        "required": ["x"],
        "additionalProperties": False,
    }


def call(arguments):
    # Independent encoding of the existing template's raw-string / JSON convention.
    params = "".join(
        f"<parameter={key}>\n{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function=call>\n{params}</function>\n</tool_call>"


def probe(cases):
    data = "".join(
        json.dumps({"schema": spec, "candidates": texts}, ensure_ascii=False) + "\n"
        for spec, texts in cases
    )
    result = subprocess.run(
        [str(PROBE), "--probe"],
        input=data,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


class ToolSchema(unittest.TestCase):
    def test_values_and_exact_decode(self):
        strings = [
            "",
            "a",
            "ab",
            "abc",
            "你好",
            "😀",
            "\0",
            " 你\n",
            "apple",
            "a\np",
            "</parameter>",
        ]
        cases = [
            (schema({"type": "number", "exclusiveMinimum": 0.1, "maximum": 0.2}),
             [{"x": x} for x in [0.1, 0.10000000000000002, 0.15, 0.2, 0.21]]),
            (schema({"type": "number", "minimum": 1e-8, "maximum": 2e-8}),
             [{"x": x} for x in [0, 1e-8, 1.5e-8, 2e-8, 3e-8]]),
            (schema({"type": "array", "prefixItems": [{"type": "string"},
                        {"type": "number", "minimum": 0, "maximum": 1}], "minItems": 2, "items": False}),
             [{"x": x} for x in [[], ["x"], ["x", 0.5], ["x", 2], ["x", 0.5, 0]]]),
            (
                schema({"type": "integer", "enum": [1.0, 2.0]}),
                [{"x": value} for value in [0, 1, 2, 3]],
            ),
            (schema({"type": "string"}), [{"x": s} for s in strings]),
            (
                schema({"type": "string", "minLength": 1, "maxLength": 2}),
                [{"x": s} for s in strings],
            ),
            (schema({"type": "string", "pattern": "p"}), [{"x": s} for s in strings]),
            (
                schema(
                    {
                        "type": "string",
                        "pattern": "你|😀",
                        "minLength": 2,
                        "maxLength": 3,
                    }
                ),
                [
                    {"x": s}
                    for s in ["你", "你好", "😀ab", "你abc", "abc", "\n</parameter>"]
                ],
            ),
            (
                schema({"type": "string", "pattern": "^a(b|c){1,2}$"}),
                [{"x": s} for s in ["ab", "abc", "abbc", "zab"]],
            ),
            (
                schema({"type": "string", "pattern": r"^\s$"}),
                [{"x": s} for s in [" ", "\t", "\n", "\u2003", "a"]],
            ),
            (
                schema({"type": "string", "pattern": r"^\S$"}),
                [{"x": s} for s in [" ", "\t", "\n", "\u2003", "a"]],
            ),
            (
                schema({"type": "string", "pattern": r"^[\^\[]$"}),
                [{"x": s} for s in ["^", "[", "a", "]"]],
            ),
            (
                schema({"type": "string", "pattern": r"^\d{2}$"}),
                [{"x": s} for s in ["12", "1", "ab"]],
            ),
            (
                schema({"type": "string", "pattern": r"^\u0041\x00$"}),
                [{"x": s} for s in ["A\0", "A", "B\0"]],
            ),
            (
                schema({"type": "integer", "minimum": -2, "maximum": 2}),
                [{"x": x} for x in [-3, -2, 0, 2, 3]],
            ),
            (
                schema({"type": ["integer", "null"]}),
                [{"x": x} for x in [None, True, 0, 7]],
            ),
            (
                schema(
                    {
                        "type": "object",
                        "properties": {"s": {"type": ["string", "null"]}},
                        "required": ["s"],
                        "additionalProperties": False,
                    }
                ),
                [{"x": {"s": x}} for x in [None, "</parameter>", 'a"\\\n', 7]],
            ),
            (
                schema(
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 2,
                    }
                ),
                [{"x": x} for x in [[], [1], [1, 2], [1, 2, 3]]],
            ),
            (
                schema({"const": {"s": "</parameter>", "v": [1, True]}}),
                [{"x": {"s": "</parameter>", "v": [1, True]}}],
            ),
            (
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}},
                    "required": ["b"],
                    "additionalProperties": False,
                },
                [
                    {},
                    {"b": True},
                    {"a": 3, "b": False},
                    {"a": 3},
                    {"b": True, "unknown": 0},
                ],
            ),
            (
                {
                    **schema({"$ref": "#/$defs/value"}),
                    "$defs": {"value": {"type": "string", "enum": ["a", "你好"]}},
                },
                [{"x": "a"}, {"x": "b"}, {"x": "你好"}],
            ),
        ]
        reports = probe([(s, [call(v) for v in values]) for s, values in cases])
        self.assertEqual(len(cases), len(reports))
        for (spec, values), report in zip(cases, reports):
            with self.subTest(schema=spec):
                self.assertNotIn("error", report)
                expected = [
                    jsonschema.Draft202012Validator(spec).is_valid(v) for v in values
                ]
                self.assertEqual(report["accepted"], expected)
                for value, accepted, decoded in zip(
                    values, expected, report["arguments"]
                ):
                    if accepted:
                        self.assertEqual(decoded, value)

    def test_representation_boundaries(self):
        finite = [
            "0",
            "-9223372036854775808",
            "9223372036854775807",
            "1.25",
            "9.9e307",
            "1e308",
            "1.7976931348623157e308",
            "1e999",
            "1.8e308",
        ]
        cases = [
            (schema({"type": "number"}), [call({"x": x}) for x in finite]),
            (schema({"type": "string"}), [call({"x": "a\n</parameter>b"})]),
            (
                schema({"type": "integer"}),
                [call({"x": str(x)}) for x in [-(2**63), 2**63 - 1, 2**63]],
            ),
        ]
        reports = probe(cases)
        self.assertEqual(reports[0]["accepted"], [True] * 7 + [False, False])
        self.assertEqual(reports[1]["accepted"], [False])
        self.assertEqual(reports[2]["accepted"], [True, True, False])

    def test_prepare_rejections(self):
        specs = [
            schema({"type": ["string", "null"]}),
            schema({"type": ["string", "integer"], "enum": ["1.0", 1.0]}),
            schema(
                {
                    "anyOf": [
                        {"type": "string", "const": "0"},
                        {"type": "integer", "const": -0.0},
                    ]
                }
            ),
            schema({"type": "string", "const": "x\n</parameter>y"}),
            schema({"type": "number", "multipleOf": 1}),
            {"type": "object", "additionalProperties": True},
        ]
        # At least one candidate asks the probe to compile the grammar.
        for report in probe([(s, [call({})]) for s in specs]):
            self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
