"""Independent schema and value oracle for GLM's raw-string/JSON tool encoding."""

import json
import os
import subprocess
import unittest

import jsonschema


def schema(value):
    return {
        "type": "object",
        "properties": {"x": value},
        "required": ["x"],
        "additionalProperties": False,
    }


def call(arguments):
    return (
        "<tool_call>call"
        + "".join(
            f"<arg_key>{key}</arg_key><arg_value>{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}</arg_value>"
            for key, value in arguments.items()
        )
        + "</tool_call>"
    )


def probe(cases):
    payload = "".join(
        json.dumps(dict(schema=spec, candidates=values), ensure_ascii=False) + "\n"
        for spec, values in cases
    )
    result = subprocess.run(
        [os.environ["NINFER_GLM_TOOL_PROBE"], "--probe"],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


class GLMToolSchema(unittest.TestCase):
    def test_schema_and_values(self):
        strings = [
            "",
            "a",
            "ab",
            "abc",
            "abcd",
            "你好",
            "😀",
            " a ",
            "a\nb",
            "a\\b",
            '"quoted"',
            "\0",
        ]
        cases = [
            (schema({"type": "string"}), [{"x": s} for s in strings]),
            (
                schema({"type": "string", "minLength": 1, "maxLength": 2}),
                [{"x": s} for s in strings],
            ),
            (
                schema(
                    {"type": "string", "pattern": "a", "minLength": 2, "maxLength": 3}
                ),
                [{"x": s} for s in strings],
            ),
            (
                schema({"type": "string", "const": " a "}),
                [{"x": x} for x in ["a", " a ", "  a  "]],
            ),
            (
                schema({"type": "integer", "minimum": -2, "maximum": 2}),
                [{"x": n} for n in [-3, -2, 0, 2, 3]],
            ),
            (
                schema({"type": "number", "exclusiveMinimum": 0.1, "maximum": 0.2}),
                [{"x": n} for n in [0.1, 0.15, 0.2, 0.21]],
            ),
            (schema({"type": "boolean"}), [{"x": x} for x in [False, True, None, 1]]),
            (
                schema(
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 2,
                    }
                ),
                [{"x": x} for x in [[], [1], [1, 2], [1, 2, 3], [True]]],
            ),
            (
                schema(
                    {
                        "type": "object",
                        "properties": {"tag": {"type": "string"}},
                        "required": ["tag"],
                        "additionalProperties": False,
                    }
                ),
                [
                    {"x": x}
                    for x in [{}, {"tag": "</arg_value>"}, {"tag": "a", "other": 1}]
                ],
            ),
            (
                schema({"type": "string", "enum": ["你好", '&"']}),
                [{"x": s} for s in ["你好", '&"', "other"]],
            ),
            (
                {
                    "type": "object",
                    "properties": {
                        'a"&键': {"type": "string"},
                        "": {"type": "integer"},
                    },
                    "required": ['a"&键'],
                    "additionalProperties": False,
                },
                [{'a"&键': "v"}, {'a"&键': "v", "": 2}, {}],
            ),
            (
                {
                    **schema({"$ref": "#/$defs/value"}),
                    "$defs": {"value": {"type": "string", "enum": ["a", "你好"]}},
                },
                [{"x": v} for v in ["a", "b", "你好"]],
            ),
        ]
        reports = probe([(s, [call(v) for v in values]) for s, values in cases])
        self.assertEqual(len(reports), len(cases))
        for (spec, values), report in zip(cases, reports):
            with self.subTest(schema=spec):
                self.assertNotIn("error", report)
                for value, result in zip(values, report["results"]):
                    valid = jsonschema.Draft202012Validator(spec).is_valid(value)
                    self.assertEqual(result["accepted"], valid, (spec, value))
                    if valid:
                        self.assertEqual(result["arguments"], value)
                        jsonschema.validate(result["arguments"], spec)

    def test_representation(self):
        reports = probe(
            [
                (
                    schema({"type": "number"}),
                    [call({"x": v}) for v in ["1e308", "1e999", "1.8e308"]],
                ),
                (schema({"type": "string"}), [call({"x": "a</arg_value>b"})]),
                (schema({"type": "string", "const": "</arg_value>"}), [call({})]),
                (schema({"type": ["string", "null"]}), [call({})]),
            ]
        )
        self.assertEqual(
            [v["accepted"] for v in reports[0]["results"]], [True, False, False]
        )
        self.assertFalse(reports[1]["results"][0]["accepted"])
        self.assertIn("error", reports[2])
        self.assertIn("error", reports[3])


if __name__ == "__main__":
    unittest.main()
