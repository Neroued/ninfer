"""Independent JSON Schema oracle for the native compiler/matcher (no model inference)."""

import json
import math
import os
import random
from decimal import Decimal
from pathlib import Path
import subprocess
import unittest

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(os.environ.get("NINFER_SCHEMA_PROBE", ROOT / "build/tests/ninfer_json_schema_test"))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class SchemaContracts(unittest.TestCase):
    def run_cases(self, cases):
        payload = "".join(compact({"schema": schema, "candidates": [compact(v) for v in values]}) + "\n"
                          for schema, values in cases)
        result = subprocess.run([str(PROBE), "--probe"], input=payload, text=True,
                                capture_output=True, check=True, timeout=90)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(reports), len(cases))
        for (schema, values), report in zip(cases, reports):
            with self.subTest(schema=schema):
                self.assertNotIn("error", report, report)
                validator = jsonschema.validators.validator_for(schema)(schema)
                expected = [validator.is_valid(value) for value in values]
                self.assertEqual(report["accepted"], expected, (schema, values, report, expected))

    def test_values_against_independent_validator(self):
        scalars = [None, False, True, -2, -1, 0, 1, 2, 3, 7, 8, "", "a", "ab", [], {}]
        strings = ["", "a", "ab", "abc", "你", "你好", "😀", "\n", "\t", "\0", '"', "\\", "a\n", "\r\b\f"]
        objects = [{}, {"n": 1}, {"n": "1"}, {"n": 2, "s": "x"}, {"s": "x"}, {"n": 1, "extra": 0}]
        cases = [
            ({}, scalars),
            ({"type": "integer", "minimum": -1, "maximum": 2}, scalars),
            ({"type": ["integer", "null"], "exclusiveMinimum": 0, "exclusiveMaximum": 3}, scalars),
            ({"type": "string", "minLength": 1, "maxLength": 2}, strings),
            ({"type": "string", "pattern": "p"}, ["", "p", "apple", "pear", "a\np", "none", 'p"\\\n']),
            ({"type": "string", "pattern": r"\x41B"}, ["AB", "xABy", "A", "Л"]),
            ({"type": "string", "pattern": "a\0b"}, ["a\0b", "xa\0by", "a", "ab"]),
            ({"type": "string", "pattern": "^a(b|c){1,2}$"}, ["ab", "abc", "acb", "a", "abbc", "zab", "abz"]),
            ({"type": "string", "pattern": "\\n"}, ["", "\n", "a\nb", "\\n"]),
            # jsonschema uses Python re, whose dot differs from ECMAScript on CR/U+2028/U+2029.
            # The native test checks those separately against the ECMAScript line terminator set.
            ({"type": "string", "pattern": "^a.b$"}, ["acb", "a\nb", "a\tb"]),
            ({"type": "string", "pattern": r"^[\^\[]$"}, ["^", "[", "x", "]"]),
            ({"type": "string", "pattern": r"^\s$"}, [" ", "\n", "\u1680", "\u2003", "a"]),
            ({"type": "string", "pattern": r"^\S$"}, [" ", "\n", "\u1680", "\u2003", "a"]),
            ({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
             [[], [1], [1, 2], [1, 2, 3], ["x"], [None]]),
            ({"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
              "required": ["n"], "additionalProperties": False}, objects),
            ({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}, objects),
            ({"type": "object", "additionalProperties": {"type": "integer"}}, [{}, {"a": 1}, {"a": "x"}, {"a": 1, "b": 2}]),
            ({"enum": [1, "x", None], "type": "string"}, scalars + ["x"]),
            ({"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1}]}, scalars),
            ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, scalars),
            ({"allOf": [{"type": "string"}]}, scalars),
            ({"$defs": {"a/b": {"type": "integer"}}, "$ref": "#/$defs/a~1b"}, scalars),
            ({"$defs": {"x": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}, "$ref": "#/$defs/x/anyOf/1"}, scalars),
            ({"properties": {"n": {"type": "integer"}}}, scalars + [{"n": 1}, {"n": "bad"}]),
            ({"type": "array", "items": False}, [[], [0]]),
            ({"type": "object", "properties": {"bad": False}, "additionalProperties": False}, [{}, {"bad": 0}]),
            ({"type": ["integer", "null"], "minimum": 2, "maximum": 1}, [None, 1, 2]),
            ({"type": "object", "properties": {"bad": {"type": "string", "minLength": 2, "maxLength": 1}},
              "additionalProperties": False}, [{}, {"bad": "a"}, {"bad": "ab"}]),
            ({"type": "array", "items": {"type": "integer", "minimum": 2, "maximum": 1}}, [[], [1], [2]]),
            ({"type": "object", "properties": {
                "bad": {"type": "integer", "const": "x", "$defs": {"value": {"type": "integer"}}},
                "good": {"$ref": "#/properties/bad/$defs/value"}}, "additionalProperties": False},
             [{}, {"good": 1}, {"good": "x"}, {"bad": "x"}]),
            ({"type": "object", "properties": {"a": {"const": {"description": "a"}}, "b": {"const": {"description": "b"}}},
              "required": ["a", "b"], "additionalProperties": False},
             [{"a": {"description": "a"}, "b": {"description": "b"}}, {"a": {"description": "a"}, "b": {"description": "a"}}]),
        ]
        for value in [-(2**63), -(2**53)-1, 2**53+1, 2**63-1]:
            cases.append(({"type": "integer", "minimum": value, "maximum": value},
                          [value-1, value, value+1]))
        tree = {"$defs": {"node": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {
            "value": {"type": "integer"}, "next": {"$ref": "#/$defs/node"}}, "required": ["value", "next"],
            "additionalProperties": False}]}}, "$ref": "#/$defs/node"}
        cases.append((tree, [None, {"value": 1, "next": None}, {"value": 1, "next": {"value": 2, "next": None}}, {"value": "bad", "next": None}]))
        self.run_cases(cases)

    def test_positional_arrays(self):
        values = [[], ["x"], ["x", 1], ["x", 1, True], ["x", "bad"],
                  [1], [None], ["x", 1, 2], ["x", 1, True, False]]
        prefix = [{"type": "string"}, {"type": "integer"}]
        cases = []
        for minimum, maximum in [(0, 0), (0, 1), (0, 2), (1, 3), (2, 2), (3, 4)]:
            for tail in [True, False, {"type": "boolean"}]:
                if not tail and minimum > len(prefix):
                    continue
                cases.append(({"type": "array", "prefixItems": prefix, "items": tail,
                               "minItems": minimum, "maxItems": maximum}, values))
        cases.extend([
            ({"type": "array", "prefixItems": [True, False]}, values),
            ({"type": "array", "prefixItems": [False]}, values),
            ({"type": "array", "prefixItems": [True, {"type": "integer", "minimum": 2, "maximum": 1}]}, values),
            ({"type": "array", "prefixItems": prefix}, values),
            ({"type": "array", "prefixItems": prefix, "items": {"const": True}}, values),
            ({"type": "array", "allOf": [
                {"prefixItems": [{"type": "string"}], "items": {"type": "integer"}},
                {"prefixItems": [True, {"type": "boolean"}]}]}, values),
            ({"type": "array", "allOf": [
                {"prefixItems": prefix}, {"items": {"type": "string"}}]}, values),
            ({"enum": values, "type": "array", "prefixItems": prefix, "items": False}, values),
            ({"type": "array", "prefixItems": [{"type": "string"}, {"$ref": "#/prefixItems/0"}],
              "items": False}, values + [["x", "y"]]),
            ({"type": "array", "prefixItems": [{"anyOf": [{"type": "null"}, {"$ref": "#"}]}],
              "items": False}, [[], [None], [[]], [[None]], [[[None]]], [0], [None, None]]),
        ])
        draft7 = "http://json-schema.org/draft-07/schema#"
        cases.extend([
            ({"$schema": draft7, "type": "array", "items": prefix, "additionalItems": False}, values),
            ({"$schema": draft7, "type": "array", "items": {"type": "string"},
              "additionalItems": False}, values),
            ({"$schema": draft7, "type": "array", "items": [{"type": "string"},
              {"$ref": "#/items/0"}], "additionalItems": False}, values + [["x", "y"]]),
        ])
        self.run_cases(cases)

    def test_supported_conjunctions(self):
        cases = [
            (
                {
                    "type": "string",
                    "pattern": "^[A-Z]+$",
                    "minLength": 2,
                    "maxLength": 4,
                },
                ["", "A", "AB", "ABCD", "ABCDE", "Ab", "ＡＢ"],
            ),
            (
                {
                    "type": "string",
                    "pattern": "你|😀|\\n",
                    "minLength": 2,
                    "maxLength": 3,
                },
                ["你", "你好", "😀a", "x\ny", "abcd", "你abc", "\n\n"],
            ),
            (
                {"enum": ["A", "AB", "ABC", 7], "type": "string", "minLength": 2},
                ["A", "AB", "ABC", 7],
            ),
            (
                {
                    "const": {"b": 2, "a": 1},
                    "type": "object",
                    "properties": {"a": {"minimum": 1}},
                    "required": ["a"],
                },
                [{"b": 2, "a": 1}, {"b": 2, "a": 0}],
            ),
            (
                {
                    "type": "integer",
                    "minimum": 2,
                    "anyOf": [{"maximum": 4}, {"minimum": 7}],
                },
                [1, 2, 4, 5, 7],
            ),
            (
                {
                    "$defs": {"s": {"type": "string", "minLength": 2}},
                    "$ref": "#/$defs/s",
                    "maxLength": 3,
                },
                ["a", "ab", "abc", "abcd"],
            ),
            (
                {
                    "allOf": [
                        {"type": "string", "pattern": "a"},
                        {"pattern": "b", "maxLength": 3},
                    ]
                },
                ["a", "ab", "ba", "bba", "bbbb"],
            ),
            (
                {
                    "allOf": [
                        {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                        {"items": {"minimum": 2}, "maxItems": 2},
                    ]
                },
                [[], [1], [2], [2, 3], [2, 3, 4], ["2"]],
            ),
            (
                {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                            "additionalProperties": False,
                        },
                        {"properties": {"b": {"type": "integer"}}},
                    ]
                },
                [{}, {"a": "x"}, {"b": 1}, {"a": "x", "b": 1}],
            ),
            (
                {
                    "type": "object",
                    "required": ["kind"],
                    "oneOf": [
                        {
                            "properties": {"kind": {"const": "a"}},
                            "additionalProperties": False,
                        },
                        {
                            "properties": {"kind": {"const": "b"}},
                            "additionalProperties": False,
                        },
                    ],
                },
                [{}, {"kind": "a"}, {"kind": "b"}, {"kind": "c"}],
            ),
            (
                {
                    "type": ["integer", "null"],
                    "allOf": [{"minimum": 2}, {"maximum": 1}],
                },
                [None, 1, 2],
            ),
        ]
        tree = {
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                        "next": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/node"}]},
                    },
                    "required": ["value", "next"],
                    "additionalProperties": False,
                }
            },
            "$ref": "#/$defs/node",
            "properties": {"value": {"minimum": 2}},
        }
        cases.append(
            (
                tree,
                [
                    {"value": 2, "next": None},
                    {"value": 2, "next": {"value": 1, "next": None}},
                    {"value": 1, "next": None},
                ],
            )
        )
        self.run_cases(cases)

    def test_number_intervals(self):
        values = [-10, -2.5, -1, -0.5, -0.0, 0, 0.1, 0.5, 1, 1.0, 1.5, 2, 10,
                  1e-8, 1.5e-8, 2e-8, 1e100, -1e100]
        cases = []
        for lo, hi in [(-2.5, -0.5), (-0.5, 0.5), (0, 1), (1e-8, 2e-8),
                       (1e100, 2e100), (-2e100, -1e100)]:
            for lower in ["minimum", "exclusiveMinimum"]:
                for upper in ["maximum", "exclusiveMaximum"]:
                    cases.append(({"type": "number", lower: lo, upper: hi}, values))
        cases.extend([
            ({"type": "number", "minimum": 0.5, "maximum": 0.5}, values),
            ({"type": "number", "minimum": 1}, values),
            ({"type": "number", "exclusiveMaximum": 0}, values),
            ({"type": ["number", "null"], "minimum": 0, "maximum": 1}, values + [None]),
            ({"allOf": [{"type": "number", "minimum": -1}, {"exclusiveMinimum": 0, "maximum": 1}]}, values),
            # Finite constants and integer branches retain their canonical integer spelling.
            ({"type": "number", "enum": [0, 0.5, 1, 2], "exclusiveMinimum": 0, "maximum": 1},
             [0, 0.1, 0.5, 1, 1.5, 2]),
            ({"allOf": [{"type": "integer"}, {"minimum": 0.5, "maximum": 2.5}]},
             [-1, 0, 0.5, 1, 1.5, 2, 2.5, 3]),
        ])
        for endpoint in [2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63)]:
            cases.append(({"type": "number", "minimum": endpoint, "maximum": endpoint},
                          [endpoint - 1, endpoint, endpoint + 1]))
        self.run_cases(cases)

    def test_number_publication_boundaries(self):
        cases = [
            ({"type": "number", "exclusiveMinimum": 0.1, "maximum": 0.2},
             ["0.1", "0.10000000000000001", "0.10000000000000002", "0.2", "0.20000000000000001"]),
            ({"type": "number", "minimum": -0.2, "exclusiveMaximum": -0.1},
             ["-0.1", "-0.10000000000000001", "-0.10000000000000002", "-0.2"]),
            ({"type": "number", "exclusiveMinimum": 0, "maximum": 1e-323},
             ["1e-324", "2e-324", "3e-324", "5e-324", "1e-323", "1.1e-323"]),
            ({"type": "number", "minimum": 1e308},
             ["1e308", "1.5e308", "1.7976931348623157e308", "1.8e308", "1e309"]),
        ]
        reports = subprocess.run([str(PROBE), "--probe"],
            input="".join(compact({"schema": s, "candidates": v}) + "\n" for s, v in cases),
            text=True, capture_output=True, check=True, timeout=45)
        for (schema, values), line in zip(cases, reports.stdout.splitlines()):
            report = json.loads(line)
            self.assertNotIn("error", report, report)
            validator = jsonschema.Draft202012Validator(
                json.loads(compact(schema), parse_float=Decimal))
            expected = []
            for text in values:
                decoded = json.loads(text)
                raw = json.loads(text, parse_float=Decimal)
                valid = math.isfinite(decoded) and validator.is_valid(raw)
                if valid:
                    published = json.loads(json.dumps(decoded), parse_float=Decimal)
                    valid = validator.is_valid(published)
                expected.append(valid)
            self.assertEqual(report["accepted"], expected, (schema, values, report, expected))

    def test_source_number_precision(self):
        sources = [
            ('{"type":"number","minimum":0.10000000000000001}', "/minimum"),
            ('{"type":"number","minimum":1e-400}', "/minimum"),
            ('{"type":"object","properties":{"a/b":{"const":0.10000000000000001}}}',
             "/properties/a~1b/const"),
            ('{"type":"number","minimum":0.10000000000000001,"minimum":0.1}', None),
            ('{"type":"number","default":0.10000000000000001,"minimum":0.1}', None),
        ]
        result = subprocess.run([str(PROBE), "--probe"],
            input="".join(compact({"schema_source": s, "candidates": []}) + "\n" for s, _ in sources),
            text=True, capture_output=True, check=True, timeout=30)
        for (source, pointer), line in zip(sources, result.stdout.splitlines()):
            with self.subTest(source=source):
                report = json.loads(line)
                if pointer:
                    self.assertEqual(report.get("pointer"), pointer, report)
                else:
                    self.assertNotIn("error", report, report)

    def test_numeric_neighbors(self):
        rng = random.Random(804)
        cases = []
        for _ in range(32):
            endpoint = float(f"{rng.uniform(-9, 9):.8g}e{rng.randrange(-300, 301)}")
            left = math.nextafter(endpoint, -math.inf)
            right = math.nextafter(endpoint, math.inf)
            # Use canonical JSON floating spellings, checking both sides of each open endpoint.
            candidates = [left, endpoint, right]
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                cases.append(({"type": "number", key: endpoint}, candidates))
        self.run_cases(cases)

    def test_additional_keys_cannot_override_declared_values(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        candidates = ['{"n":1,"n":"bad"}', '{"n":1,"\\u006e":"bad"}', '{"n":1,"x":"good"}']
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input=compact({"schema": schema, "candidates": candidates}) + "\n",
            text=True, capture_output=True, check=True, timeout=30,
        )
        self.assertEqual(json.loads(result.stdout)["accepted"], [False, False, True])

    def test_rejections_and_diagnostics(self):
        cases = [
            ({"$schema": 7}, "/$schema"),
            ({"type": "array", "prefixItems": []}, "/prefixItems"),
            ({"type": "array", "prefixItems": [7]}, "/prefixItems/0"),
            ({"type": "array", "items": [{"type": "string"}]}, "/items"),
            ({"$schema": "http://json-schema.org/draft-07/schema#", "prefixItems": [{}]}, "/prefixItems"),
            ({"type": "object", "properties": {"gap": {
                "type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": math.nextafter(1, math.inf)}}},
             "/properties/gap"),
            ({"allOf": [{"type": "number"}, {"multipleOf": 1}]}, "/allOf/1/multipleOf"),
            (
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$defs": {"x": {"type": "string"}},
                    "$ref": "#/$defs/x",
                    "maxLength": 2,
                },
                "/maxLength",
            ),
            ({"type": "string", "format": "email"}, "/format"),
            ({"type": "array", "uniqueItems": True}, "/uniqueItems"),
            (
                {
                    "type": "object",
                    "properties": {"x": {"type": "number", "multipleOf": 1}},
                },
                "/properties/x/multipleOf",
            ),
            ({"$ref": "https://example.com/schema"}, "/$ref"),
            ({"type": "string", "pattern": r"[\q]"}, "/pattern"),
            ({"type": "string", "pattern": r"\xZZ"}, "/pattern"),
            ({"type": "integer", "exclusiveMinimum": 2**63 - 1}, "/exclusiveMinimum"),
            ({"const": {"large": 2**63 + 1}}, "/const/large"),
            ({"const": -(2**63) - 1}, "/const"),
            ({"type": "string", "minLength": -1}, "/minLength"),
        ]
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input="".join(compact({"schema": s, "candidates": []}) + "\n" for s, _ in cases),
            text=True, capture_output=True, check=True, timeout=30,
        )
        self.assertEqual(len(result.stdout.splitlines()), len(cases))
        for report, (schema, pointer) in zip(result.stdout.splitlines(), cases):
            with self.subTest(schema=schema):
                data = json.loads(report)
                self.assertIn("error", data)
                self.assertEqual(data["pointer"], pointer)

    def test_empty_languages_fail_before_generation(self):
        schemas = [
            {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"a": {}},
                        "additionalProperties": False,
                    },
                    {"required": ["b"]},
                ]
            },
            False,
            {"const": "x", "type": "integer"},
            {"$ref": "#"},
            {"type": "object", "properties": {"x": False}, "required": ["x"]},
            {"anyOf": [False, {"type": "integer", "minimum": 2, "maximum": 1}]},
        ]
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input="".join(compact({"schema": s, "candidates": []}) + "\n" for s in schemas),
            text=True, capture_output=True, check=True, timeout=30,
        )
        self.assertEqual(len(result.stdout.splitlines()), len(schemas))
        for schema, line in zip(schemas, result.stdout.splitlines()):
            with self.subTest(schema=schema):
                self.assertIn("error", json.loads(line))


if __name__ == "__main__":
    unittest.main()
