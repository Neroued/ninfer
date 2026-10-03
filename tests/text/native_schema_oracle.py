"""CPU-only schema oracle/corpus; never imports a model or contacts an inference server.

Generate: python tests/text/native_schema_oracle.py --emit-corpus corpus.jsonl --report oracle.json
Compare:  python tests/text/native_schema_oracle.py --native-results native.jsonl --report comparison.json
An external native_schema_probe reads corpus.jsonl on stdin and emits native.jsonl.
Temporal oracle covers the documented ordinary-second, positive-year RFC3339 subset.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from datetime import date, time, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator, FormatChecker, ValidationError, validators


CHECKER = FormatChecker()


class EcmaPatternOracle:
    """Independent ECMA262 implementation; Unicode mode matches JSON Schema guidance."""
    source = r"""
const readline = require('node:readline');
readline.createInterface({input: process.stdin, crlfDelay: Infinity}).on('line', line => {
  try {
    const item = JSON.parse(line);
    console.log(JSON.stringify({match: new RegExp(item.pattern, 'u').test(item.value)}));
  } catch (error) {
    console.log(JSON.stringify({error: String(error)}));
  }
});
"""

    def __init__(self):
        self.node = os.environ.get("NINFER_ORACLE_NODE") or shutil.which("node")
        self.process = None
        self.cache = {}
        atexit.register(self.close)

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process.stdout.close()
            self.process = None

    def matches(self, pattern, value):
        if not isinstance(pattern, str):
            raise ValueError("pattern must be a string")
        key = (pattern, value)
        if key not in self.cache:
            if not self.node:
                raise RuntimeError("Node.js is required for the independent ECMA262 regex oracle")
            if self.process is None:
                self.process = subprocess.Popen([self.node, "-e", self.source], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, text=True, encoding="utf-8",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.process.stdin.write(json.dumps({"pattern": pattern, "value": value}, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
            answer = self.process.stdout.readline()
            if not answer:
                raise RuntimeError("ECMA262 regex oracle exited without a response")
            self.cache[key] = json.loads(answer)
        result = self.cache[key]
        if "error" in result:
            raise ValueError(result["error"])
        return result["match"]


ECMA = EcmaPatternOracle()


def ecma_pattern(validator, pattern, instance, schema):
    if isinstance(instance, str) and not ECMA.matches(pattern, instance):
        yield ValidationError("String does not satisfy the independent ECMA262 pattern")


SchemaValidator = validators.extend(Draft202012Validator, {"pattern": ecma_pattern})


def scalar_strings(value):
    """Unpaired surrogates are excluded from the documented interoperable JSON subset."""
    if isinstance(value, str):
        return not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    if isinstance(value, dict):
        return all(scalar_strings(key) and scalar_strings(item) for key, item in value.items())
    if isinstance(value, list):
        return all(scalar_strings(item) for item in value)
    return True


@CHECKER.checks("date", raises=(ValueError, TypeError))
def calendar_date(value):
    if not isinstance(value, str):
        return True
    parts = value.split("-")
    if tuple(map(len, parts)) != (4, 2, 2) or not all(p.isascii() and p.isdecimal() for p in parts):
        return False
    return date(*map(int, parts)).isoformat() == value


@CHECKER.checks("time", raises=(ValueError, TypeError))
def clock_time(value):
    if not isinstance(value, str):
        return True
    # This validates shape only. datetime validates hour/minute/second and offset bounds.
    parts = re.fullmatch(r"([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]+))?([Zz]|[+-][0-9]{2}:[0-9]{2})", value)
    if parts is None:
        return False
    hour, minute, second, fraction, offset = parts.groups()
    zone = timezone.utc
    if offset not in ("Z", "z"):
        offset_hour, offset_minute = map(int, offset[1:].split(":"))
        if offset_minute >= 60:
            return False
        zone = timezone((1 if offset[0] == "+" else -1) *
                        timedelta(hours=offset_hour, minutes=offset_minute))
    # Arbitrary RFC3339 fractional precision is legal. Truncate only to construct
    # datetime's fixed microsecond representation; shape already checks every digit.
    microsecond = int(((fraction or "") + "000000")[:6])
    parsed = time(int(hour), int(minute), int(second), microsecond, tzinfo=zone)
    return parsed.tzinfo is not None


@CHECKER.checks("date-time", raises=(ValueError, TypeError))
def timestamp(value):
    if not isinstance(value, str):
        return True
    parts = re.split("[Tt]", value)
    return len(parts) == 2 and calendar_date(parts[0]) and clock_time(parts[1])


def oracle_accepts(schema, wire):
    try:
        if isinstance(wire, bytes):
            wire = wire.decode("utf-8", errors="strict")
        def invalid_constant(value):
            raise ValueError(f"Invalid JSON numeric constant: {value}")

        parsed = json.loads(wire, parse_constant=invalid_constant)
        return scalar_strings(parsed) and SchemaValidator(schema, format_checker=CHECKER).is_valid(parsed)
    except (ValueError, TypeError):
        return False


def schema_positions(schema, pointer=""):
    """Walk schema locations, never interpreting names/data as schema keywords."""
    if not isinstance(schema, (dict, bool)):
        return
    yield pointer, schema
    if not isinstance(schema, dict):
        return
    def escaped(part):
        return str(part).replace("~", "~0").replace("/", "~1")

    maps = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies")
    singles = ("additionalProperties", "additionalItems", "items", "contains", "not",
               "if", "then", "else", "propertyNames", "unevaluatedProperties", "unevaluatedItems")
    arrays = ("anyOf", "oneOf", "allOf", "prefixItems")
    for keyword in maps:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for name, child in children.items():
                yield from schema_positions(child, f"{pointer}/{keyword}/{escaped(name)}")
    for keyword in singles:
        child = schema.get(keyword)
        if isinstance(child, list) and keyword == "items":
            for index, entry in enumerate(child):
                yield from schema_positions(entry, f"{pointer}/{keyword}/{index}")
        else:
            yield from schema_positions(child, f"{pointer}/{keyword}")
    for keyword in arrays:
        children = schema.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                yield from schema_positions(child, f"{pointer}/{keyword}/{index}")


def cases(business_schemas=()):
    sequence = 0

    def make(category, schema, value=None, wire=None, compile_error=False,
             subset_reason=None, compile_only=False, name=None, wire_bytes=None):
        nonlocal sequence
        sequence += 1
        encoded = json.dumps(value, ensure_ascii=False) if wire is None else wire
        valid = oracle_accepts(schema, bytes(wire_bytes) if wire_bytes is not None else encoded) if not compile_only else None
        return {"id": sequence, "category": category, "schema": schema, "wire": encoded,
                "expected_compile_error": compile_error,
                "oracle_valid": valid,
                "expected_accept": False if compile_error or compile_only else valid,
                "compile_only": compile_only, "name": name,
                "allowed_subset_rejection": subset_reason,
                **({"wire_bytes": wire_bytes} if wire_bytes is not None else {})}

    # Every month/day across common, divisible-by-four, century and 400-year boundaries.
    for year in (1, 4, 100, 400, 1900, 2000, 2024, 2025, 2100, 2400, 9999):
        for month in range(1, 13):
            for day in range(0, 33):
                yield make("calendar_grid", {"type": "string", "format": "date"},
                           f"{year:04d}-{month:02d}-{day:02d}")
    for value in ("0000-01-01", "10000-01-01", "2026-00-01", "2026-13-01", "2026-1-01",
                  "2026-01-1", "2026-01-01\n", " 2026-01-01", "２０２６-01-01", "2026/01/01"):
        yield make("date_shape", {"type": "string", "format": "date"}, value)

    clocks = ("00:00:00Z", "23:59:59z", "12:34:56.123456789Z", "12:34:56+08:00",
              "12:34:56-05:30", "12:34:56-00:00", "12:34:56+23:59", "24:00:00Z",
              "23:60:00Z", "23:59:60Z", "12:00:00+24:00", "12:00:00+00:60", "12:00:00",
              "12:00:00.Z", "12:00Z", "12:00:00+0800", "12:00:00Z\n", "1:00:00Z")
    for value in clocks:
        yield make("time_bounds", {"type": "string", "format": "time"}, value)
    for day in ("2000-02-29", "1900-02-29", "2024-02-29", "2025-02-29", "2026-04-31"):
        for separator in ("T", "t", " "):
            for clock in clocks:
                yield make("timestamp", {"type": "string", "format": "date-time"}, day + separator + clock)

    patterns = {
        "search": ("code", ("code", "prefix-code-suffix", "\ncode\n", "co\"de", "none")),
        "start_anchor": ("^code", ("code-suffix", "prefix-code", "code\n", "none")),
        "end_anchor": ("code$", ("prefix-code", "code-suffix", "code\n", "none")),
        "full_anchor": ("^code-[AB][0-9]+$", ("code-A12", "code-B0", "xcode-A12", "code-C12", "code-A12x")),
        "quote_escape": ('^a"b$', ('a"b', 'a\\"b', 'a"bx')),
        "backslash_escape": (r"^a\\b$", ("a\\b", "ab", "a/b")),
        "newline_escape": (r"^a\nb$", ("a\nb", "a\\nb", "ab")),
        "class": ("^[A-Z][a-z0-9_-]+$", ("Aok_9", "A", "aok", "Aok!")),
        "alternation": ("(?:cat|dog)", ("a dog!", "cat", "hotdog", "bird")),
        "dot_unicode": ("^.$", ("a", "é", "😀", '"', "\\", "\n", "\r", "\u2028", "\u2029", "ab")),
        "nonspace_ecma": (r"^\S$", ("a", "é", "😀", "\u00a0", "\ufeff", "\u2028", "\u2029", "\r", "\n")),
        "space_ecma": (r"^\s$", (" ", "\t", "\u00a0", "\ufeff", "\u2028", "\u2029", "a")),
        "unicode_literal": ("汉", ("汉", "前汉后", "字", "\\u6c49")),
        "unicode_pair": (r"^(?:\uD83D\uDE00|x)$", ("😀", "x", "😀x", "y")),
        "empty": ("", ("", "anything", "\n")),
    }
    for category, (pattern, values) in patterns.items():
        for value in values:
            schema = {"type": "string", "pattern": pattern}
            # No multiline flag is part of the Schema contract: ECMA262 $ requires
            # literal input end, unlike Python's before-final-newline behavior.
            yield make("pattern_" + category, schema, value)
            # JSON spelling must not change the logical string used by the regex.
            yield make("pattern_ascii_escapes_" + category, schema,
                       wire=json.dumps(value, ensure_ascii=True))
    for wire in ('"\\u0063ode"', '"co\\u0064e"', '"\\u0063\\u006f\\u0064\\u0065"'):
        yield make("unicode_search_escapes", {"type": "string", "pattern": "code"}, wire=wire)
    for wire in ('"\\q"', '"unterminated', '"raw\nnewline"', '"raw\x01control"', '"\\u12"',
                 'NaN', 'Infinity', '-Infinity'):
        yield make("malformed_json", {"type": "string", "pattern": ".*"}, wire=wire)
    for wire in ('"\\ud800"', '"\\udfff"', '"\\ud800x"', '"\\udfff\\ud800"'):
        yield make("unpaired_surrogate_escape", {"type": "string", "pattern": ".*"}, wire=wire)
    for wire in ('"\\ud83d\\ude00"', '"\\u0000"', '"\\u2028"', '"\\u2029"'):
        yield make("valid_unicode_escape", {"type": "string", "pattern": r"^[\s\S]*$"}, wire=wire)
    invalid_utf8 = ([0xed, 0xa0, 0x80], [0xed, 0xbf, 0xbf], [0xc0, 0xaf], [0xe0, 0x80, 0xaf],
                    [0xf4, 0x90, 0x80, 0x80], [0x80], [0xf0, 0x9f, 0x98])
    for content in invalid_utf8:
        yield make("invalid_raw_utf8", {"type": "string", "pattern": ".*"}, wire_bytes=[0x22, *content, 0x22])
    for content in ([0xf0, 0x9f, 0x98, 0x80], [0xf4, 0x8f, 0xbf, 0xbf]):
        yield make("valid_raw_utf8", {"type": "string", "pattern": ".*"}, wire_bytes=[0x22, *content, 0x22])
    # Object framing avoids a bare-root string lookahead at end-of-input and checks
    # the ordinary (non-pattern) grammar independently of regex normalization.
    for label, string_schema in (("basic", {"type": "string"}),
                                 ("bounded", {"type": "string", "minLength": 1, "maxLength": 1})):
        schema = {"type": "object", "properties": {"value": string_schema},
                  "required": ["value"], "additionalProperties": False}
        for value in ("a", "é", "汉", "😀", "\ud7ff", "\ue000", "\u2028", "\u2029", "\U0010ffff"):
            yield make(label + "_string_object_scalar", schema, {"value": value})
            yield make(label + "_string_object_escaped_scalar", schema,
                       wire=json.dumps({"value": value}, ensure_ascii=True))
        for value in ("", "ab", "😀é"):
            yield make(label + "_string_object_length", schema, {"value": value})
        for content in invalid_utf8:
            yield make(label + "_string_object_invalid_utf8", schema,
                       wire_bytes=list(b'{"value":"') + content + list(b'"}'))
        for wire in ('{"value":"\\ud800"}', '{"value":"\\udfff"}',
                     '{"value":"\\ud800x"}', '{"value":"\\udfff\\ud800"}'):
            yield make(label + "_string_object_unpaired_surrogate", schema, wire=wire)
    for value in (None, True, 1, [], {}):
        yield make("wrong_type", {"type": "string", "format": "date"}, value)

    unsupported = ({"type": "string", "pattern": r"^(a)\1$"},
                   {"type": "string", "pattern": "a(?=b)"},
                   {"type": "string", "pattern": r"\bcode\b"},
                   {"type": "string", "pattern": "a", "minLength": 1},
                   {"type": "string", "format": "date", "maxLength": 10},
                   {"type": "string", "format": "date", "pattern": ".*"},
                   {"type": "string", "format": "email"},
                   {"type": "string", "pattern": "^cat|dog$"},
                   {"type": "string", "pattern": "^cat|dog"},
                   {"type": "string", "pattern": "cat|dog$"},
                   {"type": "string", "pattern": "a(?!b)"},
                   {"type": "string", "pattern": "(?<=a)b"},
                   {"type": "string", "pattern": "(?<!a)b"},
                   {"type": "string", "pattern": "a", "maxLength": 10},
                   {"type": "string", "format": "date", "minLength": 10},
                   {"type": "string", "pattern": r"\a"},
                   {"type": "string", "pattern": r"\e"},
                   {"type": "string", "pattern": r"\Acode"},
                   {"type": "string", "pattern": r"code\Z"},
                   {"type": "string", "pattern": r"code\z"},
                   {"type": "string", "pattern": r"code\B"},
                   {"type": "string", "pattern": r"^[\s-\uFFFF]$"},
                   {"type": "string", "pattern": r"^[a-\s]$"},
                   {"type": "string", "pattern": r"^[\S-a]$"},
                   {"type": "string", "pattern": "^a{ 1 }$"},
                   {"type": "string", "pattern": "^a**$"},
                   {"type": "string", "pattern": "^a{1}{2}$"},
                   {"type": "string", "pattern": "^a*?$"},
                   {"type": "string", "pattern": r"^(?:\uD800|x)$"},
                   {"type": "string", "pattern": r"^\01$"},
                   {"pattern": "a"}, {"type": "string", "pattern": 1})
    for schema in unsupported:
        yield make("fail_closed", schema, "a", compile_error=True)
    # Required additional properties must be materialized, or closed schemas refused.
    for schema in ({"type":"object","required":["missing"],"additionalProperties":False},
                   {"type":"object","properties":{"topic":{"type":"string"}},"required":["topic","description"],"additionalProperties":False},
                   {"type":"object","required":["x","x"]},
                   {"type":"object","required":[1]},
                   {"type":"object","required":"x"}):
        yield make("required_contract_compile",schema,{},compile_error=True)
    for schema in ({"type":"object","required":["x"],"additionalProperties":True},
                   {"type":"object","required":["x"],"additionalProperties":{"type":"integer","minimum":3}}):
        for value in ({"x":4},{},{"x":1},{"x":"wrong"}):
            yield make("required_additional_contract",schema,value)
    literal_schema={"type":"array","prefixItems":[{"const":{"description":"left"}},{"const":{"description":"right"}}],"items":False,"minItems":2,"maxItems":2}
    for value in ([{"description":"left"},{"description":"right"}],
                  [{"description":"left"},{"description":"left"}]):
        yield make("metadata_named_literal_cache",literal_schema,
                   wire=json.dumps(value,ensure_ascii=False,separators=(',',':')))
    property_schema={"type":"array","prefixItems":[
        {"type":"object","properties":{"description":{"type":"string"}},"required":["description"],"additionalProperties":False},
        {"type":"object","properties":{"description":{"type":"integer"}},"required":["description"],"additionalProperties":False}],
        "items":False,"minItems":2,"maxItems":2}
    for value in ([{"description":"text"},{"description":1}],
                  [{"description":"text"},{"description":"wrong"}]):
        yield make("metadata_named_property_cache",property_schema,value)
    for schema,value in (({"const":"literal","enum":["literal"],"type":"string"},"literal"),
                         ({"const":"literal","enum":["literal"],"type":"string"},"other"),
                         ({"const":1,"enum":[1.0],"type":"integer"},1),
                         ({"const":{"description":"left","title":3},"enum":[{"title":3,"description":"left"}]},{"description":"left","title":3})):
        yield make("compatible_const_enum",schema,wire=json.dumps(value,separators=(',',':')))
    for schema in ({"const":"literal","enum":["other"]},
                   {"const":True,"enum":[1]},
                   {"const":9007199254740993,"enum":[9007199254740992.0]},
                   {"const":1,"enum":[]}):
        yield make("incompatible_const_enum",schema,None,compile_error=True)
    # Keyword-looking business property names must remain ordinary property names.
    for value in ({"pattern": "code", "format": "2024-02-29"},
                  {"pattern": "none", "format": "2025-02-29"}):
        yield make("keyword_property_names", {
            "type": "object", "properties": {
                "pattern": {"type": "string", "pattern": "code"},
                "format": {"type": "string", "format": "date"}},
            "required": ["pattern", "format"], "additionalProperties": False}, value)
    for entry in business_schemas:
        yield make("business_schema_compile", entry["schema"], compile_only=True, name=entry["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-corpus", type=Path)
    parser.add_argument("--native-results", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--schemas", type=Path,
                        help="JSON list of {name, schema} business schemas; compile-only, no inference")
    args = parser.parse_args()
    business_schemas = json.loads(args.schemas.read_text(encoding="utf-8-sig")) if args.schemas else []
    corpus = list(cases(business_schemas))
    if args.emit_corpus:
        args.emit_corpus.write_text("".join(json.dumps(c, ensure_ascii=True) + "\n" for c in corpus), encoding="utf-8")
    report = {"execution": "CPU-only independent oracle", "model_loaded": False,
              "cases": len(corpus), "categories": dict(Counter(c["category"] for c in corpus)),
              "expected_accept": sum(c["expected_accept"] for c in corpus),
              "expected_compile_error": sum(c["expected_compile_error"] for c in corpus),
              "native_tested": args.native_results is not None,
              "regex_oracle": "Independent Node.js ECMA262 RegExp with Unicode mode; JSON Schema search semantics",
              "unicode_scope": "Strict UTF-8 and Unicode scalar values; unpaired surrogate spellings excluded",
              "business_schema_source": str(args.schemas) if args.schemas else None,
              "business_schemas": [{"name": item["name"], "compile_only": True,
                  "constraints": [{"pointer": pointer, "keyword": keyword, "value": schema[keyword]}
                      for pointer, schema in schema_positions(item["schema"])
                      if isinstance(schema, dict) for keyword in ("pattern", "format") if keyword in schema]}
                  for item in business_schemas],
              "temporal_scope": "Positive four-digit Gregorian dates; ordinary RFC3339 seconds; leap-second spellings excluded."}
    failures = []
    subset_rejections = []
    if args.native_results:
        outputs = [json.loads(line) for line in args.native_results.read_text(encoding="utf-8").splitlines() if line]
        by_id = {r["id"]: r for r in outputs}
        if len(by_id) != len(outputs):
            failures.append({"error": "duplicate native result id"})
        unexpected = sorted(set(by_id) - {c["id"] for c in corpus})
        if unexpected:
            failures.append({"error": "unexpected native ids", "ids": unexpected})
        for case in corpus:
            native = by_id.get(case["id"])
            subset_rejection = (native is not None and case["expected_accept"] and
                                not native.get("accepted") and case["allowed_subset_rejection"] and
                                not native.get("compile_error") and not native.get("internal_error"))
            if subset_rejection:
                subset_rejections.append({"id": case["id"], "reason": case["allowed_subset_rejection"]})
            correct = (native is not None and
                       bool(native.get("compile_error")) == case["expected_compile_error"] and
                       (case["expected_compile_error"] or case["compile_only"] or subset_rejection or
                        bool(native.get("accepted")) == case["expected_accept"]) and
                       not native.get("internal_error"))
            if not correct:
                kind = ("missing_result" if native is None else
                        "internal_error" if native.get("internal_error") else
                        "accepted_invalid" if native.get("accepted") and case["oracle_valid"] is False else
                        "compile_contract_mismatch" if bool(native.get("compile_error")) != case["expected_compile_error"] else
                        "rejected_valid" if case["oracle_valid"] and not native.get("accepted") else
                        "acceptance_contract_mismatch")
                failures.append({"kind": kind, "case": case, "native": native})
        report.update(native_results=len(outputs), failures=failures,
                      allowed_subset_rejections=subset_rejections,
                      accepted_invalid=sum(f.get("kind") == "accepted_invalid" for f in failures),
                      rejected_valid=sum(f.get("kind") == "rejected_valid" for f in failures),
                      passed=len(corpus) - sum("case" in f for f in failures))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("categories", "failures")}, ensure_ascii=False))
    if failures:
        print(f"Native comparison failures: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
