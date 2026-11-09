#!/usr/bin/env python3
"""External validation of ninfer_engine_constraint_real_test reports (structured output P5/P6).

Each JSONL line carries the schema and the generated content; every content must parse as JSON,
validate against its schema with the independent `jsonschema` package, and end on EOS.
Usage: validate_constraint_report.py REPORT.jsonl [...]
"""

import json
import sys

import jsonschema


def main(paths: list[str]) -> int:
    total = valid = eos = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                record = json.loads(line)
                total += 1
                eos += record["finish"] == "stop_token"
                try:
                    jsonschema.validate(json.loads(record["content"]), record["schema"])
                    valid += 1
                except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                    print(f"{path}:{number} {record['case']}: {str(error).splitlines()[0]}")
                    print(f"  content: {record['content']!r}")
    print(f"{valid}/{total} valid against their schema, {eos}/{total} ended on EOS")
    return 0 if total and valid == total and eos == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
