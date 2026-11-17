"""The smoke client's pass/fail checks must reject malformed server responses."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/smoke"))
from decision import FIXTURES, in_domain, validate_response


def response_for(request):
    schema = request["schema"].get("properties", request["schema"])
    decision, fields = {}, {}
    for name, spec in schema.items():
        value = (spec.get("enum", spec.get("choices")) or
                 ([False] if spec.get("type") == "boolean" else [spec["minimum"]]))[0]
        decision[name] = value
        fields[name] = {"value": value, "probability": 0.5, "tree": True, "scored_nodes": 1}
        if spec.get("type") in ("integer", "number"):
            fields[name]["interval_p10_p90"] = [spec["minimum"], spec["maximum"]]
    return {
        "object": "decision",
        "results": [copy.deepcopy({"decision": decision, "fields": fields}) for _ in request["contexts"]],
        "usage": {"prompt_tokens": 4, "computed_tokens": 3, "cached_tokens": 1,
                  "context_tokens": 1, "scored_rows": len(fields)},
        "timings": {"prefill_ms": 1, "scoring_ms": 1, "total_ms": 2, "per_decision_ms": 2},
    }


class DecisionSmokeTest(unittest.TestCase):
    def setUp(self):
        self.presets = json.loads(FIXTURES.read_text(encoding="utf-8"))["presets"]
        self.request = next(p["request"] for p in self.presets if p["id"] == "routing")
        self.response = response_for(self.request)

    def test_all_fixture_shapes(self):
        for preset in self.presets:
            with self.subTest(preset=preset["id"]):
                validate_response(preset["request"], response_for(preset["request"]))

    def test_boolean_is_not_integer(self):
        spec = {"type": "integer", "minimum": 0, "maximum": 1}
        self.assertFalse(in_domain(True, spec))
        self.assertFalse(in_domain(1.0, spec))
        self.response["results"][0]["decision"]["urgent"] = 0
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_probability_is_finite(self):
        for invalid in (float("nan"), float("inf"), -0.1, 1.1, True):
            self.response["results"][0]["fields"]["urgent"]["probability"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_response(self.request, self.response)

    def test_missing_context(self):
        self.response["results"] = []
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_diagnostic_mismatch(self):
        self.response["results"][0]["fields"]["urgent"]["value"] = True
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_cache_opt_out_and_accounting(self):
        validate_response({**self.request, "cache_prompt": False}, self.response)
        self.response["usage"]["prompt_tokens"] = 5
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_grid_and_interval(self):
        self.assertFalse(in_domain(0.3, {"type": "number", "minimum": 0, "maximum": 1, "step": 0.5}))
        request = next(p["request"] for p in self.presets if p["id"] == "arm")
        response = response_for(request)
        response["results"][0]["fields"]["gripper_cm"]["interval_p10_p90"] = [12, 0]
        with self.assertRaises(ValueError):
            validate_response(request, response)


if __name__ == "__main__":
    unittest.main()
