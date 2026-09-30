import copy
import json
from pathlib import Path
import tempfile
import unittest

from kvmem_quality import compare, fixtures, grade


def response(text, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": text}}]}


class QualityContracts(unittest.TestCase):
    def test_whole_answer_must_match(self):
        self.assertTrue(grade(response('{"answer":"KEY-123"}'), "KEY-123")["correct"])
        for text in ('There is no code.\nKEY-123',
                     'There is no code.\n{"answer":"KEY-123"}',
                     '{"answer":"WRONG","reason":"KEY-123"}',
                     '{"answer":"KEY-123","denial":true}',
                     '{"answer":"WRONG","answer":"KEY-123"}',
                     '```json\n{"answer":"KEY-123"}\n```'):
            self.assertFalse(grade(response(text), "KEY-123")["correct"])

    def test_length_stop_cannot_pass(self):
        self.assertFalse(grade(response('{"answer":"KEY-123"}', "length"), "KEY-123")["correct"])

    def test_fixtures_are_repeatable_and_have_tool_tail(self):
        a, b = fixtures(40), fixtures(40)
        self.assertEqual(a, b)
        self.assertEqual(len({c["id"] for c in a["cases"]}), 15)
        for case in a["cases"]:
            if case["family"] == "tool_tail":
                self.assertEqual(case["messages"][-1]["role"], "tool")
                self.assertGreater(len(case["messages"][-1]["content"]), 2000)

    def test_comparison_rejects_incomplete_and_mismatched_runs(self):
        report = {"fixture_sha256": "abc", "complete": True, "total": 1, "correct": 1,
                  "cases": [{"id": "a", "grade": {"correct": True}}]}
        with tempfile.TemporaryDirectory() as root:
            left, right = Path(root) / "a.json", Path(root) / "b.json"
            left.write_text(json.dumps(report))
            right.write_text(json.dumps(report))
            self.assertFalse(compare([left, right])["equivalence_established"])
            for field, value in (("complete", False), ("fixture_sha256", "other"), ("total", 2)):
                bad = copy.deepcopy(report)
                bad[field] = value
                right.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    compare([left, right])

    def test_semantic_comparison_requires_same_judge_and_complete_labels(self):
        report = {"kind": "longmemeval-s-cleaned-full-history", "fixture_sha256": "abc",
                  "source": {"sha256": "source"},
                  "complete": True, "scored": True, "total": 1, "correct": 1,
                  "review": {"reviewer": "Qwen fixed", "rubric": "rubric-v1",
                             "judge_settings": {"model": "qwen-fixed", "temperature": 0}},
                  "cases": [{"id": "a", "family": "multi-session", "grade": {"correct": True}}]}
        with tempfile.TemporaryDirectory() as root:
            left, right = Path(root) / 'a.json', Path(root) / 'b.json'
            left.write_text(json.dumps(report))
            mapping = {'kind': 'longmemeval-question-clusters', 'source_sha256': 'source',
                       'rule': 'test map', 'question_count': 1,
                       'clusters': [{'id': 'a', 'question_ids': ['a']}]}
            right.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                compare([left, right])
            result = compare([left, right], mapping)
            self.assertEqual(result['comparisons'][0]['paired_statistics']['clusters'], 1)
            with self.assertRaises(ValueError):
                compare([left, right], {**mapping, 'source_sha256': 'stale'})
            for field, value in (("reviewer", "different judge"), ("rubric", "different rubric"),
                                 ("judge_settings", {"model": "other"}),
                                 ("adjudication", {"protocol": "different source-review rule"})):
                bad = copy.deepcopy(report)
                bad['review'][field] = value
                right.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    compare([left, right], mapping)
            for field, value in (("scored", False), ("request_errors", 1), ("correct", 0)):
                bad = copy.deepcopy(report)
                bad[field] = value
                right.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    compare([left, right], mapping)


if __name__ == "__main__":
    unittest.main()
