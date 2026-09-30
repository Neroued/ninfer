import copy
import json
import unittest

from kvmem_semantic_review import attach, review_template


class SemanticReviewContract(unittest.TestCase):
    def setUp(self):
        self.raw = json.dumps({"complete": True, "total": 1, "fixture_sha256": "fixture",
            "cases": [{"id": "a", "family": "multi-session", "question": "How many?", "gold": 3,
                       "response": {"choices": [{"message": {"content": "2"}}]}}]}).encode()
        self.review = review_template(self.raw)
        self.review.update(reviewer="manual test", rubric="whole-answer-v1")
        self.review["cases"][0].update(rubric_correct=False, strict_correct=False,
                                       rationale="Two omits one of the three required services.")

    def test_retains_failure_and_raw_response(self):
        result = attach(self.raw, self.review)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["cases"][0]["response"], json.loads(self.raw)["cases"][0]["response"])
        self.assertFalse(result["equivalence_established"])

    def test_rejects_stale_partial_and_unresolved_reviews(self):
        for field, value in (("prediction_sha256", "stale"), ("reviewer", ""), ("cases", [])):
            bad = copy.deepcopy(self.review)
            bad[field] = value
            with self.assertRaises(ValueError):
                attach(self.raw, bad)
        for field, value in (("strict_correct", None), ("strict_correct", True),
                             ("rubric_correct", "yes"), ("answer", "3"), ("rationale", "")):
            bad = copy.deepcopy(self.review)
            bad["cases"][0][field] = value
            with self.assertRaises(ValueError):
                attach(self.raw, bad)

    def test_source_review_preserves_primary_judge_and_evidence(self):
        self.review.update(judge_report_sha256="a" * 64,
                           judge_settings={"model": "qwen-fixed", "evidence_sha256": "b" * 64},
                           adjudication={"protocol": "source-audit-v1", "reviewer": "source reviewer"})
        evidence = [{"candidate_quote": "2", "comparison_source": "gold", "comparison_quote": "3"}]
        self.review["cases"][0]["evidence"] = evidence
        result = attach(self.raw, self.review)
        for field in ("judge_report_sha256", "judge_settings", "adjudication"):
            self.assertEqual(result["review"][field], self.review[field])
        self.assertEqual(result["cases"][0]["grade"]["evidence"], evidence)
        changed = copy.deepcopy(self.review)
        changed["cases"][0]["rationale"] += " Source checked."
        self.assertNotEqual(result["review"]["review_record_sha256"],
                            attach(self.raw, changed)["review"]["review_record_sha256"])


if __name__ == "__main__":
    unittest.main()
