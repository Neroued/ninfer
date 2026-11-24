import json
import tempfile
import unittest
from pathlib import Path

from ninfer_eval.backends.evalscope_records import (
    prepare_resume,
    records,
    summarize_records,
)


class EvalScopeRecordsTest(unittest.TestCase):
    def write_records(self, root, kind, rows):
        path = root / kind / "model" / "aime26_default.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_counts_tokens_truncation_and_repeated_question_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_records(
                root,
                "predictions",
                [
                    {
                        "index": 0,
                        "model_output": {
                            "choices": [
                                {"stop_reason": "stop", "message": {"content": "42"}}
                            ],
                            "usage": {"input_tokens": 10, "output_tokens": 20},
                        },
                    },
                    {
                        "index": 1,
                        "model_output": {
                            "choices": [
                                {
                                    "stop_reason": "max_tokens",
                                    "message": {
                                        "content": [
                                            {
                                                "type": "reasoning",
                                                "reasoning": "unfinished",
                                            }
                                        ],
                                    },
                                }
                            ],
                            "usage": {"input_tokens": 10, "output_tokens": 100},
                        },
                    },
                ],
            )
            self.write_records(
                root,
                "reviews",
                [
                    {
                        "index": 0,
                        "sample_score": {
                            "group_id": 0,
                            "score": {
                                "value": {"acc": 1},
                                "extracted_prediction": "42",
                            },
                        },
                    },
                    {
                        "index": 1,
                        "sample_score": {
                            "group_id": 0,
                            "score": {"value": {"acc": 0}, "extracted_prediction": ""},
                        },
                    },
                ],
            )
            diagnostics, questions = summarize_records(root)
            self.assertEqual(diagnostics["predictions"], 2)
            self.assertEqual(diagnostics["output_tokens"], 120)
            self.assertEqual(diagnostics["truncated"], 1)
            self.assertEqual(diagnostics["empty_answers"], 1)
            self.assertEqual(diagnostics["unparsed_answers"], 1)
            self.assertEqual(questions[0]["samples"], 2)
            self.assertEqual(questions[0]["correct"], 1)

    def test_resume_archives_only_infrastructure_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"index": 0, "model_output": {"error": None}},
                {"index": 1, "model_output": {"error": "timeout"}},
                {
                    "index": 2,
                    "model_output": {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"error": "400", "error_message": "bad request"}'
                                },
                            }
                        ]
                    },
                },
                {
                    "index": 3,
                    "model_output": {
                        "choices": [
                            {"stop_reason": "max_tokens", "message": {"content": ""}}
                        ],
                    },
                },
            ]
            predictions = self.write_records(root, "predictions", rows)
            reviews = self.write_records(
                root, "reviews", [{"index": i} for i in range(4)]
            )
            prepare_resume(root)
            self.assertEqual([r["index"] for r in records(predictions)], [0, 3])
            self.assertEqual([r["index"] for r in records(reviews)], [0, 3])
            archived = root / "failed-attempts" / predictions.relative_to(root)
            self.assertEqual([r["index"] for r in records(archived)], [1, 2])
            prepare_resume(root)
            self.assertEqual(len(list(records(archived))), 2)


if __name__ == "__main__":
    unittest.main()
