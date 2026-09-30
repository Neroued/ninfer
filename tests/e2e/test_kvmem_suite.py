import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kvmem_suite import Suite, parse_sse, validate_answer, validate_json, validate_dual_lane_trace


class ResponseContractTests(unittest.TestCase):
    def test_two_http_requests_are_not_proof_of_parallel_sparse_execution(self):
        placements = ("KVMEM retrieval scored=12 selected=20 promoted=3 demoted=2 lane=0\n"
                      "KVMEM retrieval scored=11 selected=20 promoted=4 demoted=1 lane=1\n")
        with self.assertRaisesRegex(AssertionError, 'no verified'):
            validate_dual_lane_trace(placements, 'mtp')
        with self.assertRaisesRegex(AssertionError, 'no verified'):
            validate_dual_lane_trace('KVMEM decode backend=mtp lanes=2\n', 'mtp')
        result = validate_dual_lane_trace(placements + 'KVMEM decode backend=mtp lanes=2\n', 'mtp')
        self.assertEqual(result['retrieved_lanes'], [0, 1])

    def test_memory_guard_stops_only_owned_server_and_fails_report(self):
        with TemporaryDirectory() as directory:
            suite = Suite(SimpleNamespace(output=Path(directory), port=8120, profile='smoke'))
            suite.proc = Mock(pid=12345)
            suite.proc.poll.return_value = None
            with patch('kvmem_suite.os.killpg') as kill:
                suite.check_memory_headroom(4 * 1024**2, 0)
                kill.assert_not_called()
                suite.check_memory_headroom(200 * 1024, 400 * 1024)
                self.assertEqual(kill.call_args.args[0], 12345)
            self.assertTrue((Path(directory) / 'memory-abort.json').exists())
            suite.log.write_text('clean log')
            with self.assertRaisesRegex(AssertionError, 'memory guard'):
                suite.check_log()

    def test_stream_error_after_http_success(self):
        with self.assertRaisesRegex(AssertionError, "SSE error"):
            parse_sse(['data: {"error":{"message":"bad_alloc"}}'])

    def test_truncated_stream_is_failure(self):
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            parse_sse(['data: {"choices":[{"delta":{"content":"hello"}}]}'])

    def test_empty_done_is_failure(self):
        with self.assertRaises(AssertionError):
            parse_sse(['data: [DONE]'])

    def test_complete_stream(self):
        result = parse_sse([
            'data: {"choices":[{"delta":{"content":"READY"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":2}}',
            'data: [DONE]'])
        self.assertEqual(result["text"], "READY")

    def test_json_requires_generation_and_usage(self):
        with self.assertRaises(AssertionError):
            validate_json({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
        with self.assertRaises(AssertionError):
            validate_json({"error": "failed"})

    def test_mentioning_answer_is_not_recall(self):
        with self.assertRaisesRegex(AssertionError, "incorrect final answer"):
            validate_answer({"text": "I cannot find ORCHID-7392 in the document."}, "ORCHID-7392")
        result = validate_answer({"text": "Explanation.\nORCHID-7392"}, "ORCHID-7392")
        self.assertFalse(result["exact_answer_format"])
        self.assertTrue(validate_answer({"text": "ORCHID-7392"}, "ORCHID-7392")["exact_answer_format"])


if __name__ == "__main__":
    unittest.main()
