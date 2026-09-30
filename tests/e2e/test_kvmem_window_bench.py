import unittest

from kvmem_window_bench import summarize_placements, validate_execution_trace


class PlacementEvidenceTests(unittest.TestCase):
    def test_single_lane_baseline_rejects_batched_or_unretrieved_runs(self):
        retrieval = 'KVMEM retrieval scored=100 selected=30 promoted=10 demoted=10 lane=0\n'
        trace = retrieval + retrieval
        records = [{'event': 'throughput', 'decode_batch': {'rounds': 100, 'row_rounds': 100},
                    'scheduler': {'running': 1}}]
        result = validate_execution_trace(trace, 'mtp', 1, records)
        self.assertEqual(result['single_lane_telemetry_intervals'], 1)
        with self.assertRaisesRegex(AssertionError, 'exclusively'):
            validate_execution_trace(trace + 'KVMEM decode backend=mtp lanes=2\n', 'mtp', 1, records)
        with self.assertRaisesRegex(AssertionError, 'two separate'):
            validate_execution_trace(retrieval, 'mtp', 1, records)
        with self.assertRaisesRegex(AssertionError, 'scheduler evidence'):
            validate_execution_trace(trace, 'mtp', 1)
        records[0]['decode_batch']['row_rounds'] = 101
        with self.assertRaisesRegex(AssertionError, 'scheduler evidence'):
            validate_execution_trace(trace, 'mtp', 1, records)

    def test_demoting_a_current_host_replica_is_not_a_transfer(self):
        trace = (
            'KVPLACEMENT phase=decode row=0 planes=64 mapped=100 selected=20 '
            'demoted=4 promoted=0 d2h_pages=0 d2h_bytes=0 h2d_bytes=0 '
            'd2h_submit_wait_ms=0 h2d_submit_wait_ms=0 total_ms=0.5\n'
            'KVPLACEMENT phase=decode row=1 planes=64 mapped=100 selected=20 '
            'demoted=2 promoted=1 d2h_pages=1 d2h_bytes=2162688 h2d_bytes=2162688 '
            'd2h_submit_wait_ms=1.5 h2d_submit_wait_ms=2.5 total_ms=5\n'
            'KVPLACEMENT phase=prefill row=0 planes=4 mapped=100 selected=20 '
            'demoted=1 promoted=0 d2h_pages=1 d2h_bytes=135168 h2d_bytes=0 '
            'd2h_submit_wait_ms=0.1 h2d_submit_wait_ms=0 total_ms=0.2\n')
        groups = summarize_placements(trace)
        decode = groups['decode/planes-64']
        self.assertEqual(decode['calls'], 2)
        self.assertEqual(decode['demoted'], 6)
        self.assertEqual(decode['d2h_pages'], 1)
        self.assertEqual(decode['d2h_bytes'], 2162688)
        self.assertEqual(decode['h2d_bytes'], 2162688)
        self.assertEqual(decode['total_ms'], 5.5)
        self.assertEqual(groups['prefill/planes-4']['d2h_bytes'], 135168)

    def test_missing_measurements_are_not_reported_as_zero_transfers(self):
        with self.assertRaisesRegex(AssertionError, 'missing actual'):
            summarize_placements('server ready\nKVMEM decode backend=mtp lanes=2\n')


if __name__ == '__main__':
    unittest.main()
