import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from kvmem_vision_suite import VisionSuite


class DraftEvidenceTests(unittest.TestCase):
    def evidence(self, backend='dflash2', width=7, accepted=4, trace=None):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            suite = VisionSuite(SimpleNamespace(output=output, port=8120, profile='vision',
                                               window=64 if trace is not None else 0))
            (output / 'requests.jsonl').write_text(json.dumps({
                'event': 'request_done', 'speculative': {
                    'backend': backend, 'draft_window': width,
                    'drafted_tokens': 21, 'accepted_tokens': accepted}}) + '\n')
            suite.log.write_text(trace or '')
            return suite.speculative_evidence()

    def test_http_success_cannot_hide_backend_fallback_or_wrong_window(self):
        for backend, width, accepted in [('none', 7, 4), ('dflash2', 3, 4), ('dflash2', 7, 0)]:
            with self.assertRaises(AssertionError):
                self.evidence(backend, width, accepted)

    def test_short_circuited_zero_extent_is_not_a_rejection(self):
        with self.assertRaisesRegex(AssertionError, 'Missing rejection'):
            self.evidence(trace='KVMEM draft lane=0 extent=0 accepted=0\n'
                                'KVMEM draft lane=0 extent=7 accepted=3\n'
                                'KVMEM draft lane=1 extent=7 accepted=7\n')

    def test_actual_draft_round_paths_are_recorded(self):
        result = self.evidence(trace='KVMEM draft lane=0 extent=7 accepted=0\n'
                                     'KVMEM draft lane=0 extent=7 accepted=3\n'
                                     'KVMEM draft lane=1 extent=7 accepted=7\n')
        self.assertEqual(result['round_acceptance'], {'zero': 1, 'partial': 1, 'full': 1})


if __name__ == '__main__':
    unittest.main()
