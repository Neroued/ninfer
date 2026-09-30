import copy
import unittest

from kvmem_judge_evidence import evidence_case, bind_evidence


class JudgeEvidenceContract(unittest.TestCase):
    def setUp(self):
        self.entry = {'question_id': 'a', 'question_type': 'temporal-reasoning',
                      'question': 'Which first?', 'question_date': '2023/05/30', 'answer': 'Album',
                      'haystack_session_ids': ['irrelevant', 'necklace', 'album'],
                      'answer_session_ids': ['album', 'necklace'],
                      'haystack_dates': ['D0', 'D1', 'D2'],
                      'haystack_sessions': [
                          [{'role': 'user', 'content': 'irrelevant data'}],
                          [{'role': 'user', 'content': 'My sister: I got her a necklace last weekend.',
                            'has_answer': True}, {'role': 'assistant', 'content': 'Happy birthday.'}],
                          [{'role': 'user', 'content': 'Photo album for my mom two weeks ago.'}]]}

    def test_evidence_uses_whole_source_sessions_in_history_order(self):
        before = copy.deepcopy(self.entry)
        row = evidence_case(self.entry)
        self.assertEqual(row['session_ids'], ['necklace', 'album'])
        self.assertIn('Happy birthday.', row['history'])
        self.assertIn('My sister:', row['history'])
        self.assertNotIn('irrelevant data', row['history'])
        self.assertNotIn('has_answer', row['history'])
        self.assertEqual(self.entry, before)
        self.entry['answer_session_ids'].append('missing')
        with self.assertRaisesRegex(ValueError, 'missing evidence session'):
            evidence_case(self.entry)

    def test_binding_rejects_source_or_question_mismatch(self):
        row = evidence_case(self.entry)
        doc = {'kind': 'longmemeval-grading-evidence', 'source': {'sha256': 's'}, 'cases': [row]}
        case = {key: row[key] for key in ('id', 'family', 'question', 'gold')}
        case.update(answer='Album', strict_correct=None)
        bound = bind_evidence([case], doc, {'sha256': 's'})
        self.assertEqual(bound[0]['question_date'], '2023/05/30')
        self.assertIn('My sister', bound[0]['history'])
        with self.assertRaisesRegex(ValueError, 'source'):
            bind_evidence([case], doc, {'sha256': 'changed'})
        with self.assertRaisesRegex(ValueError, 'question'):
            bind_evidence([{**case, 'question': 'Different'}], doc, {'sha256': 's'})

    def test_repeated_source_ids_preserve_all_occurrences(self):
        self.entry['haystack_session_ids'].append('necklace')
        self.entry['haystack_dates'].append('D3')
        self.entry['haystack_sessions'].append([{'role': 'user', 'content': 'A later clarification.'}])
        row = evidence_case(self.entry)
        self.assertEqual(row['session_ids'], ['necklace', 'album', 'necklace'])
        self.assertIn('A later clarification.', row['history'])


if __name__ == '__main__':
    unittest.main()
