import copy
import json
import unittest

from kvmem_longmemeval import convert_case, digest, prepare, dependence_clusters


def sample():
    return {"question_id": "question-1", "question_type": "knowledge-update", "answer": "secret gold",
            "question": "What is current?", "question_date": "2024-03-03",
            "haystack_session_ids": ["hidden-evidence-id"], "haystack_dates": ["2024-02-02"],
            "answer_session_ids": ["hidden-evidence-id"],
            "haystack_sessions": [[{"role": "user", "content": "I prefer blue.", "has_answer": True},
                                   {"role": "assistant", "content": "Blue noted."}]]}


class LongMemEvalContract(unittest.TestCase):
    def test_gold_and_evidence_labels_never_in_prompt(self):
        original = sample()
        before = copy.deepcopy(original)
        converted = convert_case(original)
        prompt = json.dumps(converted["messages"])
        for hidden in ("secret gold", "has_answer", "answer_session_ids", "hidden-evidence-id"):
            self.assertNotIn(hidden, prompt)
        self.assertIn("2024-02-02", prompt)
        self.assertIn("Blue noted.", prompt)
        self.assertEqual(converted["messages"][-1]["role"], "user")
        self.assertIn("What is current?", converted["messages"][-1]["content"])
        self.assertEqual(original, before)

    def test_reject_misaligned_history(self):
        entry = sample()
        entry["haystack_dates"] = []
        with self.assertRaises(ValueError):
            convert_case(entry)

    def test_selection_frozen_and_source_hash_checked(self):
        entries = []
        for i in range(8):
            entry = sample()
            entry["question_id"] = f"q{i}" + ("_abs" if i % 2 else "")
            entries.append(entry)
        data = json.dumps(entries).encode()
        lock = {"bytes": len(data), "sha256": digest(data), "questions": len(entries)}
        selected = prepare(data, lock, 2, 74191)
        self.assertEqual(selected, prepare(data, lock, 2, 74191))
        self.assertEqual(len(selected["cases"]), 4)
        self.assertEqual(sum(c["abstention"] for c in selected["cases"]), 2)
        self.assertFalse(selected["selection"]["full_500"])
        with self.assertRaises(ValueError):
            prepare(data + b" ", lock, 2, 74191)

    def test_cluster_transitive_shared_evidence_and_abstention(self):
        entries = []
        for qid, evidence in [('a', ['e1']), ('a_abs', ['e2']), ('b', ['e2']), ('c', ['e3'])]:
            entries.append({**sample(), 'question_id': qid, 'answer_session_ids': evidence})
        data = json.dumps(entries).encode()
        lock = {'bytes': len(data), 'sha256': digest(data), 'questions': len(entries)}
        groups = dependence_clusters(data, lock)
        self.assertEqual(groups['clusters'], [
            {'id': 'a', 'question_ids': ['a', 'a_abs', 'b']}, {'id': 'c', 'question_ids': ['c']}])
        # Gold answer changes cannot alter the grouping rule.
        for entry in entries:
            entry['answer'] = 'changed gold'
        changed = json.dumps(entries).encode()
        updated_lock = {'bytes': len(changed), 'sha256': digest(changed), 'questions': len(entries)}
        self.assertEqual(dependence_clusters(changed, updated_lock)['clusters'], groups['clusters'])


if __name__ == "__main__":
    unittest.main()
