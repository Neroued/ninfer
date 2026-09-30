#!/usr/bin/env python3
"""Freeze source-bound grading evidence; never use this file as inference input."""
import argparse
import json
from pathlib import Path

from kvmem_longmemeval import source_entries
from kvmem_quality import encode


def evidence_case(entry):
    ids = entry['haystack_session_ids']
    dates, sessions = entry['haystack_dates'], entry['haystack_sessions']
    if len(ids) != len(dates) or len(ids) != len(sessions):
        raise ValueError('unaligned source sessions')
    required = set(entry['answer_session_ids'])
    if required - set(ids):
        raise ValueError('missing evidence session')
    evidence, selected = [], []
    for identity, date, turns in zip(ids, dates, sessions):
        if identity not in required:
            continue
        selected.append(identity)
        evidence.append(f'Session {len(selected)}; date: {date}')
        for turn in turns:
            if turn['role'] not in ('user', 'assistant') or not isinstance(turn['content'], str):
                raise ValueError('unsupported source turn')
            evidence.append(f"{turn['role']}:\n{turn['content']}")
    return {'id': entry['question_id'], 'family': entry['question_type'],
            'question': entry['question'], 'question_date': entry['question_date'],
            'gold': entry['answer'], 'session_ids': selected, 'history': '\n\n'.join(evidence)}


def bind_evidence(cases, document, source):
    if document['kind'] != 'longmemeval-grading-evidence' or document['source'] != source:
        raise ValueError('grading evidence source mismatch')
    by_id = {row['id']: row for row in document['cases']}
    if len(by_id) != len(document['cases']):
        raise ValueError('duplicate evidence case')
    result = []
    for case in cases:
        if case['id'] not in by_id:
            raise ValueError('missing question evidence')
        row = by_id[case['id']]
        for key in ('family', 'question', 'gold'):
            if row[key] != case[key]:
                raise ValueError(f'grading evidence {key} mismatch')
        result.append({**case, 'question_date': row['question_date'], 'history': row['history']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path,
                        default=Path(__file__).with_name('longmemeval-source.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.source_lock.read_bytes())
    entries = source_entries(args.data.read_bytes(), lock)
    result = {'kind': 'longmemeval-grading-evidence', 'source': lock,
              'selection': 'all complete answer_session_ids sessions in original history order; '
                           'independent of model answers; grading only',
              'coverage': 'Relevant evidence sessions, not exhaustive unrelated history. '
                          'Do not infer absence from omission; unresolved claims need source review.',
              'cases': [evidence_case(entry) for entry in entries]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(encode(result))
    print(encode({'cases': len(result['cases']),
                  'maximum_evidence_chars': max(len(c['history']) for c in result['cases'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
