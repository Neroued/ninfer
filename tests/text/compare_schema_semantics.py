"""Compare byte-vocabulary native probe results with independent original-schema oracle."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from schema_semantics_oracle import oracle


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases',required=True,type=Path)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args = parser.parse_args()
    cases = load_jsonl(args.cases)
    rows = load_jsonl(args.results)
    results = {row['id']:row for row in rows}
    errors, subsets = [], []
    counts = Counter()
    if len(results) != len(rows):
        errors.append({'kind':'duplicate_native_case_ids'})
    for case in cases:
        counts['cases'] += 1
        expected, details = oracle(case['schema'],case.get('wire',''),case.get('wire_bytes'))
        if expected != case['expected_accept']:
            errors.append({'id':case['id'],'kind':'corpus_oracle_disagreement','recomputed':expected})
        native = results.get(case['id'])
        if native is None:
            errors.append({'id':case['id'],'kind':'missing_native_result'})
            continue
        if type(native.get('accepted')) is not bool or native.get('compile_only') is True:
            errors.append({'id':case['id'],'kind':'malformed_native_acceptance_result','native':native})
        if native.get('internal_error'):
            errors.append({'id':case['id'],'kind':'native_internal_error','error':native['internal_error']})
        accepted = native.get('accepted') is True
        if accepted and not expected:
            # Unsupported/context/serialization labels never forgive invalid
            # acceptance, including cases that were supposed to fail compile.
            counts['accepted_invalid'] += 1
            errors.append({'id':case['id'],'kind':'accepted_invalid','native':native,'oracle':details})
        compile_error = bool(native.get('compile_error'))
        if compile_error != case['expected_compile_error']:
            counts['compile_contract_failures'] += 1
            errors.append({'id':case['id'],'kind':'compile_contract_failure','expected':case['expected_compile_error'],
                           'native_error':native.get('compile_error'),'contract':case.get('compile_error_contract')})
        elif compile_error:
            counts['expected_compile_errors'] += 1
        if not compile_error and not case['expected_compile_error']:
            if expected and not accepted:
                if case.get('declared_serialization_subset'):
                    counts['documented_serialization_rejections'] += 1
                    subsets.append({'id':case['id'],'kind':'documented_serialization_subset',
                                    'reason':case['declared_serialization_subset'],'native':native})
                else:
                    counts['rejected_valid'] += 1
                    errors.append({'id':case['id'],'kind':'rejected_valid','native':native,'oracle':details})
            elif accepted == expected:
                counts['matched_acceptance'] += 1
            if not expected and not accepted and case.get('expected_rejection_offset_max') is not None:
                if type(native.get('accepted_prefix_tokens')) is not int:
                    errors.append({'id':case['id'],'kind':'missing_native_prefix_measurement','native':native})
                elif native['accepted_prefix_tokens'] > case['expected_rejection_offset_max']:
                    counts['unsafe_exhausted_enum_prefix'] += 1
                    errors.append({'id':case['id'],'kind':'exhausted_enum_prefix_not_masked_before_comma',
                        'maximum':case['expected_rejection_offset_max'],'native':native})
    extra = sorted(set(results)-{case['id'] for case in cases})
    if extra:
        errors.append({'kind':'extra_native_results','ids':extra})
    report = {'passed':not errors,'counts':dict(counts),'errors':errors,'documented_serialization_subsets':subsets,
              'invalid_acceptance_policy':'Always fatal; unsupported contracts and serialization subsets cannot waive it.'}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'counts':report['counts'],'errors':len(errors)}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
