#!/usr/bin/env python3
"""Exercise per-lane JSONL accounting on an otherwise idle two-lane server.

The server must have --request-log-jsonl enabled and periodic stats enabled.
No server process is started or stopped by this check.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import urllib.request
import uuid


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    startup = None
    with args.log.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'server_start':
                startup = event
    require(startup is not None and startup['schema_version'] == 22,
            'Start a server with lane-aware JSONL schema 22')
    require(startup['engine']['max_concurrency'] == 2, 'This smoke requires two lanes')
    interval = startup['engine']['log_stats_interval_ms'] / 1000
    require(0 < interval <= 5, 'Use a periodic stats interval between 1 and 5000 ms')
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        model = json.load(response)['data'][0]['id']
    # Flush any preceding deployment smoke before measuring this batch.
    time.sleep(interval + 0.5)
    offset = args.log.stat().st_size
    tag = uuid.uuid4().hex
    prompts = [f'Test {tag}/{i}. Count every integer from {i + 1} to 10000, separated by commas. '
               'Do not skip or abbreviate the list. No commentary or code blocks.'
               for i in range(2)]

    def generate(item: tuple[str, int]) -> dict:
        prompt, limit = item
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': limit, 'temperature': 0,
                   'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(args.url + '/v1/chat/completions',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    responses = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses.extend(pool.map(generate, [(prompt, 4096) for prompt in prompts]))
        # Reuse physical lanes and exercise a reused prompt prefix without counter reset.
        responses.extend(pool.map(generate, [(prompt, 256) for prompt in prompts]))
    time.sleep(interval * 2 + 0.5)
    with args.log.open('rb') as stream:
        stream.seek(offset)
        records = [json.loads(line) for line in stream if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'events.jsonl').write_text(
        ''.join(json.dumps(record) + '\n' for record in records))
    (args.output / 'responses.json').write_text(json.dumps(responses, ensure_ascii=False, indent=2))

    done = [r for r in records if r['event'] == 'request_done']
    windows = [r for r in records if r['event'] == 'throughput']
    require(len(done) == 4, 'Expected exactly four completions on the otherwise idle server')
    require(all(r['server_instance_id'] == startup['server_instance_id'] for r in records),
            'Server restarted during the smoke')
    require({r['execution']['lane_id'] for r in done} == {0, 1},
            'Requests did not exercise both physical lanes')
    require(len({r['execution']['engine_request_id'] for r in done}) == 4,
            'Engine request identities were reused')
    require(any(r['result']['prefix_cache_hit_tokens'] > 0 for r in done),
            'Repeated prompts did not exercise prefix reuse')
    require(all(r['result']['completion_tokens'] > 1 for r in done), 'Unexpected empty completion')
    observed_ids = set()
    observed_states = set()
    paired_decode = False
    totals = {lane: {'computed_prefill': 0, 'committed_decode': 0} for lane in range(2)}
    for window in windows:
        lanes = window['lanes']
        require([lane['lane_id'] for lane in lanes] == [0, 1], 'Missing or reordered lane entry')
        require(sum(lane['decode_rounds'] for lane in lanes) == window['decode_batch']['row_rounds'],
                'Per-lane decode rounds disagree with aggregate row-rounds')
        for metric in ('computed_prefill', 'committed_decode'):
            require(sum(lane['tokens'][metric] for lane in lanes) == window['tokens'][metric],
                    f'Per-lane {metric} does not conserve aggregate tokens')
        for lane in lanes:
            observed_states.add(lane['state'])
            if lane['engine_request_id'] is not None:
                observed_ids.add(lane['engine_request_id'])
            require((lane['state'] == 'idle') == (lane['engine_request_id'] is None),
                    'Idle and ownership gauges disagree')
            for metric, rate in (('computed_prefill', 'prefill'), ('committed_decode', 'decode')):
                count = lane['tokens'][metric]
                require(0 <= count < 10**7, 'Lane counter reset/underflow or invalid workload')
                totals[lane['lane_id']][metric] += count
                require(math.isclose(lane['throughput_tokens_per_second'][rate],
                                     count / window['interval_seconds'], rel_tol=1e-12, abs_tol=1e-12),
                        'Lane throughput used a different interval denominator')
        paired_decode |= all(lane['state'] == 'decode_ready' for lane in lanes)
    require(paired_decode, 'No periodic snapshot observed two decode-ready lanes')
    require(windows and all(lane['state'] == 'idle' for lane in windows[-1]['lanes']),
            'Final idle transition was not recorded')
    for lane_id in range(2):
        requests = [r for r in done if r['execution']['lane_id'] == lane_id]
        require(totals[lane_id]['computed_prefill'] ==
                sum(r['result']['computed_prefill_tokens'] for r in requests),
                'Lane prefill lost work or counted cached tokens')
        require(totals[lane_id]['committed_decode'] ==
                sum(r['result']['completion_tokens'] - 1 for r in requests),
                'Lane decode lost accepted output or counted the prefill token')
    require(observed_ids <= {r['execution']['engine_request_id'] for r in done},
            'Snapshot Engine ID cannot be joined to request completion')
    summary = {'status': 'passed', 'requests': len(done), 'throughput_intervals': len(windows),
               'paired_decode_observed': paired_decode, 'states_observed': sorted(observed_states),
               'lane_tokens': totals, 'server_instance_id': startup['server_instance_id']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
