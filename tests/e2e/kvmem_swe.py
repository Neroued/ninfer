#!/usr/bin/env python3
"""Run frozen SWE mini agentic fixtures with NAS EvalScope 1.12.0.

This is an executable-test supplement, not the 500-question LongMemEval gate.
Always use a fresh output directory; cached model answers are never reused.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import urllib.parse


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def proxy_environment(proxy):
    if proxy is None:
        return {}
    url = urllib.parse.urlsplit(proxy)
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ('', '/')):
        raise ValueError('container proxy must be an HTTP(S) endpoint without credentials or URL data')
    return {'HTTP_PROXY': proxy, 'HTTPS_PROXY': proxy, 'http_proxy': proxy, 'https_proxy': proxy,
            'NO_PROXY': 'localhost,127.0.0.1,::1', 'no_proxy': 'localhost,127.0.0.1,::1'}


@contextmanager
def grading_container_network(environment):
    if not environment:
        yield
        return
    from docker.models.containers import ContainerCollection
    original = ContainerCollection.create

    def create(collection, image, *args, **kwargs):
        # EvalScope's separate SWE grading container does not consume task sandbox
        # env_vars. Scope the SDK override to this run's standard SWE image family.
        if isinstance(image, str) and image.startswith('swebench/sweb.eval.'):
            prior = kwargs.get('environment') or {}
            if isinstance(prior, list):
                prior = dict(item.split('=', 1) for item in prior)
            kwargs['environment'] = {**prior, **environment}
        return original(collection, image, *args, **kwargs)

    ContainerCollection.create = create
    try:
        yield
    finally:
        ContainerCollection.create = original


def frozen_rows(raw, lock, per_repo=0, seed=74191):
    if hashlib.sha256(raw).hexdigest() != lock['sha256']:
        raise ValueError('SWE source hash mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [row['instance_id'] for row in rows]
    if len(rows) != lock['count'] or len(set(ids)) != len(ids) or ids != lock['instance_ids']:
        raise ValueError('SWE source IDs/count mismatch')
    if per_repo:
        groups = defaultdict(list)
        for row in rows:
            groups[row['repo']].append(row)
        rows = [row for repo in sorted(groups) for row in sorted(groups[repo],
            key=lambda row: hashlib.sha256(f"{seed}:{row['instance_id']}".encode()).hexdigest())[:per_repo]]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-url', required=True)
    parser.add_argument('--model', default='kvmem-test')
    parser.add_argument('--pilot-per-repo', type=int, default=0)
    parser.add_argument('--seed', type=int, default=74191)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--container-proxy', help='HTTP(S) proxy reachable from Docker; no credentials')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    network_environment = proxy_environment(args.container_proxy)
    if args.pilot_per_repo < 0 or args.max_steps <= 0:
        parser.error('invalid pilot size or step bound')
    url = urllib.parse.urlsplit(args.api_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.port in (8080, 8081):
        parser.error('use an explicit independent development endpoint, outside ports 8080/8081')

    import evalscope
    from evalscope.api.agent import NativeAgentConfig
    from evalscope.api.dataset import RemoteDataLoader, Sample
    from evalscope.config import SandboxTaskConfig, TaskConfig
    from evalscope.run import run_task

    versions = {name: importlib.metadata.version(name) for name in ('evalscope', 'swebench', 'datasets')}
    if versions['evalscope'] != '1.12.0' or versions['swebench'] != '4.1.0':
        raise ValueError('this protocol requires EvalScope 1.12.0 and swebench 4.1.0')
    raw = args.data.read_bytes()
    lock = json.loads(args.source_lock.read_bytes())
    rows = frozen_rows(raw, lock, args.pilot_per_repo, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_dir = args.output.resolve() / 'fixtures'
    fixture_dir.mkdir()
    fixture = fixture_dir / 'test.jsonl'
    fixture.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows),
                       encoding='utf-8')
    # Exercise the actual local loader, including preservation of all test metadata.
    loaded = RemoteDataLoader(str(fixture_dir), split='test', data_source='local',
        sample_fields=lambda record: Sample(input=record['problem_statement'], metadata=record), auto_id=False).load()
    if [sample.metadata['instance_id'] for sample in loaded] != [row['instance_id'] for row in rows]:
        raise ValueError('EvalScope loaded a different fixture set')
    ds = 'swe_bench_verified_mini_agentic'
    config = TaskConfig(model=args.model, api_url=args.api_url, api_key='EMPTY',
        eval_type='openai_api', datasets=[ds],
        dataset_args={ds: {'local_path': str(fixture_dir), 'extra_params': {
            'build_docker_images': True, 'pull_remote_images_if_available': True}}},
        eval_batch_size=1, seed=args.seed, no_timestamp=True, use_cache=None,
        work_dir=str(args.output.resolve() / 'evalscope'),
        agent_config=NativeAgentConfig(strategy='swe_bench_toolcall', max_steps=args.max_steps),
        sandbox=SandboxTaskConfig(default_config={'env_vars': network_environment}),
        generation_config={'temperature': 0, 'seed': args.seed, 'max_tokens': 4096,
            'top_p': 1, 'frequency_penalty': 0, 'presence_penalty': 0,
            'stream': False, 'timeout': 1800, 'retries': 0,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
    package = Path(evalscope.__file__).parent
    sources = {}
    for subdir in ('agent', 'api/agent', 'api/benchmark/adapters', 'benchmarks/swe_bench'):
        for path in sorted((package / subdir).rglob('*.py')):
            sources[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    images = subprocess.run(['docker', 'image', 'ls', '--digests', '--no-trunc',
        '--format', '{{json .}}'], check=True, capture_output=True, text=True)
    manifest = {'kind': 'swe-mini-agentic-supplement', 'source': lock,
        'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'instance_ids': [row['instance_id'] for row in rows], 'seed': args.seed,
        'pilot_per_repo': args.pilot_per_repo, 'versions': versions,
        'evalscope_source_sha256': sources,
        'existing_swe_images': [json.loads(line) for line in images.stdout.splitlines() if 'swebench/' in line],
        'container_network_environment': network_environment,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'task_config': config.model_dump(mode='json'), 'prepared': True,
        'evalscope_finished': False, 'equivalence_established': False}
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(encode(manifest), encoding='utf-8')
    print(encode({'prepared': True, 'cases': len(rows), 'ids': manifest['instance_ids'],
                  'fixture_sha256': manifest['fixture_sha256'], 'versions': versions}), flush=True)
    if args.prepare_only:
        return 0
    try:
        with grading_container_network(network_environment):
            run_task(task_cfg=config)
        manifest['evalscope_finished'] = True
    except BaseException as exc:
        manifest['execution_error_type'] = type(exc).__name__
        raise
    finally:
        # EvalScope completion alone is not a full-set pass. Inspect every instance's
        # test result, execution errors and trace before any paired acceptance claim.
        manifest_path.write_text(encode(manifest), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
