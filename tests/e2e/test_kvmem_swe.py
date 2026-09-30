import hashlib
import json
import unittest
from types import ModuleType
from unittest.mock import patch

from kvmem_swe import frozen_rows, grading_container_network, proxy_environment


class SWEFrozenContract(unittest.TestCase):
    def test_network_override_is_scoped_and_restored_after_failure(self):
        calls = []
        class Collection:
            def create(self, image, **kwargs):
                calls.append((image, kwargs))
        module = ModuleType('docker.models.containers')
        module.ContainerCollection = Collection
        original = Collection.create
        environment = proxy_environment('http://192.0.2.1:7897')
        with patch.dict('sys.modules', {'docker.models.containers': module}):
            with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                with grading_container_network(environment):
                    client = Collection()
                    client.create('swebench/sweb.eval.x86_64.example:latest', environment=['PAGER=cat'])
                    client.create('unrelated/image:latest', environment={'KEEP': 'yes'})
                    raise RuntimeError('test interruption')
        self.assertIs(Collection.create, original)
        self.assertEqual(calls[0][1]['environment'], {'PAGER': 'cat', **environment})
        self.assertEqual(calls[1][1]['environment'], {'KEEP': 'yes'})
        self.assertIn('localhost', environment['NO_PROXY'])
        with self.assertRaises(ValueError):
            proxy_environment('http://user:secret@example.test:7897')

    def test_source_binding_and_answer_independent_pilot(self):
        rows = [{'instance_id': f'{repo}-{i}', 'repo': repo, 'patch': f'answer-{i}'}
                for repo in ('a', 'b') for i in range(4)]
        def encode(entries):
            return ''.join(json.dumps(row) + '\n' for row in entries).encode()
        raw = encode(rows)
        lock = {'sha256': hashlib.sha256(raw).hexdigest(), 'count': 8,
                'instance_ids': [r['instance_id'] for r in rows]}
        self.assertEqual(frozen_rows(raw, lock), rows)
        pilot = frozen_rows(raw, lock, per_repo=1)
        self.assertEqual([r['repo'] for r in pilot], ['a', 'b'])
        for row in rows:
            row['patch'] = 'different answer'
        changed = encode(rows)
        with self.assertRaises(ValueError):
            frozen_rows(changed, lock)
        changed_lock = {**lock, 'sha256': hashlib.sha256(changed).hexdigest()}
        self.assertEqual([r['instance_id'] for r in pilot],
                         [r['instance_id'] for r in frozen_rows(changed, changed_lock, per_repo=1)])


if __name__ == '__main__':
    unittest.main()
