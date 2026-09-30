from copy import deepcopy
import hashlib
import struct

import pytest

from tools.artifact.attach_dflash2 import attach
from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def fixture(path, *, donor=False, tokenizer=b'tokens', width=2):
    specs = [TensorSpec('w', (2, 4), 'bf16', 'contiguous_le_v1'),
             TensorSpec('scale', (), 'fp32', 'contiguous_le_v1'),
             ResourceSpec('tokens', len(tokenizer)), ResourceSpec('extra', 3),
             ResourceSpec('import/dflash2/w', 3)]
    components = {'text': {'config': {'hidden_size': width},
                            'resources': {'tokenizer.json': 'tokens'}}}
    bindings = {'text/output_head': {'object': 'w'}, 'proposal/head': {'object': 'w'}}
    uses = [{'parameter': 'text/output_head', 'input': 'text/final_hidden',
             'activation_policy': 'AllowA4',
             'auxiliaries': {'activation_input_divisor': {'object': 'scale'}}},
            {'parameter': 'proposal/head', 'input': 'mtp/final_hidden',
             'activation_policy': 'A16Only'}]
    if donor:
        components['dflash2'] = {'target': 'text', 'config': {'num_hidden_layers': 5},
                                 'resources': {'test-resource': 'extra'}}
        view = {'parts': [{'object': 'w', 'range': [4, 8]}, {'object': 'w', 'range': [0, 4]}]}
        bindings.update({'dflash2/query': view, 'dflash2/context_key': deepcopy(view)})
        uses.extend([
            {'parameter': 'dflash2/query', 'input': 'dflash2/hidden', 'activation_policy': 'AllowA4',
             'auxiliaries': {'activation_input_divisor': {'object': 'scale'}}},
            {'parameter': 'text/output_head', 'input': 'dflash2/final_hidden',
             'activation_policy': 'AllowA8'},
            {'parameter': 'proposal/head', 'input': 'dflash2/final_hidden',
             'activation_policy': 'AllowA4'}])
    with ArtifactWriter(path, specs, components=components, bindings=bindings, uses=uses,
                        metadata={'name': 'donor' if donor else 'base'},
                        provenance={'fixture': True}) as writer:
        writer.write_object('w', bytes([9 if donor else 7]) * 16)
        writer.write_object('scale', struct.pack('<f', 5 if donor else 2))
        writer.write_object('tokens', tokenizer)
        writer.write_object('extra', b'NEW' if donor else b'OLD')
        writer.write_object('import/dflash2/w', b'abc')


def test_copy_preserves_base_shared_head_ranges_aliases_auxiliaries_and_collisions(tmp_path):
    base, donor, output = [tmp_path / f'{n}.ninfer' for n in ('base', 'donor', 'out')]
    fixture(base)
    fixture(donor, donor=True)
    before = hashlib.sha256(base.read_bytes()).digest()
    report = attach(base, donor, output, progress=lambda _: None)
    assert report['verified'] and report['base_objects'] == 5 and report['imported_objects'] == 3
    assert hashlib.sha256(base.read_bytes()).digest() == before
    with Artifact(base) as b, Artifact(donor) as d, Artifact(output) as o:
        for obj in b.objects:
            assert o.read_object(obj.id) == b.read_object(obj.id)
        for name, binding in b.directory.bindings.items():
            assert o.directory.bindings[name] == binding
        assert o.directory.components['text'] == b.directory.components['text']
        assert o.directory.metadata == b.directory.metadata
        assert o.directory.uses[:len(b.directory.uses)] == b.directory.uses
        view = o.directory.bindings['dflash2/query']
        assert view == o.directory.bindings['dflash2/context_key']
        assert [p['range'] for p in view['parts']] == [[4, 8], [0, 4]]
        assert view['parts'][0]['object'] == 'import/dflash2/w_'
        assert o.read_object('import/dflash2/w_') == d.read_object('w')
        own = next(u for u in o.directory.uses if u['parameter'] == 'dflash2/query')
        scalar = own['auxiliaries']['activation_input_divisor']['object']
        assert o.read_object(scalar) == struct.pack('<f', 5)
        head = next(u for u in o.directory.uses if u['input'] == 'dflash2/final_hidden')
        assert head == {**b.directory.uses[0], 'input': 'dflash2/final_hidden'}
        proposal = next(u for u in o.directory.uses if u['parameter'] == 'proposal/head'
                        and u['input'] == 'dflash2/final_hidden')
        assert proposal['activation_policy'] == 'A16Only'
        resource = o.directory.components['dflash2']['resources']['test-resource']
        assert o.read_object(resource) == b'NEW'
    with pytest.raises(FileExistsError):
        attach(base, donor, output)


@pytest.mark.parametrize('kwargs,message', [({'width': 3}, 'configs differ'),
                                           ({'tokenizer': b'other'}, 'tokenizer bytes differ')])
def test_incompatible_target_rejected_without_output(tmp_path, kwargs, message):
    base, donor, output = [tmp_path / f'{n}.ninfer' for n in ('base', 'donor', 'out')]
    fixture(base)
    fixture(donor, donor=True, **kwargs)
    with pytest.raises(ArtifactError, match=message):
        attach(base, donor, output)
    assert not output.exists()


def test_existing_draft_rejected(tmp_path):
    base, donor, output = [tmp_path / f'{n}.ninfer' for n in ('base', 'donor', 'out')]
    fixture(base, donor=True)
    fixture(donor, donor=True)
    with pytest.raises(ArtifactError, match='already contains'):
        attach(base, donor, output)
    assert not output.exists()
