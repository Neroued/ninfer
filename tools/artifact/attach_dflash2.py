"""Copy an existing DFlash2 component into a compatible v3 target, without requantizing.

Run under Linux: python -m tools.artifact.attach_dflash2 --base ... --donor ... --out ...
Geometry/tokenizer compatibility does not establish draft acceptance or model quality.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from .reader import Artifact
from .schema import ArtifactError, ResourceSpec, TensorObject, TensorSpec, binding_parts
from .writer import ArtifactWriter


def _references(binding):
    return [binding['object']] if 'object' in binding else [p['object'] for p in binding['parts']]


def _remap(binding, names):
    result = deepcopy(binding)
    if 'object' in result:
        result['object'] = names[result['object']]
    else:
        for part in result['parts']:
            part['object'] = names[part['object']]
    return result


def _spec(obj, name):
    if isinstance(obj, TensorObject):
        return TensorSpec(name, obj.shape, obj.format, obj.layout)
    return ResourceSpec(name, obj.bytes, obj.encoding)


def plan(base: Artifact, donor: Artifact):
    b, d = base.directory, donor.directory
    if 'dflash2' in b.components or any(n.startswith('dflash2/') for n in b.bindings):
        raise ArtifactError('base already contains DFlash2')
    component = deepcopy(d.components.get('dflash2'))
    if not component or component.get('target') != 'text':
        raise ArtifactError('donor must contain DFlash2 targeting text')
    if b.components['text']['config'] != d.components['text']['config']:
        raise ArtifactError('target text configs differ')
    for artifact in (base, donor):
        if 'tokenizer.json' not in artifact.directory.components['text'].get('resources', {}):
            raise ArtifactError('both artifacts must contain tokenizer.json')
    tokenizers = [a.read_object(a.directory.components['text']['resources']['tokenizer.json'])
                  for a in (base, donor)]
    if tokenizers[0] != tokenizers[1]:
        raise ArtifactError('tokenizer bytes differ')
    imported = {n: binding for n, binding in d.bindings.items() if n.startswith('dflash2/')}
    if not imported:
        raise ArtifactError('donor contains no DFlash2 bindings')
    ids = {oid for binding in imported.values() for oid in _references(binding)}
    ids.update(component.get('resources', {}).values())
    own_uses, shared_uses = [], []
    for use in d.uses:
        parameter, activation = use['parameter'], use['input']
        if parameter in imported:
            own_uses.append(use)
            for binding in use.get('auxiliaries', {}).values():
                ids.update(_references(binding))
        elif activation.startswith('dflash2/'):
            if parameter not in ('text/output_head', 'proposal/head') or activation != 'dflash2/final_hidden':
                raise ArtifactError(f'unsupported shared DFlash2 use: {parameter}@{activation}')
            # The output head is the BASE target's weight, so its activation policy and
            # any quantization auxiliaries must also come from the base target.
            candidates = [{k: deepcopy(v) for k, v in u.items() if k != 'input'}
                          for u in b.uses if u['parameter'] == parameter]
            if not candidates or any(u != candidates[0] for u in candidates[1:]):
                raise ArtifactError(f'base has missing or ambiguous head uses: {parameter}')
            target_use = candidates[0]
            for artifact in (base, donor):
                if parameter not in artifact.directory.bindings:
                    raise ArtifactError('missing shared output-head binding')
            sizes = [sum(end - begin for _, begin, end in binding_parts(
                a.directory.bindings[parameter], a.by_id)) for a in (base, donor)]
            if sizes[0] != sizes[1]:
                raise ArtifactError('shared output-head sizes differ')
            shared_uses.append({**deepcopy(target_use), 'input': activation})
    if sum(u['parameter'] == 'text/output_head' for u in shared_uses) != 1:
        raise ArtifactError('expected exactly one shared DFlash2 output-head use')
    if any(u['input'].startswith('dflash2/') for u in b.uses):
        raise ArtifactError('base already contains DFlash2 uses')
    objects = [obj for obj in donor.objects if obj.id in ids]
    if len(objects) != len(ids):
        raise ArtifactError('missing donor objects')
    used = set(base.by_id)
    names = {}
    for obj in objects:
        name = 'import/dflash2/' + obj.id
        while name in used:
            name += '_'
        names[obj.id] = name
        used.add(name)
    if 'resources' in component:
        component['resources'] = {role: names[oid] for role, oid in component['resources'].items()}
    components = {**deepcopy(b.components), 'dflash2': component}
    bindings = {**deepcopy(b.bindings), **{n: _remap(v, names) for n, v in imported.items()}}
    uses = deepcopy(list(b.uses))
    for original in own_uses:
        use = deepcopy(original)
        if 'auxiliaries' in use:
            use['auxiliaries'] = {role: _remap(v, names) for role, v in use['auxiliaries'].items()}
        uses.append(use)
    uses.extend(shared_uses)
    copies = [(base, obj.id, obj.id) for obj in base.objects]
    copies.extend((donor, obj.id, names[obj.id]) for obj in objects)
    specs = [_spec(a.object(old), new) for a, old, new in copies]
    provenance = deepcopy(b.provenance)
    imports = provenance.setdefault('component_imports', [])
    if not isinstance(imports, list):
        raise ArtifactError('component_imports provenance must be a list')
    imports.append({'component': 'dflash2', 'operation': 'encoded-byte-copy',
                    'source_file': donor.path.name, 'source_artifact_id': donor.artifact_id.hex(),
                    'source_provenance': deepcopy(d.provenance),
                    'base_artifact_id': base.artifact_id.hex()})
    return specs, components, bindings, uses, provenance, copies


def attach(base_path, donor_path, output_path, *, progress=print):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    report_path = Path(str(output_path) + '.attach.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.monotonic()
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        specs, components, bindings, uses, provenance, copies = plan(base, donor)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        required = sum(a.by_id[old].bytes for a, old, _ in copies) + 64 * 1024**2
        if shutil.disk_usage(output_path.parent).free < required:
            raise ArtifactError(f'insufficient output disk space; need approximately {required} bytes')
        report = {'base': str(base.path), 'donor': str(donor.path), 'output': str(output_path),
                  'base_artifact_id': base.artifact_id.hex(), 'donor_artifact_id': donor.artifact_id.hex(),
                  'base_objects': len(base.objects), 'imported_objects': len(copies) - len(base.objects),
                  'imported_bindings': len(bindings) - len(base.directory.bindings),
                  'imported_uses': len(uses) - len(base.directory.uses), 'objects': []}
        progress(f'Copying {len(base.objects)} base objects and {report["imported_objects"]} draft objects')
        with ArtifactWriter(output_path, specs, components=components, bindings=bindings,
                            uses=uses, metadata=deepcopy(base.directory.metadata), provenance=provenance) as writer:
            for index, (source, old, new) in enumerate(copies):
                digest = hashlib.sha256()
                def chunks():
                    for chunk in source.iter_object(old):
                        digest.update(chunk)
                        yield chunk
                writer.write_object(new, chunks())
                report['objects'].append({'origin': 'base' if source is base else 'donor',
                    'source': old, 'output': new, 'bytes': source.by_id[old].bytes,
                    'sha256': digest.hexdigest()})
                if (index + 1) % 100 == 0:
                    progress(f'Copied {index + 1}/{len(copies)} objects')
        progress('Reading back every output object and checking SHA-256')
        with Artifact(output_path) as output:
            if (output.directory.components != components or output.directory.bindings != bindings
                    or list(output.directory.uses) != uses or output.directory.metadata != base.directory.metadata):
                raise ArtifactError('output directory differs from composition plan')
            for row in report['objects']:
                digest = hashlib.sha256()
                for chunk in output.iter_object(row['output']):
                    digest.update(chunk)
                if digest.hexdigest() != row['sha256']:
                    raise ArtifactError(f'output hash mismatch: {row["output"]}')
            report.update(verified=True, output_artifact_id=output.artifact_id.hex(),
                          file_bytes=output.file_bytes, elapsed_seconds=time.monotonic() - started)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    progress(f'Verified {len(copies)} objects; report: {report_path}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--donor', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    attach(args.base, args.donor, args.out, progress=lambda s: print(s, flush=True))


if __name__ == '__main__':
    main()
