"""Publish original Class Selection datasets through the CryoSPARC SDK."""

from contextlib import suppress
import hashlib
import json

import numpy as np


def export_class_selection(project, workspace_uid, manifest, selected_class_numbers,
                           *, job_uid=None, on_created=None, on_creating=None):
    """Partition source rows, preserving UIDs, fields and source image references.

    The caller serializes attempts, records the creation boundary with
    ``on_creating``, and persists the job UID in ``on_created``.
    A failed request after creation must be retried with that same UID.
    """
    if manifest.get('schema_version') != 1 or manifest['project_uid'] != project.uid or manifest['workspace_uid'] != workspace_uid:
        raise ValueError('Selection source project or workspace does not match')
    allowed = {row['class_number'] for row in manifest['classes']}
    selected = set(selected_class_numbers)
    if not selected or not selected <= allowed:
        raise ValueError('Select at least one valid source class')
    job = project.find_external_job(job_uid) if job_uid else None
    if job is not None and job.status == 'completed':
        for kind in ('particles', 'templates'):
            for suffix in ('selected', 'excluded'):
                job.load_output(f'{kind}_{suffix}', slots='all')
        return {'job_uid': job.uid, 'state': 'completed'}
    datasets = {}
    for kind in ('particles', 'templates'):
        source = manifest[f'{kind}_source']
        datasets[kind] = project.find_job(source['job_uid']).load_output(source['output'], slots='all')
    template_ids = list(datasets['templates']['blob/idx'])
    if len(set(template_ids)) != len(template_ids) or {int(n) + 1 for n in template_ids} != allowed:
        raise ValueError('Source template class membership changed')
    particle_ids = datasets['particles']['alignments2D/class']
    if not {int(n) + 1 for n in particle_ids} <= allowed:
        raise ValueError('Source particle class membership is outside input classes')
    if manifest.get('particles_membership_digest'):
        pairs = sorted([int(uid), int(class_id)] for uid, class_id in
                       zip(datasets['particles']['uid'], particle_ids, strict=True))
        digest = hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()
        if digest != manifest['particles_membership_digest']:
            raise ValueError('Source particle membership changed since completion')
    if job is None:
        if on_creating is not None:
            on_creating()
        job = project.create_external_job(workspace_uid, title='2D Class Selection')
    if not job_uid and on_created is not None:
        on_created(job.uid)
    for kind, dataset in datasets.items():
        source = manifest[f'{kind}_source']
        data_type = 'particle' if kind == 'particles' else 'template'
        slots = sorted({field.split('/')[0] for field in dataset.fields() if '/' in field})
        if kind not in job.model.spec.inputs.root:
            job.add_input(type=data_type, name=kind, min=1, max=1, slots=slots)
        existing_input = job.model.spec.inputs.root.get(kind)
        connections = existing_input.connections if existing_input is not None else []
        if connections:
            if len(connections) != 1 or (connections[0].job_uid, connections[0].output) != (source['job_uid'], source['output']):
                raise ValueError('Existing selection job has different source connections')
        else:
            job.connect(kind, source['job_uid'], source['output'])
        for suffix in ('selected', 'excluded'):
            if f'{kind}_{suffix}' not in job.model.spec.outputs.root:
                job.add_output(type=data_type, name=f'{kind}_{suffix}', slots=slots)
    try:
        job.start('running')
        for kind, dataset in datasets.items():
            field = 'alignments2D/class' if kind == 'particles' else 'blob/idx'
            mask = np.isin(dataset[field], [number - 1 for number in selected])
            for suffix, rows in (('selected', mask), ('excluded', ~mask)):
                job.save_output(f'{kind}_{suffix}', dataset.mask(rows))
    except Exception as error:
        with suppress(Exception):
            job.stop(error=str(error))
        raise
    job.stop()
    return {'job_uid': job.uid, 'state': 'completed'}
