from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from cryosparc.dataset import Dataset
from cryosparc.models.job_spec import Input, Connection

from cryosparc_2d_projection.class_selection_export import export_class_selection


def setup_export():
    particles = Dataset({'uid': np.array([10, 11, 12], dtype='u8'),
                         'alignments2D/class': np.array([4, 1, 4], dtype='u4'),
                         'blob/idx': np.array([7, 8, 9], dtype='u4')})
    templates = Dataset({'uid': np.array([20, 21], dtype='u8'),
                         'blob/idx': np.array([1, 4], dtype='u4')})
    project = MagicMock()
    project.uid = 'P1'
    project.find_job.side_effect = lambda uid: SimpleNamespace(load_output=lambda output, **kw: particles if uid == 'J1' else templates)
    job = project.create_external_job.return_value
    job.uid = 'J9'
    job.status = 'building'
    job.model.spec.inputs.root = {}
    job.model.spec.outputs.root = {}
    manifest = dict(schema_version=1, project_uid='P1', workspace_uid='W1', source_job_uid='J3',
                    particles_source={'job_uid': 'J1', 'output': 'particles'},
                    templates_source={'job_uid': 'J2', 'output': 'templates'},
                    classes=[{'class_number': 2}, {'class_number': 5}])
    return project, job, manifest, particles, templates


def test_export_partitions_original_rows_preserving_all_fields_and_uids():
    project, job, manifest, particles, templates = setup_export()
    created = []
    job.add_input.side_effect = lambda **kw: created == ['J9'] or (_ for _ in ()).throw(AssertionError('ID not persisted'))
    result = export_class_selection(project, 'W1', manifest, [5], on_created=created.append)
    assert result == {'job_uid': 'J9', 'state': 'completed'}
    outputs = {call.args[0]: call.args[1] for call in job.save_output.call_args_list}
    assert {name: list(data['uid']) for name, data in outputs.items()} == {
        'particles_selected': [10, 12], 'particles_excluded': [11],
        'templates_selected': [21], 'templates_excluded': [20]}
    assert outputs['particles_selected'].fields() == particles.fields()
    assert list(outputs['particles_selected']['blob/idx']) == [7, 9]
    job.stop.assert_called_once_with()


def test_all_selected_keeps_empty_excluded_schema():
    project, job, manifest, particles, templates = setup_export()
    export_class_selection(project, 'W1', manifest, [2, 5])
    outputs = {call.args[0]: call.args[1] for call in job.save_output.call_args_list}
    for kind, source in [('particles', particles), ('templates', templates)]:
        assert len(outputs[f'{kind}_excluded']) == 0
        assert outputs[f'{kind}_excluded'].descr() == source.descr()


def test_invalid_membership_is_rejected_before_creating_job():
    import pytest
    project, job, manifest, particles, templates = setup_export()
    particles['alignments2D/class'][0] = 99
    with pytest.raises(ValueError, match='class'):
        export_class_selection(project, 'W1', manifest, [5])
    project.create_external_job.assert_not_called()


def test_completed_retry_returns_same_job_without_writes():
    project, job, manifest, _, _ = setup_export()
    project.find_external_job.return_value = job
    job.status = 'completed'
    assert export_class_selection(project, 'W1', manifest, [5], job_uid='J9')['job_uid'] == 'J9'
    project.create_external_job.assert_not_called()
    job.save_output.assert_not_called()
    assert [call.args[0] for call in job.load_output.call_args_list] == ['particles_selected', 'particles_excluded', 'templates_selected', 'templates_excluded']


def test_partial_retry_reuses_registered_outputs():
    project, job, manifest, _, _ = setup_export()
    project.find_external_job.return_value = job
    job.status = 'failed'
    job.model.spec.inputs.root = {kind: Input(type='particle' if kind == 'particles' else 'template', title=kind, count_max=1, connections=[Connection(**manifest[f'{kind}_source'])]) for kind in ('particles', 'templates')}
    job.connect.side_effect = AssertionError('Repeated connection exceeds max=1')
    job.model.spec.outputs.root = {name: object() for name in ('particles_selected', 'particles_excluded', 'templates_selected', 'templates_excluded')}
    export_class_selection(project, 'W1', manifest, [5], job_uid='J9')
    project.create_external_job.assert_not_called()
    job.add_output.assert_not_called()
    job.add_input.assert_not_called()
    assert job.save_output.call_count == 4


def test_failed_upload_marks_same_job_failed_and_propagates_error():
    import pytest
    project, job, manifest, _, _ = setup_export()
    job.save_output.side_effect = TimeoutError('upload interrupted')
    with pytest.raises(TimeoutError, match='upload interrupted'):
        export_class_selection(project, 'W1', manifest, [5])
    assert 'upload interrupted' in job.stop.call_args.kwargs['error']


def test_changed_particle_membership_digest_rejects_export():
    import pytest
    project, job, manifest, _, _ = setup_export()
    manifest['particles_membership_digest'] = 'stale'
    with pytest.raises(ValueError, match='membership'):
        export_class_selection(project, 'W1', manifest, [5])
    project.create_external_job.assert_not_called()


def test_completed_job_with_missing_output_is_not_reported_successful():
    import pytest
    project, job, manifest, _, _ = setup_export()
    project.find_external_job.return_value = job
    job.status = 'completed'
    job.load_output.side_effect = TypeError('Missing output')
    with pytest.raises(TypeError, match='Missing output'):
        export_class_selection(project, 'W1', manifest, [5], job_uid='J9')
    project.create_external_job.assert_not_called()
    job.save_output.assert_not_called()


def test_preflight_failure_does_not_signal_creation_but_create_failure_does():
    import pytest
    project, job, manifest, _, _ = setup_export()
    creating = MagicMock()
    manifest['particles_membership_digest'] = 'stale'
    with pytest.raises(ValueError):
        export_class_selection(project, 'W1', manifest, [5], on_creating=creating)
    creating.assert_not_called()
    del manifest['particles_membership_digest']
    project.create_external_job.side_effect = TimeoutError('creation uncertain')
    with pytest.raises(TimeoutError):
        export_class_selection(project, 'W1', manifest, [5], on_creating=creating)
    creating.assert_called_once_with()
