import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from cryosparc_2d_projection.class_selection_jobs import (
    ClassSelectionExportLifecycle,
    ExportRequestConflict,
)
from cryosparc_2d_projection.web_jobs import JobStore


def setup_lifecycle(tmp_path, *, launch=None):
    from test_class_selection_export import setup_export

    store = JobStore({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example'})
    with store.connect() as db:
        db.execute(
            "INSERT INTO jobs (id,owner,request_id,workflow,profile,values_json,state,created) "
            "VALUES ('one','alice','r','orientation','local','{}','completed','now')"
        )
    project, job, source_manifest, _, _ = setup_export()
    manifest = dict(
        schema_version=1,
        project_uid='P1',
        workspace_uid='W1',
        source_job_uid='J3',
        particles_source=source_manifest['particles_source'],
        templates_source=source_manifest['templates_source'],
        classes=[
            dict(class_number=2, particle_count=1, score=.5,
                 orientation_method='image', confidence='low', image='class_2.png'),
            dict(class_number=5, particle_count=2, score=.4,
                 orientation_method='image', confidence='low', image='class_5.png'),
        ],
    )
    lifecycle = ClassSelectionExportLifecycle(
        store,
        project_factory=lambda identity, project_uid: project,
        launch=launch,
    )
    return lifecycle, store, project, job, manifest


def wait_for_state(lifecycle, identity, state):
    for _ in range(200):
        exports = lifecycle.list_exports(identity, 'one')
        if exports and exports[0]['state'] == state:
            return exports[0]
        time.sleep(.01)
    raise AssertionError(exports)


def test_start_publishes_one_snapshot_through_lifecycle(tmp_path):
    lifecycle, _, project, _, manifest = setup_lifecycle(tmp_path)
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    request_id = str(uuid.uuid4())

    accepted = lifecycle.start(identity, 'one', request_id, [5], manifest)

    assert accepted['state'] == 'preparing'
    completed = wait_for_state(lifecycle, identity, 'completed')
    assert completed['selected_class_numbers'] == [5]
    assert completed['job_uid'] == 'J9'
    assert project.create_external_job.call_count == 1


def test_duplicate_request_keeps_its_first_snapshot_and_selection(tmp_path):
    entered, release = Event(), Event()
    lifecycle, _, project, _, manifest = setup_lifecycle(tmp_path)
    lifecycle.project_factory = lambda identity, project_uid: (
        entered.set() or (release.wait(3) and project)
    )
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    request_id = str(uuid.uuid4())
    lifecycle.start(identity, 'one', request_id, [5], manifest)
    assert entered.wait(3)

    changed = json.loads(json.dumps(manifest))
    changed['classes'] = [changed['classes'][0]]
    duplicate = lifecycle.start(identity, 'one', request_id, [5], changed)

    assert duplicate['selected_class_numbers'] == [5]
    with pytest.raises(ExportRequestConflict):
        lifecycle.start(identity, 'one', request_id, [2], changed)
    release.set()
    assert wait_for_state(lifecycle, identity, 'completed')['selected_class_numbers'] == [5]


class SimulatedCrash(BaseException):
    pass


def test_restart_recovers_preparing_after_worker_crash(tmp_path):
    crashed = Event()

    def launch(target, *args):
        try:
            target(*args)
        except SimulatedCrash:
            pass

    lifecycle, store, project, _, manifest = setup_lifecycle(tmp_path, launch=launch)

    def project_factory(identity, project_uid):
        crashed.set()
        raise SimulatedCrash()

    lifecycle.project_factory = project_factory
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    lifecycle.start(identity, 'one', str(uuid.uuid4()), [5], manifest)
    assert crashed.wait(3)

    restarted = ClassSelectionExportLifecycle(
        store, project_factory=lambda identity, project_uid: project)
    recovered = restarted.list_exports(identity, 'one')[0]
    assert recovered['state'] == 'failed'
    assert 'retry' in recovered['detail']
    restarted.retry(identity, 'one', recovered['id'])
    assert wait_for_state(restarted, identity, 'completed')['job_uid'] == 'J9'
    assert project.create_external_job.call_count == 1


def test_restart_blocks_creation_with_unknown_remote_job(tmp_path):
    entered = Event()

    def launch(target, *args):
        try:
            target(*args)
        except SimulatedCrash:
            pass

    lifecycle, store, project, _, manifest = setup_lifecycle(tmp_path, launch=launch)

    def create(*args, **kwargs):
        entered.set()
        raise SimulatedCrash()

    project.create_external_job.side_effect = create
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    lifecycle.start(identity, 'one', str(uuid.uuid4()), [5], manifest)
    assert entered.wait(3)
    restarted = ClassSelectionExportLifecycle(
        store, project_factory=lambda identity, project_uid: project)
    recovered = restarted.list_exports(identity, 'one')[0]
    assert recovered['state'] == 'unknown'
    assert recovered['job_uid'] is None

    from cryosparc_2d_projection.class_selection_jobs import ExportUnknownCreation
    with pytest.raises(ExportUnknownCreation):
        restarted.retry(identity, 'one', recovered['id'])
    with pytest.raises(ExportUnknownCreation):
        restarted.start(identity, 'one', str(uuid.uuid4()), [2], manifest)
    assert project.create_external_job.call_count == 1


def test_restart_reconciles_publishing_with_recorded_uid(tmp_path):
    started = Event()

    def launch(target, *args):
        try:
            target(*args)
        except SimulatedCrash:
            pass

    lifecycle, store, project, job, manifest = setup_lifecycle(tmp_path, launch=launch)
    job.start.side_effect = lambda *args: (started.set() or (_ for _ in ()).throw(SimulatedCrash()))
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    lifecycle.start(identity, 'one', str(uuid.uuid4()), [5], manifest)
    assert started.wait(3)

    restarted = ClassSelectionExportLifecycle(
        store, project_factory=lambda identity, project_uid: project)
    recovered = restarted.list_exports(identity, 'one')[0]
    assert recovered['state'] == 'unknown'
    assert recovered['job_uid'] == 'J9'
    job.start.side_effect = None
    project.find_external_job.return_value = job
    restarted.retry(identity, 'one', recovered['id'])
    assert wait_for_state(restarted, identity, 'completed')['job_uid'] == 'J9'
    assert project.create_external_job.call_count == 1


def test_capacity_is_four_active_exports(tmp_path):
    launched = []
    lifecycle, _, _, _, manifest = setup_lifecycle(
        tmp_path, launch=lambda target, *args: launched.append((target, args)))
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    for _ in range(4):
        lifecycle.start(identity, 'one', str(uuid.uuid4()), [5], manifest)
    from cryosparc_2d_projection.class_selection_jobs import ExportCapacityReached
    with pytest.raises(ExportCapacityReached):
        lifecycle.start(identity, 'one', str(uuid.uuid4()), [5], manifest)
    assert len(launched) == 4


def test_concurrent_retry_claims_one_publication(tmp_path):
    lifecycle, _, project, job, manifest = setup_lifecycle(tmp_path)
    project.find_external_job.return_value = job
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'secret'}
    job.save_output.side_effect = RuntimeError('publication unavailable')
    lifecycle.start(identity, 'one', str(uuid.uuid4()), [5], manifest)
    failed = wait_for_state(lifecycle, identity, 'failed')
    entered, release = Event(), Event()

    def save(*args, **kwargs):
        entered.set()
        assert release.wait(3)

    job.save_output.side_effect = save
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(
            lambda _: lifecycle.retry(identity, 'one', failed['id']), range(2)))
    assert {response['id'] for response in responses} == {failed['id']}
    assert entered.wait(3)
    release.set()
    assert wait_for_state(lifecycle, identity, 'completed')['job_uid'] == 'J9'
    assert project.find_external_job.call_count == 1
    assert project.create_external_job.call_count == 1


@pytest.mark.parametrize('operation', ['list', 'start', 'retry'])
def test_exports_are_isolated_by_source_owner(tmp_path, operation):
    from cryosparc_2d_projection.class_selection_jobs import ExportNotFound

    lifecycle, _, project, _, manifest = setup_lifecycle(
        tmp_path, launch=lambda target, *args: None)
    export = lifecycle.start({'owner': 'alice'}, 'one', str(uuid.uuid4()), [5], manifest)
    with pytest.raises(ExportNotFound):
        if operation == 'list':
            lifecycle.list_exports({'owner': 'bob'}, 'one')
        elif operation == 'start':
            lifecycle.start({'owner': 'bob'}, 'one', str(uuid.uuid4()), [2], manifest)
        else:
            lifecycle.retry({'owner': 'bob'}, 'one', export['id'])
    assert len(lifecycle.list_exports({'owner': 'alice'}, 'one')) == 1
    project.create_external_job.assert_not_called()
