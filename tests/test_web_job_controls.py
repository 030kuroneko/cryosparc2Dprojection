"""Owner-visible stop and deletion contracts."""
from tests.test_web import app, login, submission


def test_queued_stop_and_delete_preserve_results_and_owner_boundary(app):
    alice, bob = app.test_client(), app.test_client()
    headers, other = login(alice), login(bob, 'bob@example.org')
    job = alice.post('/api/jobs', json=submission(), headers=headers).json
    url = '/api/jobs/' + job['id']
    assert bob.post(url + '/stop', headers=other).status_code == 404
    assert alice.post(url + '/stop').status_code == 403
    stopped = alice.post(url + '/stop', headers=headers)
    assert stopped.status_code == 200
    assert stopped.json['state'] == 'interrupted'
    deleted = alice.delete(url, headers=headers)
    assert deleted.status_code == 200
    assert alice.get(url).status_code == 404
    assert alice.get('/api/jobs').json['jobs'] == []


def test_dispatcher_kills_running_local_job_and_preserves_files(tmp_path):
    import os
    import sys
    from tests.test_web_job_lifecycle import make_store, submit
    from cryosparc_2d_projection.web_execution import Dispatcher
    from cryosparc_2d_projection.web_jobs import JobStore
    executable = tmp_path / 'worker'
    executable.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(120)\n')
    executable.chmod(0o700)
    config = make_store(tmp_path).config
    config['profiles']['local']['python'] = str(executable)
    store = JobStore(config)
    job_id = submit(store)
    directory = store.directory(job_id)
    result = directory / 'preserved.txt'
    result.write_text('result')
    dispatcher = Dispatcher(store)
    dispatcher.tick()
    process = dispatcher.processes[job_id]
    try:
        store.request_control('alice', job_id, delete=True)
        dispatcher.tick()
        assert process.poll() is not None
        assert store.get('alice', job_id) is None
        assert result.read_text() == 'result'
        assert not (directory / 'config/cryosparc-tools/auth.json').exists()
    finally:
        if process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_slurm_delete_waits_for_confirmed_cancellation(tmp_path, monkeypatch):
    import os
    from tests.test_web_job_lifecycle import make_store, submit
    from cryosparc_2d_projection.web_jobs import JobStore
    from cryosparc_2d_projection.web_execution import Dispatcher
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    for name, reply in [('sbatch', '471'), ('squeue', 'RUNNING'), ('sacct', '471|CANCELLED|0:9'), ('scancel', '')]:
        path = binaries / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "' + reply + '"\n')
        path.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    config = make_store(tmp_path).config
    store = JobStore(dict(config, profiles={'local': {'backend': 'slurm'}}))
    job_id = submit(store)
    dispatcher = Dispatcher(store)
    dispatcher.tick()
    store.request_control('alice', job_id, delete=True)
    dispatcher.tick()
    assert store.get('alice', job_id)['state'] == 'stopping'
    (binaries / 'squeue').write_text('#!/bin/sh\nexit 0\n')
    dispatcher.tick()
    assert store.get('alice', job_id) is None


def test_external_stop_sync_preserves_completed_and_reports_failures(tmp_path):
    import json
    from types import SimpleNamespace
    from cryosparc_2d_projection.web_job_stop import sync_external_stop
    (tmp_path / 'external-job.json').write_text(json.dumps({'project_uid': 'P1', 'job_uid': 'J8'}))
    calls = []
    api = SimpleNamespace(jobs=SimpleNamespace(
        find_one=lambda *args: {'status': 'running'},
        mark_failed=lambda *args, **kwargs: calls.append((args, kwargs))))
    assert sync_external_stop(tmp_path, api=api) == ''
    assert calls[0][0] == ('P1', 'J8')
    assert 'Stopped by user' in calls[0][1]['error']
    calls.clear()
    api.jobs.find_one = lambda *args: {'status': 'completed'}
    assert sync_external_stop(tmp_path, api=api) == ''
    assert calls == []
    def unavailable(*args):
        raise RuntimeError('SECRET')
    api.jobs.find_one = unavailable
    warning = sync_external_stop(tmp_path, api=api)
    assert 'CryoSPARC' in warning and 'SECRET' not in warning


def test_created_external_job_can_be_synchronized_after_stop(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cryosparc_2d_projection.external_job_adapter import CryoSPARCExternalJobAdapter
    from cryosparc_2d_projection.web_job_stop import sync_external_stop
    monkeypatch.setenv('CRYOSPARC2D_EXTERNAL_JOB_PATH', str(tmp_path / 'external-job.json'))
    calls = []
    api = SimpleNamespace(jobs=SimpleNamespace(
        find_one=lambda *args: {'status': 'running'},
        mark_failed=lambda *args, **kwargs: calls.append(args)))
    def create(*args, **kwargs):
        assert 'CryoSPARC' in sync_external_stop(tmp_path, api=api)
        assert calls == []
        return SimpleNamespace(uid='J9', project_uid='P1')
    CryoSPARCExternalJobAdapter(SimpleNamespace(create_external_job=create), 'W1')
    assert sync_external_stop(tmp_path, api=api) == ''
    assert calls == [('P1', 'J9')]


def test_stop_survives_dispatcher_restart(tmp_path):
    import os
    import signal
    import sys
    from tests.test_web_job_lifecycle import make_store, submit
    from cryosparc_2d_projection.web_execution import Dispatcher
    from cryosparc_2d_projection.web_jobs import JobStore
    executable = tmp_path / 'worker'
    executable.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(120)\n')
    executable.chmod(0o700)
    config = make_store(tmp_path).config
    config['profiles']['local']['python'] = str(executable)
    store = JobStore(config)
    job_id = submit(store)
    old = Dispatcher(store)
    old.tick()
    process = old.processes[job_id]
    try:
        store.request_control('alice', job_id)
        Dispatcher(JobStore(config)).tick()
        assert store.get('alice', job_id)['state'] == 'interrupted'
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_deleted_job_keeps_sync_warning_visible_without_result_access(app, tmp_path):
    import json
    from cryosparc_2d_projection.web_jobs import JobStore
    from cryosparc_2d_projection.web_execution import Dispatcher
    alice = app.test_client()
    headers = login(alice)
    job = alice.post('/api/jobs', json=submission(), headers=headers).json
    store = JobStore(dict(data_dir=str(tmp_path), cryosparc_url='https://cryo.example'))
    directory = store.directory(job['id'])
    (directory / 'external-job.json').write_text(json.dumps({'pending': True}))
    # A durable local identity proves the stopped group no longer exists.
    (directory / 'local-process.json').write_text(json.dumps({'pid': 99999999, 'identity': 'gone'}))
    store.update(job['id'], 'running')
    alice.delete('/api/jobs/' + job['id'], headers=headers)
    Dispatcher(store).tick()
    listing = alice.get('/api/jobs').json
    assert listing['jobs'] == []
    assert 'CryoSPARC' in listing['deletion_warnings'][0]['detail']
    assert alice.get('/api/jobs/' + job['id'] + '/log').status_code == 404
    bob = app.test_client()
    login(bob, 'bob@example.org')
    assert bob.get('/api/jobs').json['deletion_warnings'] == []


def test_stop_during_submission_stays_stopping_until_scheduler_ends(tmp_path, monkeypatch):
    import os
    from tests.test_web_job_lifecycle import make_store, submit
    from cryosparc_2d_projection.web_jobs import JobStore
    from cryosparc_2d_projection.web_execution import Dispatcher
    import threading
    import time
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    started, release = tmp_path / 'started', tmp_path / 'release'
    for name, body in {
        'sbatch': f'touch "{started}"\nwhile [ ! -e "{release}" ]; do sleep 0.01; done\necho 471',
        'squeue': 'echo RUNNING', 'sacct': 'echo "471|CANCELLED|0:9"', 'scancel': 'exit 0'
    }.items():
        path = binaries / name
        path.write_text('#!/bin/sh\n' + body + '\n')
        path.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    config = make_store(tmp_path).config
    store = JobStore(dict(config, profiles={'local': {'backend': 'slurm'}}))
    job_id = submit(store)
    dispatcher = Dispatcher(store)
    thread = threading.Thread(target=dispatcher.tick)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert started.exists()
        store.request_control('alice', job_id, delete=True)
    finally:
        release.touch()
        thread.join(5)
    assert store.get('alice', job_id)['state'] == 'stopping'
    dispatcher.tick()
    assert store.get('alice', job_id) is not None
    (binaries / 'squeue').write_text('#!/bin/sh\nexit 0\n')
    dispatcher.tick()
    assert store.get('alice', job_id) is None


def test_pending_slurm_stop_cancels_allocation_not_only_steps(tmp_path, monkeypatch):
    import os
    from cryosparc_2d_projection.web_execution import SlurmBackend
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    cancelled = tmp_path / 'cancelled'
    for name, body in {
        'squeue': f'[ -e "{cancelled}" ] || echo PENDING',
        'sacct': 'echo "471|CANCELLED|0:0"',
        'scancel': f'case "$*" in *--signal*) exit 0;; *) touch "{cancelled}";; esac'
    }.items():
        executable = binaries / name
        executable.write_text('#!/bin/sh\n' + body + '\n')
        executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    assert SlurmBackend({}).stop('471') is True


def test_completed_worker_evidence_wins_after_restart_without_identity(tmp_path):
    from tests.test_web_job_lifecycle import make_store, submit
    from cryosparc_2d_projection.web_jobs import record_worker_completion
    from cryosparc_2d_projection.web_execution import Dispatcher
    store = make_store(tmp_path)
    job_id = submit(store)
    store.update(job_id, 'running')
    record_worker_completion(store.directory(job_id), 0, '')
    store.request_control('alice', job_id)
    Dispatcher(make_store(tmp_path)).tick()
    assert store.get('alice', job_id)['state'] == 'completed'


def test_unreadable_execution_evidence_retains_record_with_retry_reason(tmp_path):
    from tests.test_web_job_lifecycle import make_store, submit
    from cryosparc_2d_projection.web_execution import Dispatcher
    store = make_store(tmp_path)
    job_id = submit(store)
    store.update(job_id, 'running')
    (store.directory(job_id) / 'local-process.json').write_text('{partial')
    store.request_control('alice', job_id, delete=True)
    Dispatcher(store).tick()
    job = store.get('alice', job_id)
    assert job['state'] == 'stopping'
    assert 'retry' in job['detail'].lower()
