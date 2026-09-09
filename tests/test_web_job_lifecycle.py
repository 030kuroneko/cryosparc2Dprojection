"""Web Job lifecycle behavior with durable local records and OS failures."""
from pathlib import Path
import uuid

import pytest

from cryosparc_2d_projection.web_execution import Dispatcher
from cryosparc_2d_projection.web_jobs import JobStore
from cryosparc_2d_projection.workflow_config import default_values


def make_store(tmp_path, **kwargs):
    return JobStore({'data_dir': str(tmp_path / 'data'),
                     'cryosparc_url': 'https://cryo.example',
                     'profiles': {'local': {'backend': 'local',
                                            'python': str(tmp_path / 'missing-python')}}}, **kwargs)


def submit(store):
    values = default_values('axis')
    values.pop('url')
    values.update(project='P1', workspace='W2', select_job='J3', volume_job='J4')
    return store.submit({'owner': 'alice', 'email': 'alice@example.org', 'token': 'private'},
                        {'workflow': 'axis', 'profile': 'local', 'values': values,
                         'request_id': str(uuid.uuid4())})['id']


@pytest.mark.parametrize('outcome', ['completed', 'failed', 'interrupted'])
def test_cleanup_failure_preserves_outcome_and_does_not_block_next_job(tmp_path, monkeypatch, outcome):
    store = make_store(tmp_path)
    first, second = submit(store), submit(store)
    auth = store.directory(first) / 'config/cryosparc-tools/auth.json'
    unlink = Path.unlink

    def fail_cleanup(path, **kwargs):
        if path == auth:
            raise PermissionError('private path and token must not reach the user')
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_cleanup)
    store.update(first, outcome)
    result = store.get('alice', first)
    assert result['state'] == outcome
    assert result['cleanup_pending'] is True
    assert result['detail'] == ''
    assert store.get('bob', first) is None
    Dispatcher(store).tick()
    # The configured executable is deliberately absent: reaching failure proves dispatch.
    assert store.get('alice', second)['state'] == 'failed'
    assert auth.exists()


def test_cleanup_backoff_survives_restart_and_clears_warning_after_success(tmp_path, monkeypatch):
    now = [1000.0]
    store = make_store(tmp_path, clock=lambda: now[0])
    job_id = submit(store)
    auth = store.directory(job_id) / 'config/cryosparc-tools/auth.json'
    unlink = Path.unlink
    attempts = []
    blocked = [True]

    def remove(path, **kwargs):
        if path == auth:
            attempts.append(now[0])
            if blocked[0]:
                raise PermissionError('not yet')
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, 'unlink', remove)
    store.update(job_id, 'completed')
    assert attempts == [1000.0]
    for delay in [5, 10, 20, 40, 80, 160, 300, 300]:
        # Reconstruct both objects each time: retries cannot rely on process memory.
        store = make_store(tmp_path, clock=lambda: now[0])
        dispatcher = Dispatcher(store)
        count = len(attempts)
        store.update(job_id, 'completed')  # duplicate terminal evidence must not reset backoff
        assert len(attempts) == count
        now[0] += delay - 1
        dispatcher.tick()
        assert len(attempts) == count
        now[0] += 1
        dispatcher.tick()
        assert len(attempts) == count + 1
        assert store.get('alice', job_id)['cleanup_pending'] is True
        assert store.get('alice', job_id)['state'] == 'completed'
    blocked[0] = False
    now[0] += 300
    Dispatcher(make_store(tmp_path, clock=lambda: now[0])).tick()
    assert not auth.exists()
    assert store.get('alice', job_id)['cleanup_pending'] is False
    count = len(attempts)
    now[0] += 300
    Dispatcher(store).tick()
    assert len(attempts) == count


def test_worker_cleanup_failure_still_records_success(tmp_path):
    import json
    import subprocess
    import sys

    store = make_store(tmp_path)
    job_id = submit(store)
    directory = store.directory(job_id)
    # --help exercises a successful CLI exit without a CryoSPARC connection.
    (directory / 'request.json').write_text(json.dumps(
        {'workflow': 'axis', 'argv': ['--help'], 'email': 'alice@example.org'}))
    result = subprocess.run([sys.executable, '-c', '''
from pathlib import Path
import sys
from cryosparc_2d_projection.web_worker import main
unlink = Path.unlink
def remove(path, **kwargs):
    if path.name == 'auth.json':
        raise PermissionError('sensitive error text')
    return unlink(path, **kwargs)
Path.unlink = remove
sys.exit(main([sys.argv[1]]))
''', str(directory)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads((directory / 'result.json').read_text())['exit_code'] == 0
    assert 'sensitive error text' not in (directory / 'output.log').read_text()
    assert (directory / 'config/cryosparc-tools/auth.json').exists()
    store.update(job_id, 'running')
    Dispatcher(store).tick()
    assert store.get('alice', job_id)['state'] == 'completed'
    assert not (directory / 'config/cryosparc-tools/auth.json').exists()


def test_late_execution_update_cannot_reopen_a_completed_job(tmp_path):
    store = make_store(tmp_path)
    job_id = submit(store)
    store.update(job_id, 'completed', 'Original outcome')
    store.update(job_id, 'unknown', 'Stale scheduler observation')
    job = store.get('alice', job_id)
    assert job['state'] == 'completed'
    assert job['detail'] == 'Original outcome'
    assert job['cleanup_pending'] is False


def test_restart_waits_for_late_worker_completion_without_resubmitting(tmp_path):
    from cryosparc_2d_projection.web_jobs import record_worker_completion
    store = make_store(tmp_path)
    first, second = submit(store), submit(store)
    store.update(first, 'running')
    restarted = make_store(tmp_path)
    Dispatcher(restarted).tick()
    assert store.get('alice', first)['state'] == 'unknown'
    assert store.get('alice', second)['state'] == 'queued'
    record_worker_completion(store.directory(first), 0, '')
    Dispatcher(make_store(tmp_path)).tick()
    assert store.get('alice', first)['state'] == 'completed'
    assert store.get('alice', second)['state'] == 'failed'


def test_scheduler_success_without_worker_evidence_is_failure(tmp_path, monkeypatch):
    import os
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    for name, reply in [('sbatch', '471'), ('squeue', ''), ('sacct', '471|COMPLETED|0:0')]:
        binary = binaries / name
        binary.write_text('#!/bin/sh\nprintf "%s\\n" "' + reply + '"\n')
        binary.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    config = make_store(tmp_path).config
    store = JobStore(dict(config, profiles={'local': {'backend': 'slurm'}}))
    job_id = submit(store)
    dispatcher = Dispatcher(store)
    dispatcher.tick()
    assert store.get('alice', job_id)['state'] == 'pending'
    dispatcher.tick()
    job = store.get('alice', job_id)
    assert job['state'] == 'failed'
    assert 'without a workflow completion record' in job['detail']
    assert job['cleanup_pending'] is False


def test_completion_arriving_during_scheduler_check_remains_success(tmp_path, monkeypatch):
    import os
    import sys
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    config = make_store(tmp_path).config
    store = JobStore(dict(config, profiles={'local': {'backend': 'slurm'}}))
    job_id = submit(store)
    directory = store.directory(job_id)
    replies = {'sbatch': "print('471')", 'squeue': "print('')", 'sacct':
               f"from pathlib import Path\nPath({str(directory / 'result.json')!r}).write_text('{{\"exit_code\": 0}}')\nprint('471|COMPLETED|0:0')"}
    for name, body in replies.items():
        binary = binaries / name
        binary.write_text(f'#!{sys.executable}\n' + body + '\n')
        binary.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    dispatcher = Dispatcher(store)
    dispatcher.tick()
    dispatcher.tick()
    assert store.get('alice', job_id)['state'] == 'completed'


def test_upgrade_retries_cleanup_for_existing_terminal_jobs(tmp_path, monkeypatch):
    import sqlite3
    root = tmp_path / 'data'
    root.mkdir(mode=0o700)
    auth = root / 'legacy/config/cryosparc-tools/auth.json'
    auth.parent.mkdir(parents=True)
    auth.write_text('legacy credentials')
    # Fixture for the shipped schema, before cleanup tracking existed.
    with sqlite3.connect(root / 'jobs.sqlite3') as db:
        db.execute('''CREATE TABLE jobs (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_id TEXT NOT NULL,
            workflow TEXT NOT NULL, profile TEXT NOT NULL, values_json TEXT NOT NULL,
            state TEXT NOT NULL, created TEXT NOT NULL, scheduler_id TEXT,
            detail TEXT NOT NULL DEFAULT '', UNIQUE(owner, request_id))''')
        db.execute("INSERT INTO jobs VALUES ('legacy','alice','old','axis','local','{}',"
                   "'completed','2026-09-09',NULL,'')")
    unlink = Path.unlink
    def remove(path, **kwargs):
        if path == auth:
            raise PermissionError('temporarily unavailable')
        return unlink(path, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', remove)
        store = make_store(tmp_path, clock=lambda: 1000)
        Dispatcher(store).tick()
        assert store.get('alice', 'legacy')['state'] == 'completed'
        assert store.get('alice', 'legacy')['cleanup_pending'] is True
    restarted = make_store(tmp_path, clock=lambda: 1005)
    Dispatcher(restarted).tick()
    assert restarted.get('alice', 'legacy')['cleanup_pending'] is False
    assert not auth.exists()
