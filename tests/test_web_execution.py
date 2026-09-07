"""Scheduler contracts at the OS command boundary, without a real cluster."""
import subprocess
import sys
import json
import os
import pytest

from cryosparc_2d_projection.web_execution import SlurmBackend


def test_slurm_submission_returns_job_id_and_waits_for_accounting(tmp_path):
    commands = []
    replies = iter(['471;cluster\n', 'PENDING\n', '', '471|COMPLETED|0:0\n'])

    def run(argv, **kwargs):
        commands.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, next(replies), '')

    backend = SlurmBackend({'python': sys.executable, 'partition': 'cpu', 'cpus': 4,
                            'memory_mb': 8000, 'time_minutes': 30}, run=run)
    job_id = backend.submit(tmp_path)
    assert job_id == '471'
    assert backend.status(job_id) == ('pending', '')
    assert backend.status(job_id) == ('completed', '')
    argv = commands[0][0]
    assert '--parsable' in argv and '--cpus-per-task=4' in argv
    assert '--partition=cpu' in argv
    assert all(not kwargs.get('shell') for _, kwargs in commands)
    assert 'token' not in (tmp_path / 'submit.sh').read_text()


def test_slurm_missing_accounting_is_not_success():
    backend = SlurmBackend({}, run=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, '', ''))
    assert backend.status('471') == ('unknown', 'Waiting for Slurm accounting; not resubmitting.')


def test_job_removed_from_squeue_is_reconciled_with_sacct():
    def run(argv, **kwargs):
        if argv[0] == 'squeue':
            raise subprocess.CalledProcessError(1, argv, stderr='Invalid job id specified')
        return subprocess.CompletedProcess(argv, 0, '471|TIMEOUT|0:15\n', '')
    assert SlurmBackend({}, run=run).status('471') == ('failed', 'Slurm TIMEOUT; exit 0:15')


def test_worker_records_invalid_workflow_input_without_live_connection(tmp_path):
    auth_dir = tmp_path / 'config' / 'cryosparc-tools'
    auth_dir.mkdir(parents=True)
    (auth_dir / 'auth.json').write_text(json.dumps({'https://cryo.example': {'alice': {
        'token': {'access_token': 'a-private-test-token'}}}}))
    (tmp_path / 'request.json').write_text(json.dumps({'workflow': 'orientation', 'argv': [], 'email': 'alice'}))
    result = subprocess.run([sys.executable, '-m', 'cryosparc_2d_projection.web_worker', str(tmp_path)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert json.loads((tmp_path / 'result.json').read_text())['exit_code'] != 0
    assert 'required' in (tmp_path / 'output.log').read_text()
    assert 'a-private-test-token' not in (tmp_path / 'output.log').read_text()
    assert not (auth_dir / 'auth.json').exists()


@pytest.mark.parametrize('state,exit_code', [('TIMEOUT', '0:0'), ('OUT_OF_MEMORY', '0:125'),
                                           ('CANCELLED by 7', '0:15'), ('COMPLETED', '2:0')])
def test_scheduler_failure_cannot_be_reported_as_success(state, exit_code):
    def run(argv, **kwargs):
        text = '' if argv[0] == 'squeue' else f'471|{state}|{exit_code}\n'
        return subprocess.CompletedProcess(argv, 0, text, '')
    assert SlurmBackend({}, run=run).status('471')[0] == 'failed'


def test_uncertain_submission_blocks_further_dispatch_instead_of_resubmitting(tmp_path, monkeypatch):
    from cryosparc_2d_projection.web_jobs import JobStore
    from cryosparc_2d_projection.web_execution import Dispatcher
    from cryosparc_2d_projection.gui_model import default_values
    import uuid
    binary = tmp_path / 'sbatch'
    binary.write_text('#!/bin/sh\nprintf "unexpected scheduler response\\n"\n')
    binary.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    store = JobStore({'data_dir': str(tmp_path / 'data'), 'cryosparc_url': 'https://cryo.example',
                      'profiles': {'cluster': {'backend': 'slurm'}}})
    values = default_values('axis')
    values.pop('url')
    values.update(project='P1', workspace='W2', select_job='J3', volume_job='J4')
    identity = {'owner': 'alice', 'email': 'alice@example.org', 'token': 'private'}
    first, second = [store.submit(identity, {'workflow': 'axis', 'profile': 'cluster', 'values': values,
                                           'request_id': str(uuid.uuid4())}) for _ in range(2)]
    dispatcher = Dispatcher(store)
    dispatcher.tick()
    dispatcher.tick()
    assert store.get('alice', first['id'])['state'] == 'unknown'
    assert store.get('alice', second['id'])['state'] == 'queued'


def test_log_redacts_token_split_across_writes_and_bounds_output():
    import io
    from cryosparc_2d_projection.web_worker import JobLog
    out = io.StringIO()
    log = JobLog(out, ['private-token'])
    log.write('Token: private-')
    log.write('token\n')
    log.finish()
    assert out.getvalue() == 'Token: [REDACTED]\n'
    log.write('x' * 100000)
    assert len(log.pending) < 65536
