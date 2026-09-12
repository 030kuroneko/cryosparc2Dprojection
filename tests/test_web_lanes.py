"""Administrator lane workflows at HTTP and scheduler boundaries."""
import os
from pathlib import Path
import sys
import uuid

import pytest

from cryosparc_2d_projection.web import create_app
from tests.test_web import authenticate, login, submission


@pytest.fixture
def lab(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/' + name)
    config = dict(data_dir=str(tmp_path / 'state'), cryosparc_url='https://cryo.example',
                  public_url='http://localhost', allow_http=True, admin_token='admin-key')
    app = create_app(config, authenticate=authenticate)
    admin, user = app.test_client(), app.test_client()
    headers, user_headers = login(admin), login(user, 'bob@example.org')
    admin.post('/api/admin/unlock', json={'token': 'admin-key'}, headers=headers)
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o700)
    settings = dict(label='CPU lane', backend='slurm', work_dir=str(shared), python=sys.executable,
                    partition='cpu', account='', qos='', cpus=4, gpus=0, memory_mb=4096,
                    time_minutes=60, max_concurrent=1, enabled=True, shared_confirmed=True)
    return app, config, admin, headers, user, user_headers, settings


def apply_lane(admin, headers, lane_id, settings, *, reload_template=False):
    preview = admin.post('/api/admin/lanes/preview', json=dict(
        id=lane_id, settings=settings, reload_template=reload_template), headers=headers)
    assert preview.status_code == 200, preview.text
    response = admin.post('/api/admin/lanes', json={'preview_token': preview.json['preview_token']}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json['lane']


def test_admin_publishes_multiple_lanes_for_users_and_persists_changes(lab):
    app, config, admin, headers, user, user_headers, settings = lab
    body = dict(id='cpu', settings=settings, reload_template=False)
    assert user.post('/api/admin/lanes/preview', json=body, headers=user_headers).status_code == 403
    apply_lane(admin, headers, 'cpu', settings)
    gpu = apply_lane(admin, headers, 'gpu', dict(settings, label='GPU lane', partition='gpu', gpus=1))
    profiles = user.get('/api/schema').json['profiles']
    assert {p['id'] for p in profiles} == {'local', 'cpu', 'gpu'}
    assert str(settings['work_dir']) not in user.get('/api/schema').text
    job = user.post('/api/jobs', json=dict(submission(), profile='gpu'), headers=user_headers)
    assert job.status_code == 201
    apply_lane(admin, headers, 'gpu', dict(gpu, enabled=False, shared_confirmed=True))
    assert 'gpu' not in {p['id'] for p in user.get('/api/schema').json['profiles']}
    assert user.post('/api/jobs', json=dict(submission(), profile='gpu', request_id=str(uuid.uuid4())), headers=user_headers).status_code == 400
    assert user.get('/api/jobs/' + job.json['id']).json['state'] == 'queued'
    restarted = create_app(config, authenticate=authenticate).test_client()
    login(restarted)
    assert {p['id'] for p in restarted.get('/api/schema').json['profiles']} == {'local', 'cpu'}
    assert restarted.get('/api/admin/lanes').status_code == 403


def test_local_and_slurm_lanes_dispatch_independently_with_live_limits(lab, tmp_path, monkeypatch):
    import json
    from cryosparc_2d_projection.web_execution import Dispatcher
    app, config, admin, headers, user, user_headers, settings = lab
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    scheduler = '#!' + sys.executable + '\n' + '''import pathlib, sys
if pathlib.Path(sys.argv[0]).name == 'sbatch':
    path = pathlib.Path(sys.argv[-1])
    (path.parent / 'submitted').write_text('yes')
    print('471')
elif pathlib.Path(sys.argv[0]).name == 'squeue':
    print('PENDING')
'''
    for name in ('sbatch', 'squeue', 'sacct'):
        path = binaries / name
        path.write_text(scheduler)
        path.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    worker = tmp_path / 'worker'
    worker.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(60)\n')
    worker.chmod(0o700)
    # Local executable is administrator-installed configuration, not a user command.
    config['profiles'] = {'local': dict(backend='local', label='Local', python=str(worker))}
    config['data_dir'] = str(tmp_path / 'parallel-state')
    app = create_app(config, authenticate=authenticate)
    admin, user = app.test_client(), app.test_client()
    headers, user_headers = login(admin), login(user, 'bob@example.org')
    admin.post('/api/admin/unlock', json={'token': 'admin-key'}, headers=headers)
    local = admin.get('/api/admin/lanes').json['lanes'][0]
    apply_lane(admin, headers, 'local', dict(local, max_concurrent=2))
    cpu = apply_lane(admin, headers, 'cpu', settings)
    apply_lane(admin, headers, 'gpu', dict(settings, label='GPU', gpus=1))
    jobs = []
    for lane in ('local', 'local', 'local', 'cpu', 'cpu', 'gpu'):
        response = user.post('/api/jobs', json=dict(submission(), profile=lane, request_id=str(uuid.uuid4())), headers=user_headers)
        assert response.status_code == 201
        jobs.append(response.json['id'])
    from cryosparc_2d_projection.web_jobs import JobStore
    store = JobStore(config)
    dispatcher = Dispatcher(store)
    try:
        dispatcher.tick()
        states = [user.get('/api/jobs/' + job).json['state'] for job in jobs]
        assert states == ['running', 'running', 'queued', 'pending', 'queued', 'pending']
        # A lower limit does not kill or replace existing Local processes.
        local = next(p for p in admin.get('/api/admin/lanes').json['lanes'] if p['id'] == 'local')
        apply_lane(admin, headers, 'local', dict(local, max_concurrent=1))
        # Disable keeps accepted CPU work, but new submissions are blocked.
        apply_lane(admin, headers, 'cpu', dict(cpu, enabled=False, shared_confirmed=True))
        dispatcher.tick()
        assert [user.get('/api/jobs/' + job).json['state'] for job in jobs] == states
        for job in (jobs[0], jobs[3]):
            directory = store.directory(job)
            (directory / 'result.json').write_text(json.dumps({'exit_code': 0}))
        dispatcher.processes[jobs[0]].terminate()
        dispatcher.tick()
        assert user.get('/api/jobs/' + jobs[2]).json['state'] == 'queued'
        assert user.get('/api/jobs/' + jobs[4]).json['state'] == 'pending'
    finally:
        for process in dispatcher.processes.values():
            process.terminate()
            process.wait(timeout=5)


def test_external_template_is_previewed_activated_and_snapshotted_per_job(lab, tmp_path):
    import subprocess
    from cryosparc_2d_projection.web_execution import SlurmBackend
    from cryosparc_2d_projection.web_jobs import JobStore
    app, config, admin, headers, user, user_headers, settings = lab
    template = tmp_path / 'cluster.sh'
    template.write_text('''#!/bin/bash
#SBATCH --cpus-per-task=999 --mem=1M
#SBATCH --partition=wrong
module load {{ module_name | quote }}
cd {{ job_dir | quote }}
exec {{ run_cmd }}
''')
    settings.update(template_path=str(template), variables={'module_name': 'cuda/12'})
    lane = apply_lane(admin, headers, 'gpu', settings, reload_template=True)
    assert lane['template_sha256']
    template.write_text('#!/bin/bash\nmodule load cuda/13\nexec {{ run_cmd }}\n')
    # File edits alone do not affect new jobs using the currently activated version.
    first = user.post('/api/jobs', json=dict(submission(), profile='gpu'), headers=user_headers).json
    lane = apply_lane(admin, headers, 'gpu', dict(lane, shared_confirmed=True), reload_template=True)
    second = user.post('/api/jobs', json=dict(submission(), profile='gpu', request_id=str(uuid.uuid4())), headers=user_headers).json
    template.unlink()
    # Persisted jobs and lane versions work after restart without re-reading the file.
    store = JobStore(config)
    for job, expected_module in ((first, 'cuda/12'), (second, 'cuda/13')):
        claimed = store.claim_next()
        assert claimed['id'] == job['id']
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, '471\n', '')
        SlurmBackend(claimed['profile'], run=run).submit(claimed['directory'])
        script = (claimed['directory'] / 'submit.sh').read_text()
        assert 'module load ' + expected_module in script
        assert '{{' not in script
        assert '999' not in script and '--partition=wrong' not in script
        assert '--cpus-per-task=4' in calls[0] and '--mem=4096M' in calls[0]
        assert 'cryosparc_2d_projection.web_worker' in script
        assert 'secret-token' not in script
        store.update(claimed['id'], 'completed')


def test_template_with_only_a_commented_worker_command_cannot_be_activated(lab, tmp_path):
    app, config, admin, headers, user, user_headers, settings = lab
    template = tmp_path / 'invalid.sh'
    template.write_text('#!/bin/bash\n# {{ run_cmd }}\necho done\n')
    response = admin.post('/api/admin/lanes/preview', json=dict(id='bad',
        settings=dict(settings, template_path=str(template)), reload_template=True), headers=headers)
    assert response.status_code == 400
    assert 'run_cmd' in response.json['error']
    assert 'bad' not in {p['id'] for p in user.get('/api/schema').json['profiles']}


def test_duplicate_uses_the_registered_template_even_if_external_file_is_gone(lab, tmp_path):
    app, config, admin, headers, user, user_headers, settings = lab
    template = tmp_path / 'cluster.sh'
    template.write_text('#!/bin/bash\nmodule load special\nexec {{ run_cmd }}\n')
    original = apply_lane(admin, headers, 'original', dict(settings, template_path=str(template)), reload_template=True)
    template.unlink()
    copy_settings = dict(original, label='Copy', copy_from='original', shared_confirmed=True)
    copy_settings.pop('id')
    copy_settings.pop('revision')
    copied = apply_lane(admin, headers, 'copy', copy_settings)
    assert copied['template_sha256'] == original['template_sha256']
    assert copied['id'] == 'copy'
    assert {p['id'] for p in user.get('/api/schema').json['profiles']} == {'local', 'original', 'copy'}


def test_local_capacity_edit_accepts_an_unlabelled_existing_profile(tmp_path):
    config = dict(data_dir=str(tmp_path), cryosparc_url='https://cryo.example', public_url='http://localhost',
                  allow_http=True, admin_token='admin-key', profiles={'local': {'backend': 'local'}})
    admin = create_app(config, authenticate=authenticate).test_client()
    headers = login(admin)
    admin.post('/api/admin/unlock', json={'token': 'admin-key'}, headers=headers)
    local = admin.get('/api/admin/lanes').json['lanes'][0]
    lane = apply_lane(admin, headers, 'local', dict(local, max_concurrent=3))
    assert lane['max_concurrent'] == 3
    assert lane['label'] == 'Local'


@pytest.mark.parametrize('changes,error', [
    ({'max_concurrent': 0}, 'concurrency'),
    ({'max_concurrent': True}, 'concurrency'),
    ({'variables': {'run_cmd': 'something else'}}, 'reserved'),
    ({'variables': {'module': 'one\ntwo'}}, 'single-line'),
])
def test_invalid_lane_values_cannot_be_published(lab, changes, error):
    app, config, admin, headers, user, user_headers, settings = lab
    result = admin.post('/api/admin/lanes/preview', json=dict(id='bad',
        settings=dict(settings, **changes), reload_template=False), headers=headers)
    assert result.status_code == 400
    assert error in result.json['error']


def test_stale_preview_cannot_overwrite_a_later_admin_change(lab):
    app, config, admin, headers, user, user_headers, settings = lab
    lane = apply_lane(admin, headers, 'cpu', settings)
    second_admin = app.test_client()
    second_headers = login(second_admin, 'carol@example.org')
    second_admin.post('/api/admin/unlock', json={'token': 'admin-key'}, headers=second_headers)
    old_preview = admin.post('/api/admin/lanes/preview', json=dict(id='cpu',
        settings=dict(lane, cpus=5, shared_confirmed=True), reload_template=False), headers=headers).json
    apply_lane(second_admin, second_headers, 'cpu', dict(lane, cpus=8, shared_confirmed=True))
    response = admin.post('/api/admin/lanes', json={'preview_token': old_preview['preview_token']}, headers=headers)
    assert response.status_code == 409
    current = next(l for l in admin.get('/api/admin/lanes').json['lanes'] if l['id'] == 'cpu')
    assert current['cpus'] == 8
