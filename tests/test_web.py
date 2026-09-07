"""HTTP contracts for the multi-user launcher; no live CryoSPARC required."""
import pytest
import sys
import time
from concurrent.futures import ThreadPoolExecutor
pytest.importorskip('flask', reason='Install the web extra to test the HTTP launcher')
from cryosparc_2d_projection.gui_model import default_values

from cryosparc_2d_projection.web import create_app


def authenticate(url, email, password):
    if password != 'correct':
        raise ValueError('private upstream error containing a password')
    return {'owner': email, 'email': email, 'token': 'secret-token-' + email}


@pytest.fixture
def app(tmp_path):
    return create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                       'public_url': 'http://localhost', 'allow_http': True},
                      authenticate=authenticate)


def login(client, email='alice@example.org'):
    csrf = client.get('/api/session').json['csrf']
    response = client.post('/api/login', json={'email': email, 'password': 'correct'},
                           headers={'X-CSRF-Token': csrf, 'Origin': 'http://localhost'})
    assert response.status_code == 200
    return {'X-CSRF-Token': response.json['csrf'], 'Origin': 'http://localhost'}


@pytest.mark.parametrize('plain', [True, False])
def test_sdk_login_accepts_plain_json_and_model_responses(tmp_path, monkeypatch, plain):
    from types import SimpleNamespace
    token = {'access_token': 'upstream-secret', 'token_type': 'bearer'}
    user = {'_id': 'authoritative-user-id'}
    class API:
        def __init__(self, *args, **kwargs):
            self.users = SimpleNamespace(me=lambda: user if plain else SimpleNamespace(id=user['_id']))
        def login(self, **kwargs):
            return token if plain else SimpleNamespace(**token)
        def __call__(self, auth):
            assert auth == 'upstream-secret'
    monkeypatch.setattr('cryosparc.api.APIClient', API)
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                      'public_url': 'http://localhost', 'allow_http': True})
    client = app.test_client()
    headers = login(client)
    response = client.post('/api/jobs', json=submission(), headers=headers)
    assert response.status_code == 201
    assert 'upstream-secret' not in response.text
    assert client.get('/api/session').json['email'] == 'alice@example.org'


@pytest.mark.parametrize('token,user', [
    ({'access_token': ''}, {'_id': 'user'}),
    ({'access_token': None}, {'_id': 'user'}),
    ({'access_token': 'secret'}, {}),
    ({'access_token': 'secret'}, {'_id': ''}),
    ({'access_token': 'secret'}, {'_id': 123}),
])
def test_malformed_sdk_identity_cannot_create_a_session(tmp_path, monkeypatch, token, user):
    from types import SimpleNamespace
    class API:
        def __init__(self, *args, **kwargs):
            self.users = SimpleNamespace(me=lambda: user)
        def login(self, **kwargs):
            return token
        def __call__(self, auth):
            pass
    monkeypatch.setattr('cryosparc.api.APIClient', API)
    client = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                        'public_url': 'http://localhost', 'allow_http': True}).test_client()
    csrf = client.get('/api/session').json['csrf']
    response = client.post('/api/login', json={'email': 'alice', 'password': 'correct'},
                           headers={'Origin': 'http://localhost', 'X-CSRF-Token': csrf})
    assert response.status_code == 401
    assert client.get('/api/jobs').status_code == 401


@pytest.mark.parametrize('host', ['localhost', '127.0.0.1'])
def test_loopback_alias_supports_login_and_submission_on_configured_port(tmp_path, host):
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                      'public_url': 'http://127.0.0.1:40000', 'allow_http': True},
                     authenticate=authenticate)
    client = app.test_client()
    origin = f'http://{host}:40000'
    response = client.get('/api/session', base_url=origin)
    assert response.status_code == 200
    response = client.post('/api/login', base_url=origin,
        json={'email': 'alice', 'password': 'correct'},
        headers={'Origin': origin, 'X-CSRF-Token': response.json['csrf']})
    assert response.status_code == 200
    headers = {'Origin': origin, 'X-CSRF-Token': response.json['csrf']}
    assert client.post('/api/jobs', base_url=origin, json=submission(), headers=headers).status_code == 201
    for other in ('http://localhost:40001', 'https://localhost:40000', 'http://evil.example:40000'):
        assert client.post('/api/logout', base_url=origin,
                           headers=dict(headers, Origin=other)).status_code == 403
    assert client.get('/api/session', base_url='http://evil.example:40000').status_code == 400


def test_https_does_not_trust_loopback_aliases(tmp_path):
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                      'public_url': 'https://lab.example'}, authenticate=authenticate)
    client = app.test_client()
    csrf = client.get('/api/session', base_url='https://lab.example').json['csrf']
    for origin in ('http://localhost', 'https://127.0.0.1', 'https://lab.example:40001'):
        response = client.post('/api/login', base_url='https://lab.example',
            json={'email': 'alice', 'password': 'correct'},
            headers={'Origin': origin, 'X-CSRF-Token': csrf})
        assert response.status_code == 403
    assert client.get('/api/session', base_url='https://localhost').status_code == 400


def test_login_requires_csrf_rotates_session_and_never_returns_credentials(app):
    client = app.test_client()
    assert client.get('/api/jobs').status_code == 401
    assert client.post('/api/login', json={}).status_code == 403
    before = client.get('/api/session').json['csrf']
    headers = login(client)
    assert headers['X-CSRF-Token'] != before
    session = client.get('/api/session')
    assert session.json['email'] == 'alice@example.org'
    assert 'secret-token' not in session.text
    assert client.post('/api/logout', headers=headers).status_code == 200
    assert client.get('/api/jobs').status_code == 401


def submission(**changes):
    values = default_values('orientation')
    values.update(project='P1', workspace='W2', select_job='J3', refinement_job='J4')
    values.pop('url')
    values.update(changes)
    return {'workflow': 'orientation', 'values': values, 'profile': 'local',
            'request_id': '3fe2ecb5-57a1-4d0a-8e51-d4270ba8f71d'}


def test_jobs_are_validated_idempotent_and_private_even_after_restart(app, tmp_path):
    alice, bob = app.test_client(), app.test_client()
    a, b = login(alice), login(bob, 'bob@example.org')
    assert alice.post('/api/jobs', json=submission(project='wrong'), headers=a).status_code == 400
    response = alice.post('/api/jobs', json=submission(), headers=a)
    assert response.status_code == 201
    job = response.json
    assert job['state'] == 'queued'
    assert 'token' not in response.text
    again = alice.post('/api/jobs', json=submission(), headers=a)
    assert again.json['id'] == job['id']
    assert len(alice.get('/api/jobs').json['jobs']) == 1
    assert bob.get('/api/jobs').json['jobs'] == []
    assert bob.get('/api/jobs/' + job['id']).status_code == 404
    assert bob.get('/api/jobs/' + job['id'] + '/log').status_code == 404
    restarted = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                            'public_url': 'http://localhost', 'allow_http': True}, authenticate=authenticate)
    client = restarted.test_client()
    login(client)
    assert client.get('/api/jobs/' + job['id']).json['state'] == 'queued'


def test_ui_schema_covers_cli_fields_and_server_only_profiles(app):
    client = app.test_client()
    login(client)
    response = client.get('/api/schema')
    assert response.status_code == 200
    for name in ('orientation', 'axis'):
        fields = response.json['workflows'][name]['fields']
        assert {f['key'] for f in fields} == set(default_values(name)) - {'url'}
    assert response.json['profiles'] == [{'id': 'local', 'label': 'Local · sequential', 'backend': 'local'}]
    assert client.get('/').status_code == 200
    assert client.get('/assets/app.js').status_code == 200


def test_login_failure_is_generic_and_foreign_origin_cannot_submit(app):
    client = app.test_client()
    csrf = client.get('/api/session').json['csrf']
    response = client.post('/api/login', json={'email': 'alice', 'password': 'bad'},
                           headers={'Origin': 'http://localhost', 'X-CSRF-Token': csrf})
    assert response.status_code == 401
    assert 'upstream' not in response.text and 'password' not in response.text
    headers = login(client)
    headers['Origin'] = 'https://attacker.example'
    assert client.post('/api/jobs', json=submission(), headers=headers).status_code == 403


def test_concurrent_duplicate_requests_only_create_one_job(app):
    def send(_):
        client = app.test_client()
        headers = login(client)
        result = client.post('/api/jobs', json=submission(), headers=headers)
        assert result.status_code == 201
        return result.json['id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(set(pool.map(send, range(4)))) == 1


@pytest.mark.parametrize('changes', [{'url': 'http://attacker'}, {'password': 'secret'},
                                    {'project': []}, {'auto_crop_2d': 'true'}])
def test_server_rejects_injected_parameters(app, changes):
    client = app.test_client()
    headers = login(client)
    assert client.post('/api/jobs', json=submission(**changes), headers=headers).status_code == 400


def test_data_directory_cannot_be_reused_for_another_cryosparc_instance(app, tmp_path):
    with pytest.raises(ValueError, match='instance'):
        create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://different.example',
                    'public_url': 'http://localhost', 'allow_http': True}, authenticate=authenticate)


def test_public_deployment_requires_https(tmp_path):
    with pytest.raises(ValueError, match='HTTPS'):
        create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                    'public_url': 'http://lab.example'}, authenticate=authenticate)


def test_optional_shared_work_directory_keeps_database_on_local_disk(tmp_path):
    local, shared = tmp_path / 'local', tmp_path / 'shared'
    app = create_app({'data_dir': str(local), 'work_dir': str(shared),
                      'cryosparc_url': 'https://cryo.example', 'public_url': 'http://localhost',
                      'allow_http': True}, authenticate=authenticate)
    client = app.test_client()
    headers = login(client)
    job = client.post('/api/jobs', json=submission(), headers=headers).json
    assert (shared / job['id'] / 'request.json').exists()
    assert (local / 'jobs.sqlite3').exists()
    assert not (shared / 'jobs.sqlite3').exists()


def test_schema_preserves_physical_units_and_domain_labels(app):
    client = app.test_client()
    login(client)
    fields = {f['key']: f for f in client.get('/api/schema').json['workflows']['axis']['fields']}
    assert fields['low_resolution_A']['label'] == 'Low resolution (Å)'
    assert fields['render_grid_size']['label'] == 'Surface sampling grid size'


def test_slurm_can_be_configured_in_browser_only_by_unlocked_admin(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/' + name)
    config = {'data_dir': str(tmp_path / 'state'), 'cryosparc_url': 'https://cryo.example',
              'public_url': 'http://localhost', 'allow_http': True, 'admin_token': 'admin-secret'}
    app = create_app(config, authenticate=authenticate)
    admin, ordinary = app.test_client(), app.test_client()
    headers, other = login(admin), login(ordinary, 'bob@example.org')
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o700)
    settings = {'work_dir': str(shared), 'python': sys.executable, 'partition': 'cpu',
                'account': '', 'qos': '', 'cpus': 4, 'memory_mb': 16384,
                'time_minutes': 120, 'shared_confirmed': True}
    assert ordinary.post('/api/admin/slurm', json=settings, headers=other).status_code == 403
    assert admin.post('/api/admin/unlock', json={'token': 'wrong'}, headers=headers).status_code == 403
    assert admin.post('/api/admin/unlock', json={'token': 'admin-secret'}, headers=headers).status_code == 200
    assert admin.post('/api/admin/slurm', json=settings, headers=headers).status_code == 200
    assert admin.get('/api/admin/slurm').json['settings']['cpus'] == 4
    body = dict(submission(), profile='slurm')
    result = ordinary.post('/api/jobs', json=body, headers=other)
    assert result.status_code == 201
    assert (shared / result.json['id'] / 'request.json').exists()
    assert str(shared) not in ordinary.get('/api/schema').text
    restarted = create_app(config, authenticate=authenticate).test_client()
    login(restarted)
    assert any(p['id'] == 'slurm' for p in restarted.get('/api/schema').json['profiles'])
    assert restarted.get('/api/admin/slurm').status_code == 403


def test_slurm_setup_form_is_initially_hidden_and_has_no_required_local_fields(app):
    from html.parser import HTMLParser
    class Inputs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = {}
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.elements[attrs['id']] = attrs
    markup = Inputs()
    markup.feed(app.test_client().get('/').text)
    assert 'hidden' in markup.elements['slurm-panel']
    for name in ('work_dir', 'python', 'partition', 'account', 'cpus', 'memory_mb', 'time_minutes'):
        assert 'required' not in markup.elements['slurm-' + name]


def test_unused_invalid_slurm_does_not_block_local_launch_or_jobs(tmp_path):
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
        'public_url': 'http://localhost', 'allow_http': True,
        'profiles': {'local': {'backend': 'local'}, 'cluster': {'backend': 'slurm', 'cpus': -1}}},
        authenticate=authenticate, start_dispatcher=True)
    try:
        client = app.test_client()
        headers = login(client)
        assert client.post('/api/jobs', json=submission(), headers=headers).status_code == 201
        response = client.post('/api/jobs', json=dict(submission(), profile='cluster'), headers=headers)
        assert response.status_code == 400
        assert 'cpus' in response.json['error']
    finally:
        app.extensions['dispatcher'].close()


def test_missing_slurm_and_invalid_settings_are_reported_without_affecting_local_jobs(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    app = create_app({'data_dir': str(tmp_path / 'state'), 'cryosparc_url': 'https://cryo.example',
                     'public_url': 'http://localhost', 'allow_http': True, 'admin_token': 'key'},
                    authenticate=authenticate)
    client = app.test_client()
    headers = login(client)
    assert client.post('/api/admin/unlock', json={'token': '錯誤'}, headers=headers).status_code == 403
    client.post('/api/admin/unlock', json={'token': 'key'}, headers=headers)
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o700)
    settings = dict(work_dir=str(shared), python=sys.executable, partition='', account='', qos='',
                    cpus=4, memory_mb=16384, time_minutes=120, shared_confirmed=True)
    for change in ({'cpus': 0}, {'cpus': True}, {'partition': 'cpu\n--uid=root'},
                   {'python': 'python'}, {'shared_confirmed': False}):
        assert client.post('/api/admin/slurm', json=dict(settings, **change), headers=headers).status_code == 400
    response = client.post('/api/admin/slurm', json=settings, headers=headers)
    assert response.status_code == 400
    assert 'sbatch' in response.json['error']
    assert client.post('/api/jobs', json=submission(), headers=headers).status_code == 201


def test_queued_slurm_jobs_keep_their_directory_and_resources_after_edit_and_restart(tmp_path, monkeypatch):
    import os
    import uuid
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    # Fake the cluster command boundary; dispatch, persistence and HTTP remain real.
    script = f'#!{sys.executable}\n' + '''import json, pathlib, sys
if pathlib.Path(sys.argv[0]).name == 'sbatch':
    directory = pathlib.Path(sys.argv[-1]).parent
    cpus = next(a for a in sys.argv if a.startswith('--cpus-per-task='))
    (directory / 'output.log').write_text(cpus)
    (directory / 'result.json').write_text(json.dumps({'exit_code': 0}))
    print('471')
'''
    for name in ('sbatch', 'squeue', 'sacct'):
        path = binaries / name
        path.write_text(script)
        path.chmod(0o700)
    monkeypatch.setenv('PATH', str(binaries) + os.pathsep + os.environ['PATH'])
    config = {'data_dir': str(tmp_path / 'state'), 'cryosparc_url': 'https://cryo.example',
              'public_url': 'http://localhost', 'allow_http': True, 'admin_token': 'admin-secret'}
    client = create_app(config, authenticate=authenticate).test_client()
    headers = login(client)
    client.post('/api/admin/unlock', json={'token': 'admin-secret'}, headers=headers)
    job_ids = []
    for cpus in (4, 9):
        directory = tmp_path / f'shared-{cpus}'
        directory.mkdir(mode=0o700)
        settings = dict(work_dir=str(directory), python=sys.executable, partition='', account='', qos='',
                        cpus=cpus, memory_mb=16384, time_minutes=120, shared_confirmed=True)
        assert client.post('/api/admin/slurm', json=settings, headers=headers).status_code == 200
        response = client.post('/api/jobs', json=dict(submission(), profile='slurm', request_id=str(uuid.uuid4())), headers=headers)
        assert response.status_code == 201
        job_ids.append(response.json['id'])
    restarted = create_app(config, authenticate=authenticate, start_dispatcher=True)
    try:
        client = restarted.test_client()
        login(client)
        deadline = time.monotonic() + 18
        while time.monotonic() < deadline:
            if all(client.get('/api/jobs/' + job_id).json['state'] == 'completed' for job_id in job_ids):
                break
            time.sleep(.05)
        for cpus, job_id in zip((4, 9), job_ids):
            assert client.get('/api/jobs/' + job_id).json['state'] == 'completed'
            assert client.get('/api/jobs/' + job_id + '/log').json['log'] == f'--cpus-per-task={cpus}'
            assert not (tmp_path / f'shared-{cpus}' / job_id / 'config/cryosparc-tools/auth.json').exists()
    finally:
        restarted.extensions['dispatcher'].close()


def test_local_queue_is_serial_and_survives_browser_logout(tmp_path):
    # Fake only the external executable; use real HTTP routes, SQLite, queue and processes.
    worker = tmp_path / 'fake-python'
    worker.write_text(f'#!{sys.executable}\n' + '''import json, pathlib, sys, time
directory = pathlib.Path(sys.argv[-1])
email = json.loads((directory / 'request.json').read_text())['email']
(directory / 'output.log').write_text('Owner: ' + email)
while not (directory.parent / 'release').exists():
    time.sleep(.02)
(directory / 'result.json').write_text(json.dumps({'exit_code': 0}))
''')
    worker.chmod(0o700)
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                      'public_url': 'http://localhost', 'allow_http': True,
                      'profiles': {'local': {'backend': 'local', 'python': str(worker)}}},
                     authenticate=authenticate, start_dispatcher=True)
    def wait_for(client, job_id, state):
        deadline = time.monotonic() + 18
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/' + job_id).json
            if job['state'] == state:
                return
            time.sleep(.05)
        pytest.fail(f'Expected {state}, got {job}')
    try:
        alice, bob = app.test_client(), app.test_client()
        a, b = login(alice), login(bob, 'bob@example.org')
        first = alice.post('/api/jobs', json=submission(), headers=a).json['id']
        second = bob.post('/api/jobs', json=submission(), headers=b).json['id']
        wait_for(alice, first, 'running')
        assert bob.get('/api/jobs/' + second).json['state'] == 'queued'
        alice.post('/api/logout', headers=a)
        (tmp_path / 'release').touch()
        login(alice)
        wait_for(alice, first, 'completed')
        wait_for(bob, second, 'completed')
        assert 'alice@example.org' in alice.get('/api/jobs/' + first + '/log').json['log']
        assert 'alice@example.org' not in bob.get('/api/jobs/' + second + '/log').json['log']
        assert not (tmp_path / first / 'config/cryosparc-tools/auth.json').exists()
    finally:
        (tmp_path / 'release').touch()
        app.extensions['dispatcher'].close()
