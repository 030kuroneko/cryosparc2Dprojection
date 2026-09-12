"""HTTP contracts for the multi-user launcher; no live CryoSPARC required."""
import pytest
import sys
import time
from concurrent.futures import ThreadPoolExecutor
pytest.importorskip('flask', reason='Install the web extra to test the HTTP launcher')
from cryosparc_2d_projection.workflow_config import default_values

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


def test_login_preserves_bearer_token_through_cryosparc_web_proxy(tmp_path, monkeypatch):
    import re
    import httpx
    from hashlib import sha256
    from urllib.parse import parse_qs

    response_schema = {'200': {'content': {'application/json': {'schema': {}}}}}
    schema = {'info': {}, 'components': {}, 'paths': {
        '/token': {'post': {'summary': 'login', 'responses': response_schema,
                           'requestBody': {'content': {'application/x-www-form-urlencoded': {}}}}},
        '/users/me': {'get': {'summary': 'users.me', 'responses': response_schema}},
    }}
    calls = []

    def upstream(request):
        path = request.url.path
        calls.append(path)
        if path.endswith('/openapi.json'):
            return httpx.Response(200, json=schema)
        if path.endswith('/token'):
            form = parse_qs(request.content.decode())
            assert form['username'] == ['alice@example.org']
            assert form['password'] == [sha256(b'correct').hexdigest()]
            return httpx.Response(200, json={'access_token': 'upstream-secret', 'token_type': 'bearer'})
        assert path.endswith('/users/me')
        # CryoSPARC 5.0.6 forwards Authorization only for its Tools user agent;
        # other clients use the browser session (empty in this API-only flow).
        tools_client = re.fullmatch(r'cryosparc-tools/\S+', request.headers.get('user-agent', ''), re.I)
        bearer = request.headers.get('authorization') if tools_client else 'Bearer '
        if bearer != 'Bearer upstream-secret':
            return httpx.Response(401, json={'detail': 'Could not validate credentials'})
        return httpx.Response(200, json={'_id': 'authoritative-user-id'})

    real_client = httpx.Client
    def client_with_transport(*args, **kwargs):
        return real_client(*args, **kwargs, transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(httpx, 'Client', client_with_transport)
    client = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                         'public_url': 'http://localhost', 'allow_http': True}).test_client()
    login(client)
    assert client.get('/api/session').json['email'] == 'alice@example.org'
    assert any(path.endswith('/token') for path in calls)
    assert calls[-1].endswith('/users/me')


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


def test_lan_http_login_can_allocate_request_ids_and_submit_without_browser_crypto(tmp_path):
    import uuid
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
        'public_url': 'http://192.168.1.20:40000', 'host': '0.0.0.0'}, authenticate=authenticate)
    client = app.test_client()
    origin = 'http://192.168.1.20:40000'
    assert client.get('/api/request-id', base_url=origin).status_code == 401
    csrf = client.get('/api/session', base_url=origin).json['csrf']
    result = client.post('/api/login', base_url=origin,
        json={'email': 'alice', 'password': 'correct'},
        headers={'Origin': origin, 'X-CSRF-Token': csrf})
    assert result.status_code == 200
    ids = [client.get('/api/request-id', base_url=origin).json['request_id'] for _ in range(2)]
    assert ids[0] != ids[1]
    assert all(uuid.UUID(value).version == 4 for value in ids)
    headers = {'Origin': origin, 'X-CSRF-Token': result.json['csrf']}
    response = client.post('/api/jobs', base_url=origin,
        json=dict(submission(), request_id=ids[0]), headers=headers)
    assert response.status_code == 201
    assert client.post('/api/jobs', base_url=origin,
        json=dict(submission(), request_id=ids[0]), headers=headers).json['id'] == response.json['id']
    assert client.post('/api/logout', base_url=origin,
        headers=dict(headers, Origin='http://192.168.1.20:40001')).status_code == 403


@pytest.mark.parametrize('hostname', ['192.168.1.20', '10.2.3.4', '[2001:db8::20]', 'lab-node'])
def test_automatic_origin_supports_same_origin_login_and_jobs(tmp_path, monkeypatch, hostname):
    import socket
    monkeypatch.setattr(socket, 'gethostname', lambda: 'lab-node')
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                      'host': '0.0.0.0', 'port': 40100}, authenticate=authenticate)
    client = app.test_client()
    origin = f'http://{hostname}:40100'
    csrf = client.get('/api/session', base_url=origin).json['csrf']
    response = client.post('/api/login', base_url=origin,
        json={'email': 'alice', 'password': 'correct'},
        headers={'Origin': origin, 'X-CSRF-Token': csrf})
    assert response.status_code == 200
    headers = {'Origin': origin, 'X-CSRF-Token': response.json['csrf']}
    assert client.post('/api/jobs', base_url=origin, json=submission(), headers=headers).status_code == 201
    for other_origin in ('http://evil.example:40100', 'http://192.168.1.20:40101', 'null', ''):
        assert client.post('/api/logout', base_url=origin,
            headers=dict(headers, Origin=other_origin)).status_code == 403
    assert client.post('/api/logout', base_url=origin,
        headers={'Origin': origin, 'X-CSRF-Token': 'wrong'}).status_code == 403


@pytest.mark.parametrize('authority', [
    'evil.example:40000', 'localhost.evil.example:40000', '127.0.0.1.evil.example:40000',
    '192.168.1.20:40001', '0.0.0.0:40000', '[::]:40000', '224.0.0.1:40000',
])
def test_automatic_origin_rejects_untrusted_authorities_even_with_forwarded_headers(tmp_path, authority):
    app = create_app({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
                      'host': '0.0.0.0'}, authenticate=authenticate)
    response = app.test_client().get('/api/session', base_url='http://' + authority,
        headers={'X-Forwarded-Host': 'localhost:40000', 'X-Forwarded-Proto': 'https'})
    assert response.status_code == 400
    assert 'Set-Cookie' not in response.headers


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


def test_job_log_separates_details_and_exposes_private_latest_progress(app, tmp_path):
    import json
    alice, bob = app.test_client(), app.test_client()
    headers = login(alice)
    login(bob, 'bob@example.org')
    job = alice.post('/api/jobs', json=submission(), headers=headers).json
    directory = tmp_path / job['id']
    (directory / 'output.log').write_text(
        'Reading input data\n[Details] stage=input-loading\nWARNING: Check map\n')
    (directory / 'progress.json').write_text(json.dumps({
        'state': 'running', 'stage': 'Uploading results', 'remaining_seconds': None,
    }))
    response = alice.get('/api/jobs/' + job['id'] + '/log')
    assert response.status_code == 200
    assert response.json['log'] == 'Reading input data\nWARNING: Check map'
    assert response.json['details'] == 'stage=input-loading'
    assert response.json['progress']['stage'] == 'Uploading results'
    assert bob.get('/api/jobs/' + job['id'] + '/log').status_code == 404


def test_progress_script_is_served_with_the_launcher(app):
    client = app.test_client()
    assert '/assets/progress.js' in client.get('/').text
    script = client.get('/assets/progress.js')
    assert script.status_code == 200
    assert 'JobProgressView' in script.text


@pytest.mark.parametrize('cut_inside_line', [False, True])
def test_log_tail_keeps_complete_lines_and_hides_truncated_details(app, tmp_path, cut_inside_line):
    client = app.test_client()
    headers = login(client)
    job = client.post('/api/jobs', json=submission(), headers=headers).json
    suffix = '[Details] complete diagnostic\nWARNING: Check map\n'
    if cut_inside_line:
        content = ('[Details] ' + 'internal-state ' * 120 + '\n') * 40 + suffix
    else:
        # The 64 KiB boundary is already at a complete warning line.
        suffix = 'WARNING: Keep boundary warning\n' + suffix
        padding = '[Details] ' + 'x' * (65536 - len(suffix) - 11) + '\n'
        content = 'previous line\n' + suffix + padding
    (tmp_path / job['id'] / 'output.log').write_text(content)
    result = client.get('/api/jobs/' + job['id'] + '/log').json
    assert 'internal-state' not in result['log']
    assert 'WARNING: Check map' in result['log']
    assert 'complete diagnostic' in result['details']
    if not cut_inside_line:
        assert 'WARNING: Keep boundary warning' in result['log']


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
        assert {f['key'] for f in fields} == {
            key for key in default_values(name) if key != 'url' and not key.endswith('_output')}
    assert response.json['profiles'] == [{'id': 'local', 'label': 'Local · sequential', 'backend': 'local'}]
    assert client.get('/').status_code == 200
    assert client.get('/assets/app.js').status_code == 200


@pytest.mark.parametrize('workflow,expected', [
    ('orientation', {'select_output': 'particles_selected', 'templates_output': 'templates_selected',
                     'refinement_particles_output': 'particles', 'volume_output': 'volume'}),
    ('axis', {'select_output': 'templates_selected', 'volume_output': 'volume'}),
])
def test_hidden_source_outputs_use_workflow_defaults(app, workflow, expected):
    client = app.test_client()
    headers = login(client)
    fields = client.get('/api/schema').json['workflows'][workflow]['fields']
    assert not (set(expected) & {field['key'] for field in fields})
    values = {field['key']: field['default'] for field in fields}
    values.update(project='P1', workspace='W2', select_job='J3')
    values['refinement_job' if workflow == 'orientation' else 'volume_job'] = 'J4'
    response = client.post('/api/jobs', json=dict(submission(), workflow=workflow, values=values), headers=headers)
    assert response.status_code == 201
    assert {key: response.json['values'][key] for key in expected} == expected


def test_ui_offers_settings_export_and_run_without_copy_command(app):
    page = app.test_client().get('/').text
    assert 'id="copy-command"' not in page
    assert 'id="save-settings"' in page
    assert 'id="submit-job"' in page


def test_theme_control_and_script_are_available_before_login(app):
    client = app.test_client()
    page = client.get('/').text
    assert 'id="theme-toggle"' in page
    assert page.index('/assets/theme.js') < page.index('/assets/app.css')
    assert client.get('/assets/theme.js').status_code == 200


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


@pytest.mark.parametrize('gpus', [None, 1])
def test_slurm_can_be_configured_in_browser_only_by_unlocked_admin(tmp_path, monkeypatch, gpus):
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
    if gpus is not None:
        settings['gpus'] = gpus
    assert ordinary.post('/api/admin/slurm', json=settings, headers=other).status_code == 403
    assert admin.post('/api/admin/unlock', json={'token': 'wrong'}, headers=headers).status_code == 403
    assert admin.post('/api/admin/unlock', json={'token': 'admin-secret'}, headers=headers).status_code == 200
    assert admin.post('/api/admin/slurm', json=settings, headers=headers).status_code == 200
    assert admin.get('/api/admin/slurm').json['settings']['cpus'] == 4
    if gpus is not None:
        assert admin.get('/api/admin/slurm').json['settings']['gpus'] == gpus
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


def test_symmetry_field_allows_arbitrary_point_group_order(app):
    client = app.test_client()
    login(client)
    fields = client.get('/api/schema').json['workflows']['orientation']['fields']
    symmetry = next(field for field in fields if field['key'] == 'symmetry')
    assert symmetry['choices'] == []
    assert 'Cn' in symmetry['hint'] and 'Dn' in symmetry['hint']


@pytest.mark.parametrize('symmetry', ['C3', 'C11', 'D1', 'D7', 'T', 'O'])
def test_submitted_job_preserves_point_group_symmetry(app, symmetry):
    client = app.test_client()
    headers = login(client)
    response = client.post('/api/jobs', json=submission(symmetry=symmetry), headers=headers)
    assert response.status_code == 201
    assert response.json['values']['symmetry'] == symmetry


def test_axis_schema_and_submission_support_selected_symmetry(app):
    client = app.test_client()
    headers = login(client)
    fields = {f['key']: f for f in client.get('/api/schema').json['workflows']['axis']['fields']}
    assert fields['symmetry']['default'] == 'I'
    assert fields['symmetry']['group'] == 'basic'
    assert fields['symmetry']['choices'] == []
    assert 'I convention only' not in fields['axis_family']['hint']
    values = default_values('axis')
    values.update(project='P1', workspace='W1', select_job='J1', volume_job='J2',
                  symmetry='O', axis_family='4fold', axis_roll='4fold=30')
    values.pop('url')
    body = dict(submission(), workflow='axis', values=values)
    response = client.post('/api/jobs', json=body, headers=headers)
    assert response.status_code == 201
    assert response.json['values']['symmetry'] == 'O'
    body['values']['axis_family'] = '5fold'
    assert client.post('/api/jobs', json=body, headers=headers).status_code == 400


def test_schema_explains_every_visible_field_and_provides_job_examples(app):
    client = app.test_client()
    login(client)
    workflows = client.get('/api/schema').json['workflows']
    for workflow in workflows.values():
        for field in workflow['fields']:
            assert field['hint'].strip(), field['key']
            assert 'help' in field
        fields = {field['key']: field for field in workflow['fields']}
        assert fields['select_job']['placeholder'] == 'e.g. J123'
        assert '256' in fields['render_grid_size']['help']
        assert 'Volumes' in fields['surface_level']['help']
        assert 'same map' in fields['surface_level']['help']
        assert '1024' in fields['render_size']['hint']
    fields = {field['key']: field for field in workflows['orientation']['fields']}
    assert fields['refinement_job']['placeholder'] == 'e.g. J124'
    assert 'overlap' in fields['refinement_job']['help']


def test_page_exposes_help_script_and_admin_field_guidance(app):
    client = app.test_client()
    page = client.get('/').text
    assert '/assets/help.js' in page
    assert client.get('/assets/help.js').status_code == 200
    for field in ('profile', 'admin-key', 'slurm-work_dir', 'slurm-python',
                  'slurm-partition', 'slurm-account', 'slurm-qos', 'slurm-cpus',
                  'slurm-memory_mb', 'slurm-time_minutes'):
        assert f'id="{field}-hint"' in page
        assert f'aria-describedby="{field}-hint"' in page


def test_cleanup_warning_is_owner_scoped_and_clears_without_changing_outcome(app, tmp_path, monkeypatch):
    from pathlib import Path
    from cryosparc_2d_projection.web_jobs import JobStore
    from cryosparc_2d_projection.web_execution import Dispatcher
    client = app.test_client()
    headers = login(client)
    job_id = client.post('/api/jobs', json=submission(), headers=headers).json['id']
    now = [1000]
    store = JobStore({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example'},
                     clock=lambda: now[0])
    auth = store.directory(job_id) / 'config/cryosparc-tools/auth.json'
    unlink = Path.unlink
    def remove(path, **kwargs):
        if path == auth:
            raise PermissionError('private cleanup error')
        return unlink(path, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', remove)
        store.update(job_id, 'completed')
    response = client.get('/api/jobs/' + job_id)
    assert response.json['state'] == 'completed'
    assert response.json['cleanup_pending'] is True
    assert 'private cleanup error' not in response.text
    assert client.get('/api/jobs').json['jobs'][0]['cleanup_pending'] is True
    bob = app.test_client()
    login(bob, 'bob@example.org')
    assert bob.get('/api/jobs/' + job_id).status_code == 404
    assert bob.get('/api/jobs').json['jobs'] == []
    now[0] = 1005
    Dispatcher(store).tick()
    response = client.get('/api/jobs/' + job_id)
    assert response.json['state'] == 'completed'
    assert response.json['cleanup_pending'] is False
