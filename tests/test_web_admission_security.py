"""Public HTTP and CLI regression contracts for bounded Web admission."""
import time

import pytest

from cryosparc_2d_projection.web import create_app
from cryosparc_2d_projection.workflow_config import default_values


@pytest.fixture
def app(tmp_path):
    def authenticate(url, email, password):
        if password != 'correct':
            raise ValueError('Invalid credentials')
        return {'owner': email, 'email': email, 'token': 'private-token-' + email}

    application = create_app({
        'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example',
        'public_url': 'http://localhost', 'allow_http': True,
    }, authenticate=authenticate)
    application.testing = True
    return application


def sign_in(client, csrf=None, email='alice@example.org'):
    if csrf is None:
        csrf = client.get('/api/session').json['csrf']
    return client.post('/api/login', json={'email': email, 'password': 'correct'},
                       headers={'Origin': 'http://localhost', 'X-CSRF-Token': csrf})


def test_anonymous_flood_preserves_existing_and_new_sign_ins(app):
    signed_in, waiting = app.test_client(), app.test_client()
    assert sign_in(signed_in).status_code == 200
    waiting_csrf = waiting.get('/api/session').json['csrf']

    # Same source address as the legitimate browsers, without retaining cookies.
    flood = app.test_client(use_cookies=False)
    for _ in range(2050):
        assert flood.get('/api/session').status_code == 200

    assert signed_in.get('/api/session').json['email'] == 'alice@example.org'
    assert signed_in.get('/api/jobs').status_code == 200
    assert sign_in(waiting, waiting_csrf, 'waiting@example.org').status_code == 200
    assert sign_in(app.test_client(), email='new@example.org').status_code == 200


@pytest.mark.parametrize('recovery', ['logout', 'expiry'])
def test_full_authenticated_capacity_preserves_sessions_and_recovers(app, monkeypatch, recovery):
    first = None
    for number in range(2048):
        client = app.test_client()
        client.environ_base['REMOTE_ADDR'] = f'10.0.{number // 256}.{number % 256}'
        response = sign_in(client)
        assert response.status_code == 200
        if first is None:
            first = client

    newcomer = app.test_client()
    csrf = newcomer.get('/api/session').json['csrf']
    refused = sign_in(newcomer, csrf)
    assert refused.status_code == 503
    assert 'try again' in refused.json['error'].lower()
    assert int(refused.headers['Retry-After']) > 0
    assert first.get('/api/jobs').status_code == 200

    # Reauthentication replaces the caller's own slot, even when otherwise full.
    renewed = sign_in(first)
    assert renewed.status_code == 200
    if recovery == 'logout':
        headers = {'Origin': 'http://localhost', 'X-CSRF-Token': renewed.json['csrf']}
        assert first.post('/api/logout', headers=headers).status_code == 200
        assert sign_in(newcomer, csrf).status_code == 200
    else:
        later = time.time() + 28801
        monkeypatch.setattr(time, 'time', lambda: later)
        assert first.get('/api/jobs').status_code == 401
        assert sign_in(newcomer).status_code == 200


@pytest.mark.parametrize('invalid', ['different_browser', 'tampered_cookie', 'expired', 'forged_cookie'])
def test_anonymous_bootstrap_rejects_invalid_credentials_and_allows_refresh(app, monkeypatch, invalid):
    client = app.test_client()
    csrf = client.get('/api/session').json['csrf']
    if invalid == 'different_browser':
        csrf = app.test_client().get('/api/session').json['csrf']
    elif invalid == 'tampered_cookie':
        client.set_cookie('projection_session', client.get_cookie('projection_session').value + 'x')
    elif invalid == 'expired':
        later = time.time() + 1801
        monkeypatch.setattr(time, 'time', lambda: later)
    else:
        client.set_cookie('projection_session', '{"owner":"alice","admin":true}')

    assert client.get('/api/jobs').status_code == 401
    assert sign_in(client, csrf).status_code == 403
    assert sign_in(client).status_code == 200


def test_login_rotates_bootstrap_and_logout_revokes_authenticated_cookie(app):
    client = app.test_client()
    bootstrap_csrf = client.get('/api/session').json['csrf']
    bootstrap_cookie = client.get_cookie('projection_session').value
    logged_in = sign_in(client, bootstrap_csrf)
    authenticated_cookie = client.get_cookie('projection_session').value
    assert logged_in.json['csrf'] != bootstrap_csrf
    assert authenticated_cookie != bootstrap_cookie
    headers = {'Origin': 'http://localhost', 'X-CSRF-Token': logged_in.json['csrf']}
    assert client.post('/api/logout', headers=headers).status_code == 200
    replay = app.test_client()
    for cookie in (bootstrap_cookie, authenticated_cookie):
        replay.set_cookie('projection_session', cookie)
        assert replay.get('/api/jobs').status_code == 401
    assert sign_in(client).status_code == 200


def job_request(workflow, symmetry):
    values = default_values(workflow)
    values.pop('url')
    values.update(project='P1', workspace='W1', select_job='J1', symmetry=symmetry)
    if symmetry is None:
        values.pop('symmetry')
    values['volume_job' if workflow == 'axis' else 'refinement_job'] = 'J2'
    return {'workflow': workflow, 'values': values, 'profile': 'local',
            'request_id': '3fe2ecb5-57a1-4d0a-8e51-d4270ba8f71d'}


@pytest.mark.parametrize('workflow', ['orientation', 'axis'])
@pytest.mark.parametrize('symmetry', ['C33', 'D33', ' d33 '])
def test_web_rejects_orders_above_budget_without_creating_a_job(app, workflow, symmetry):
    client = app.test_client()
    logged_in = sign_in(client)
    headers = {'Origin': 'http://localhost', 'X-CSRF-Token': logged_in.json['csrf']}
    response = client.post('/api/jobs', json=job_request(workflow, symmetry), headers=headers)
    assert response.status_code == 400
    assert '32' in response.json['error'] and 'CLI' in response.json['error']
    assert client.get('/api/jobs').json['jobs'] == []


@pytest.mark.parametrize('workflow', ['orientation', 'axis'])
@pytest.mark.parametrize('symmetry', ['C32', 'D32', ' d32 ', 'T', 'O', 'I', '', ' ', None])
def test_web_accepts_budget_boundary_and_preserves_defaults_and_retries(app, workflow, symmetry):
    client = app.test_client()
    logged_in = sign_in(client)
    headers = {'Origin': 'http://localhost', 'X-CSRF-Token': logged_in.json['csrf']}
    payload = job_request(workflow, symmetry)
    response = client.post('/api/jobs', json=payload, headers=headers)
    assert response.status_code == 201
    expected = default_values(workflow)['symmetry'] if symmetry is None else symmetry
    assert response.json['values']['symmetry'] == expected
    retried = client.post('/api/jobs', json=payload, headers=headers)
    assert retried.status_code == 201
    assert retried.json['id'] == response.json['id']


@pytest.mark.parametrize('workflow', ['orientation', 'axis'])
@pytest.mark.parametrize('symmetry', ['C01', 'C0', 'C+32', 'D32.0', 'C３２', 'I1'])
def test_web_still_rejects_invalid_symmetry_encodings(app, workflow, symmetry):
    client = app.test_client()
    logged_in = sign_in(client)
    response = client.post('/api/jobs', json=job_request(workflow, symmetry),
        headers={'Origin': 'http://localhost', 'X-CSRF-Token': logged_in.json['csrf']})
    assert response.status_code == 400
    assert client.get('/api/jobs').json['jobs'] == []


@pytest.mark.parametrize('workflow', ['orientation', 'axis'])
@pytest.mark.parametrize('symmetry', ['C33', 'D33', ' d33 '])
def test_cli_still_accepts_orders_above_web_budget(workflow, symmetry):
    from cryosparc_2d_projection import axis_cli, cli
    parser = (axis_cli if workflow == 'axis' else cli).build_parser()
    parsed = parser.parse_args([
        '--url', 'https://cryo.example', '--project', 'P1', '--workspace', 'W1',
        '--select-job', 'J1', '--volume-job' if workflow == 'axis' else '--refinement-job',
        'J2', '--symmetry', symmetry,
    ])
    assert parsed.symmetry == symmetry.strip().upper()


def test_web_schema_explains_its_symmetry_budget(app):
    client = app.test_client()
    assert sign_in(client).status_code == 200
    for workflow in client.get('/api/schema').json['workflows'].values():
        field = next(field for field in workflow['fields'] if field['key'] == 'symmetry')
        assert '32' in field['help'] and 'CLI' in field['help']


@pytest.mark.parametrize('workflow,status', [('orientation', 201), ('axis', 400)])
def test_c1_keeps_its_workflow_specific_behavior(app, workflow, status):
    client = app.test_client()
    logged_in = sign_in(client)
    response = client.post('/api/jobs', json=job_request(workflow, 'C1'),
        headers={'Origin': 'http://localhost', 'X-CSRF-Token': logged_in.json['csrf']})
    assert response.status_code == status
