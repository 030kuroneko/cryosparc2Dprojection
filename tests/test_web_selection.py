"""Selection behavior through its authenticated HTTP boundary."""
import json
import time
import uuid
from flask import Flask, g
from cryosparc_2d_projection.web_jobs import JobStore
from cryosparc_2d_projection.web_selection import register_selection_routes


def setup(tmp_path):
    store = JobStore({'data_dir': str(tmp_path), 'cryosparc_url': 'https://cryo.example'})
    with store.connect() as db:
        db.execute("INSERT INTO jobs (id,owner,request_id,workflow,profile,values_json,state,created) VALUES ('one','alice','r','orientation','local','{}','completed','now')")
    directory = store.directory('one') / 'selection'
    directory.mkdir(parents=True)
    manifest = dict(schema_version=1, project_uid='P1', workspace_uid='W1', source_job_uid='J3',
                    particles_source={'job_uid':'J1','output':'particles'}, templates_source={'job_uid':'J1','output':'class_averages'},
                    classes=[dict(class_number=n,particle_count=10,score=.5,orientation_method='image',confidence='low',image=f'class_{n}.png') for n in [1,2]])
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    app = Flask(__name__)
    @app.before_request
    def identity():
        g.identity = {'owner': app.config.get('OWNER', 'alice'), 'email':'alice@example.org', 'token':'secret'}
    register_selection_routes(app, store)
    return app, store


def test_selection_is_empty_then_saved_with_revision_and_owner_isolation(tmp_path):
    app, store = setup(tmp_path)
    client = app.test_client()
    path = '/api/jobs/one/selection'
    assert client.get(path).json['selected_class_numbers'] == []
    saved = client.put(path, json={'selected_class_numbers':[2], 'revision':0})
    assert saved.status_code == 200
    assert saved.json['revision'] == 1
    assert client.get(path).json['selected_class_numbers'] == [2]
    assert client.put(path, json={'selected_class_numbers':[1], 'revision':0}).status_code == 409
    app.config['OWNER'] = 'bob'
    assert client.get(path).status_code == 404


def test_images_require_owner_and_manifest_membership_and_legacy_is_unavailable(tmp_path):
    app, store = setup(tmp_path)
    client = app.test_client()
    image = store.directory('one') / 'selection' / 'class_1.png'
    image.write_bytes(b'image')
    path = '/api/jobs/one/selection'
    assert client.get(path + '/images/class_1.png').data == b'image'
    assert client.get(path + '/images/manifest.json').status_code == 404
    app.config['OWNER'] = 'bob'
    assert client.get(path + '/images/class_1.png').status_code == 404
    app.config['OWNER'] = 'alice'
    (image.parent / 'manifest.json').unlink()
    assert client.get(path).json['available'] is False


def wait_export(client, path, state):
    for _ in range(200):
        exports = client.get(path).json['exports']
        if exports and exports[0]['state'] == state:
            return exports[0]
        time.sleep(.01)
    raise AssertionError(exports)


def test_export_snapshots_selection_and_duplicate_request_never_creates_twice(tmp_path):
    from test_class_selection_export import setup_export
    from threading import Event
    app, store = setup(tmp_path)
    project, job, manifest, _, _ = setup_export()
    path = store.directory('one') / 'selection' / 'manifest.json'
    original = json.loads(path.read_text())
    original.update(templates_source=manifest['templates_source'])
    original['classes'][0]['class_number'] = 2
    original['classes'][0]['image'] = 'class_2.png'
    original['classes'][1]['class_number'] = 5
    original['classes'][1]['image'] = 'class_5.png'
    path.write_text(json.dumps(original))
    app.config['SELECTION_PROJECT_FACTORY'] = lambda identity, uid: project
    started, release = Event(), Event()
    def create(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return job
    project.create_external_job.side_effect = create
    client = app.test_client()
    path = '/api/jobs/one/selection'
    body = {'request_id':str(uuid.uuid4()), 'selected_class_numbers':[5]}
    response = client.post(path + '/exports', json=body)
    assert response.status_code == 202
    assert started.wait(3)
    try:
        assert client.put(path, json={'selected_class_numbers':[2], 'revision':0}).status_code == 200
        assert client.post(path + '/exports', json=body).json['id'] == response.json['id']
    finally:
        release.set()
    exported = wait_export(client, path, 'completed')
    assert exported['selected_class_numbers'] == [5]
    assert client.get(path).json['selected_class_numbers'] == [2]
    assert project.create_external_job.call_count == 1


def export_app(tmp_path):
    from test_class_selection_export import setup_export
    app, store = setup(tmp_path)
    project, job, manifest, _, _ = setup_export()
    path = store.directory('one') / 'selection' / 'manifest.json'
    data = json.loads(path.read_text())
    data['templates_source'] = manifest['templates_source']
    for item, n in zip(data['classes'], [2,5]):
        item.update(class_number=n, image=f'class_{n}.png')
    path.write_text(json.dumps(data))
    app.config['SELECTION_PROJECT_FACTORY'] = lambda identity, uid: project
    return app, store, project, job


def test_failed_publication_retry_uses_same_job_and_original_snapshot(tmp_path):
    app, store, project, job = export_app(tmp_path)
    project.find_external_job.return_value = job
    job.save_output.side_effect = TimeoutError('secret-token')
    client = app.test_client()
    path = '/api/jobs/one/selection'
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[5]})
    failed = wait_export(client, path, 'failed')
    assert failed['job_uid'] == 'J9'
    assert 'secret-token' not in failed['detail']
    job.save_output.side_effect = None
    response = client.post(path + '/exports/' + failed['id'] + '/retry')
    assert response.status_code == 202
    assert wait_export(client, path, 'completed')['selected_class_numbers'] == [5]
    assert project.create_external_job.call_count == 1


def test_unknown_creation_blocks_retry_and_new_export(tmp_path):
    app, store, project, job = export_app(tmp_path)
    project.create_external_job.side_effect = TimeoutError('secret')
    client = app.test_client()
    path = '/api/jobs/one/selection'
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[5]})
    unknown = wait_export(client, path, 'unknown')
    assert client.post(path + '/exports/' + unknown['id'] + '/retry').status_code == 409
    assert client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[2]}).status_code == 409
    assert project.create_external_job.call_count == 1


def test_restart_preserves_selection_and_completed_exports(tmp_path):
    app, store, project, job = export_app(tmp_path)
    client = app.test_client()
    path = '/api/jobs/one/selection'
    client.put(path, json={'selected_class_numbers':[5], 'revision':0})
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[5]})
    completed = wait_export(client, path, 'completed')
    restarted = Flask('restarted')
    @restarted.before_request
    def identity():
        g.identity = {'owner':'alice'}
    register_selection_routes(restarted, store)
    result = restarted.test_client().get(path).json
    assert result['selected_class_numbers'] == [5]
    assert result['exports'][0] == completed


def test_sdk_token_session_does_not_load_or_save_global_credentials(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    app, store = setup(tmp_path)
    api = MagicMock()
    api.users.me.return_value = SimpleNamespace(id='alice')
    # Authentication reaches project lookup before any source loading or job creation.
    api.projects.find_one.side_effect = RuntimeError('project unavailable secret')
    api_type = MagicMock(return_value=api)
    monkeypatch.setattr('cryosparc.api.APIClient', api_type)
    from cryosparc.auth import InstanceAuthSessions
    monkeypatch.setattr(InstanceAuthSessions, 'load', lambda: (_ for _ in ()).throw(AssertionError('global credentials accessed')))
    client = app.test_client()
    path = '/api/jobs/one/selection'
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[1]})
    failed = wait_export(client, path, 'failed')
    assert 'secret' not in failed['detail']
    assert api_type.call_args.kwargs['auth'] == 'secret'
    assert api_type.call_args.kwargs['timeout'] == 30


def test_corrupt_manifest_and_invalid_selection_cannot_start_exports(tmp_path):
    app, store = setup(tmp_path)
    client = app.test_client()
    path = '/api/jobs/one/selection'
    for values in ([True], [99], [1,1]):
        assert client.put(path, json={'selected_class_numbers':values,'revision':0}).status_code == 400
    assert client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[]}).status_code == 400
    manifest = store.directory('one') / 'selection' / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['classes'][0]['orientation_method'] = 'x' * 100000
    manifest.write_text(json.dumps(data))
    assert client.get(path).json['available'] is False


def test_concurrent_retry_claims_only_one_remote_publication(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    app, store, project, job = export_app(tmp_path)
    project.find_external_job.return_value = job
    job.save_output.side_effect = TimeoutError('interrupted')
    client = app.test_client()
    path = '/api/jobs/one/selection'
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[5]})
    failed = wait_export(client, path, 'failed')
    entered, release = Event(), Event()
    def save(*args, **kwargs):
        entered.set()
        assert release.wait(3)
    job.save_output.side_effect = save
    retry = path + '/exports/' + failed['id'] + '/retry'
    try:
        with ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(lambda _: app.test_client().post(retry).status_code, range(2)))
        assert responses == [202,202]
        assert entered.wait(3)
    finally:
        release.set()
    wait_export(client, path, 'completed')
    assert project.find_external_job.call_count == 1
    assert project.create_external_job.call_count == 1


def test_selection_errors_are_json_and_explain_recovery(tmp_path):
    app, store, project, job = export_app(tmp_path)
    client = app.test_client()
    path = '/api/jobs/one/selection'
    invalid = client.put(path, json={'selected_class_numbers':[999], 'revision':0})
    assert invalid.json['error'] == 'Invalid class selection'
    project.create_external_job.side_effect = TimeoutError('unknown')
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[5]})
    unknown = wait_export(client, path, 'unknown')
    retry = client.post(path + '/exports/' + unknown['id'] + '/retry')
    assert retry.status_code == 409
    assert 'Administrator' in retry.json['error']
    unrelated = client.get('/missing')
    assert unrelated.status_code == 404 and not unrelated.is_json


def test_malformed_manifest_shapes_are_unavailable_not_server_errors(tmp_path):
    app, store = setup(tmp_path)
    client = app.test_client()
    path = store.directory('one') / 'selection' / 'manifest.json'
    original = json.loads(path.read_text())
    for classes in ([None], [[]], ['class'], {'class_number':1}, None):
        path.write_text(json.dumps(dict(original, classes=classes)))
        result = client.get('/api/jobs/one/selection')
        assert result.status_code == 200
        assert result.json['available'] is False
    original['classes'][0]['score'] = 10 ** 1000
    path.write_text(json.dumps(original))
    assert client.get('/api/jobs/one/selection').json['available'] is False


def test_source_preflight_failure_is_retryable_without_unknown_creation(tmp_path):
    app, store, project, job = export_app(tmp_path)
    original_lookup = project.find_job.side_effect
    project.find_job.side_effect = TimeoutError('source unavailable')
    client = app.test_client()
    path = '/api/jobs/one/selection'
    client.post(path + '/exports', json={'request_id':str(uuid.uuid4()),'selected_class_numbers':[5]})
    failed = wait_export(client, path, 'failed')
    assert failed['job_uid'] is None
    assert project.create_external_job.call_count == 0
    project.find_job.side_effect = original_lookup
    assert client.post(path + '/exports/' + failed['id'] + '/retry').status_code == 202
    wait_export(client, path, 'completed')
    assert project.create_external_job.call_count == 1


def test_missing_export_retry_preserves_not_found_response(tmp_path):
    from werkzeug.exceptions import NotFound

    app, _ = setup(tmp_path)
    response = app.test_client().post('/api/jobs/one/selection/exports/missing/retry')
    assert response.status_code == 404
    assert response.json == {'error': NotFound.description}
