"""Owner-scoped saved class selections and durable external-job exports."""
import json
import math
import re
from functools import wraps
from werkzeug.exceptions import HTTPException
from flask import abort, g, jsonify, request, send_file

from cryosparc_2d_projection.class_selection_jobs import (
    ClassSelectionExportLifecycle,
    ExportLifecycleError,
    ExportNotFound,
)


def _project(url, identity, project_uid):
    # The SDK constructor only accepts passwords or global disk sessions. Its
    # controllers use these public attributes; initialize an isolated token API.
    from cryosparc import __version__
    from cryosparc.api import APIClient
    from cryosparc.constants import API_SUFFIX
    from cryosparc.tools import CryoSPARC

    class TokenSession(CryoSPARC):
        def __init__(self):
            self.base_url = url
            self.api = APIClient(url + API_SUFFIX, auth=identity['token'], timeout=30,
                                 headers={'User-Agent': f'cryosparc-tools/{__version__}'})
            self.user

    return TokenSession().find_project(project_uid)


def register_selection_routes(app, store):
    def project_factory(identity, project_uid):
        # Look up the factory when the worker starts. Tests and deployments can
        # configure it after route registration without changing token scope.
        factory = app.config.get('SELECTION_PROJECT_FACTORY')
        return factory(identity, project_uid) if factory else _project(
            store.config['cryosparc_url'], identity, project_uid)

    lifecycle = ClassSelectionExportLifecycle(store, project_factory=project_factory)
    app.extensions['class_selection_export_lifecycle'] = lifecycle

    with store.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS class_selections (job_id TEXT PRIMARY KEY, selected TEXT NOT NULL, revision INTEGER NOT NULL)')

    def json_errors(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except HTTPException as error:
                return jsonify(error=error.description), error.code
        return wrapped

    def manifest_for(job_id):
        job = store.get(g.identity['owner'], job_id)
        if not job:
            abort(404)
        if job['workflow'] != 'orientation' or job['state'] != 'completed':
            return None
        path = store.directory(job_id) / 'selection' / 'manifest.json'
        try:
            if path.stat().st_size > 8_000_000:
                return None
            manifest = json.loads(path.read_text())
            if manifest['schema_version'] != 1 or not 0 < len(manifest['classes']) <= 10000:
                return None
            seen = set()
            for item in manifest['classes']:
                number = item['class_number']
                if type(number) is not int or not 1 <= number <= 1000000 or number in seen:
                    return None
                seen.add(number)
                for field in ('orientation_method', 'confidence'):
                    if not isinstance(item.get(field), str) or len(item[field]) > 128:
                        return None
                if type(item['particle_count']) is not int or item['particle_count'] < 0:
                    return None
                if item['score'] is not None and (type(item['score']) not in (int, float) or not math.isfinite(item['score'])):
                    return None
                if item['image'] != f'class_{number}.png':
                    return None
            for key, pattern in [('project_uid', r'P\d+'), ('workspace_uid', r'W\d+'), ('source_job_uid', r'J\d+')]:
                if not re.fullmatch(pattern, manifest[key]):
                    return None
            for key in ('particles_source', 'templates_source'):
                source = manifest[key]
                if not re.fullmatch(r'J\d+', source['job_uid']) or not re.fullmatch(r'[A-Za-z0-9_]{1,128}', source['output']):
                    return None
            return manifest
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return None

    def selected(body, manifest):
        values = body.get('selected_class_numbers') if isinstance(body, dict) else None
        allowed = {item['class_number'] for item in manifest['classes']}
        if not isinstance(values, list) or len(values) > len(allowed) or any(type(n) is not int or n not in allowed for n in values) or len(set(values)) != len(values):
            abort(400, 'Invalid class selection')
        return sorted(values)

    def public_selection(job_id, manifest):
        if manifest is None:
            return dict(available=False, reason='Rerun Class Orientation to enable class selection.')
        with store.connect() as db:
            row = db.execute('SELECT * FROM class_selections WHERE job_id=?', (job_id,)).fetchone()
        return dict(available=True, classes=manifest['classes'], selected_class_numbers=json.loads(row['selected']) if row else [], revision=row['revision'] if row else 0, exports=lifecycle.list_exports(g.identity, job_id))

    @app.get('/api/jobs/<job_id>/selection')
    @json_errors
    def get_selection(job_id):
        return jsonify(public_selection(job_id, manifest_for(job_id)))

    @app.put('/api/jobs/<job_id>/selection')
    @json_errors
    def save_selection(job_id):
        manifest = manifest_for(job_id)
        if manifest is None:
            abort(409, 'Rerun Class Orientation to enable class selection.')
        body = request.get_json(silent=True)
        values = selected(body, manifest)
        revision = body.get('revision')
        if type(revision) is not int or revision < 0:
            abort(400, 'Invalid selection revision; reload before saving.')
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM class_selections WHERE job_id=?', (job_id,)).fetchone()
            if revision != (row['revision'] if row else 0):
                abort(409, 'Selection changed; reload before saving.')
            db.execute('INSERT OR REPLACE INTO class_selections VALUES (?,?,?)', (job_id, json.dumps(values), revision + 1))
        return jsonify(public_selection(job_id, manifest))

    @app.get('/api/jobs/<job_id>/selection/images/<filename>')
    @json_errors
    def selection_image(job_id, filename):
        manifest = manifest_for(job_id)
        if manifest is None or filename not in {item['image'] for item in manifest['classes']}:
            abort(404)
        directory = (store.directory(job_id) / 'selection').resolve()
        path = directory / filename
        if path.resolve().parent != directory or not path.is_file():
            abort(404)
        return send_file(path, mimetype='image/png', max_age=0)


    @app.post('/api/jobs/<job_id>/selection/exports')
    @json_errors
    def start_export(job_id):
        manifest = manifest_for(job_id)
        if manifest is None:
            abort(409, 'Rerun Class Orientation to enable class selection.')
        body = request.get_json(silent=True)
        values = selected(body, manifest)
        if not values:
            abort(400, 'Select at least one class')
        try:
            result = lifecycle.start(dict(g.identity), job_id, body.get('request_id'), values, manifest)
        except ExportLifecycleError as error:
            abort(error.status_code, str(error))
        except ValueError as error:
            abort(400, str(error))
        return jsonify(result), 202

    @app.post('/api/jobs/<job_id>/selection/exports/<export_id>/retry')
    @json_errors
    def retry_export(job_id, export_id):
        if manifest_for(job_id) is None:
            abort(409, 'Rerun Class Orientation to enable class selection.')
        try:
            result = lifecycle.retry(dict(g.identity), job_id, export_id)
        except ExportNotFound:
            abort(404)
        except ExportLifecycleError as error:
            abort(error.status_code, str(error))
        return jsonify(result), 202
