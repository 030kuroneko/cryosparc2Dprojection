"""Owner-scoped saved class selections and durable external-job exports."""
import json
import math
import re
import threading
import uuid
from functools import wraps
from werkzeug.exceptions import HTTPException
from flask import abort, g, jsonify, request, send_file


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
    with store.connect() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS class_exports (
            id TEXT PRIMARY KEY, job_id TEXT NOT NULL, request_id TEXT NOT NULL,
            selected TEXT NOT NULL, manifest TEXT NOT NULL, state TEXT NOT NULL,
            job_uid TEXT, detail TEXT NOT NULL DEFAULT '', UNIQUE(job_id,request_id))''')
        db.execute("UPDATE class_exports SET state='unknown', detail='Service restarted; retry reconciles the recorded job. Without a job ID, administrator reconciliation is required.' WHERE state IN ('creating','publishing')")
        db.execute("UPDATE class_exports SET state='failed', detail='Service restarted before remote creation; retry this export.' WHERE state='preparing'")
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
        return dict(available=True, classes=manifest['classes'], selected_class_numbers=json.loads(row['selected']) if row else [], revision=row['revision'] if row else 0, exports=export_list(job_id))

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


    def public_export(row):
        return dict(id=row['id'], state=row['state'], job_uid=row['job_uid'], detail=row['detail'],
                    selected_class_numbers=json.loads(row['selected']))

    def export_list(job_id):
        with store.connect() as db:
            return [public_export(row) for row in db.execute('SELECT * FROM class_exports WHERE job_id=? ORDER BY rowid DESC', (job_id,))]

    def execute_export(export_id, identity):
        try:
            from cryosparc_2d_projection.class_selection_export import export_class_selection
            with store.connect() as db:
                row = db.execute('SELECT * FROM class_exports WHERE id=?', (export_id,)).fetchone()
            manifest = json.loads(row['manifest'])
            factory = app.config.get('SELECTION_PROJECT_FACTORY')
            project = factory(identity, manifest['project_uid']) if factory else _project(store.config['cryosparc_url'], identity, manifest['project_uid'])
            def created(uid):
                if not isinstance(uid, str) or not re.fullmatch(r'J\d+', uid):
                    raise ValueError('Invalid remote job identity')
                with store.connect() as db:
                    db.execute("UPDATE class_exports SET job_uid=?,state='publishing' WHERE id=?", (uid, export_id))
            def creating():
                with store.connect() as db:
                    db.execute("UPDATE class_exports SET state='creating' WHERE id=?", (export_id,))
            result = export_class_selection(project, manifest['workspace_uid'], manifest,
                                            json.loads(row['selected']), job_uid=row['job_uid'], on_created=created, on_creating=creating)
            with store.connect() as db:
                db.execute("UPDATE class_exports SET state='completed',job_uid=?,detail='' WHERE id=?", (result['job_uid'], export_id))
        except Exception:
            with store.connect() as db:
                row = db.execute('SELECT job_uid,state FROM class_exports WHERE id=?', (export_id,)).fetchone()
                uncertain = row['state'] == 'creating' and not row['job_uid']
                detail = ('CryoSPARC creation outcome is unknown. Administrator reconciliation is required before another export; do not create a replacement job.' if uncertain else
                          'Export could not finish. Sign in again if needed, then retry this export.')
                db.execute('UPDATE class_exports SET state=?,detail=? WHERE id=?', ('unknown' if uncertain else 'failed', detail, export_id))

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
            request_id = str(uuid.UUID(body.get('request_id')))
        except (ValueError, TypeError, AttributeError):
            abort(400, 'Invalid request ID')
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT * FROM class_exports WHERE job_id=? AND request_id=?', (job_id, request_id)).fetchone()
            if prior:
                if json.loads(prior['selected']) != values:
                    abort(409, 'Request ID already used with another selection')
                return jsonify(public_export(prior)), 202
            if db.execute("SELECT 1 FROM class_exports WHERE job_id=? AND state='unknown' AND job_uid IS NULL", (job_id,)).fetchone():
                abort(409, 'Resolve the unknown export with an administrator before creating another job.')
            if db.execute("SELECT COUNT(*) FROM class_exports WHERE state IN ('preparing','creating','publishing')").fetchone()[0] >= 4:
                abort(429, 'Export capacity reached; try again later')
            export_id = uuid.uuid4().hex
            db.execute('INSERT INTO class_exports (id,job_id,request_id,selected,manifest,state) VALUES (?,?,?,?,?,?)',
                       (export_id, job_id, request_id, json.dumps(values), json.dumps(manifest), 'preparing'))
            row = db.execute('SELECT * FROM class_exports WHERE id=?', (export_id,)).fetchone()
        threading.Thread(target=execute_export, args=(export_id, dict(g.identity)), daemon=True).start()
        return jsonify(public_export(row)), 202

    @app.post('/api/jobs/<job_id>/selection/exports/<export_id>/retry')
    @json_errors
    def retry_export(job_id, export_id):
        if manifest_for(job_id) is None:
            abort(409, 'Rerun Class Orientation to enable class selection.')
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM class_exports WHERE id=? AND job_id=?', (export_id, job_id)).fetchone()
            if not row:
                abort(404)
            if row['state'] in ('preparing', 'creating', 'publishing', 'completed'):
                return jsonify(public_export(row)), 202
            if row['state'] == 'unknown' and not row['job_uid']:
                abort(409, 'Administrator must reconcile the unknown CryoSPARC job before retry.')
            if db.execute("SELECT COUNT(*) FROM class_exports WHERE state IN ('preparing','creating','publishing')").fetchone()[0] >= 4:
                abort(429, 'Export capacity reached; try again later.')
            db.execute("UPDATE class_exports SET state='preparing',detail='' WHERE id=?", (export_id,))
            row = db.execute('SELECT * FROM class_exports WHERE id=?', (export_id,)).fetchone()
        threading.Thread(target=execute_export, args=(export_id, dict(g.identity)), daemon=True).start()
        return jsonify(public_export(row)), 202
