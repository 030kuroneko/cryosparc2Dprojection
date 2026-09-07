"""Durable job records and private per-job credentials for the web launcher."""
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import uuid

from cryosparc_2d_projection.gui_model import WORKFLOWS, build_arguments, default_values

TERMINAL = ('completed', 'failed', 'interrupted')
VALIDATION_LOCK = threading.Lock()


def private_json(path, value):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
        json.dump(value, out)


class JobStore:
    def __init__(self, config):
        self.config = config
        self.state_root = Path(config['data_dir']).resolve()
        self.root = Path(config.get('work_dir', config['data_dir'])).resolve()
        for directory in (self.state_root, self.root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if directory.stat().st_mode & 0o077 or directory.stat().st_uid != os.getuid():
                raise ValueError('Data and work directories must be private (chmod 700), owned by the service account')
        self.database = self.state_root / 'jobs.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            for key, value in [('instance', config['cryosparc_url']), ('work_dir', str(self.root))]:
                db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key, value))
                if db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()[0] != value:
                    raise ValueError(f'This data directory belongs to a different {key}; use a new directory')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_id TEXT NOT NULL,
                workflow TEXT NOT NULL, profile TEXT NOT NULL, values_json TEXT NOT NULL,
                state TEXT NOT NULL, created TEXT NOT NULL, scheduler_id TEXT,
                detail TEXT NOT NULL DEFAULT '', UNIQUE(owner, request_id))''')
            columns = {r[1] for r in db.execute('PRAGMA table_info(jobs)')}
            for name in ('directory', 'execution_json'):
                if name not in columns:
                    db.execute(f'ALTER TABLE jobs ADD COLUMN {name} TEXT')
        self.database.chmod(0o600)

    @property
    def profiles(self):
        profiles = dict(self.config.get('profiles', {'local': {'backend': 'local', 'label': 'Local · sequential'}}))
        with self.connect() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='slurm_profile'").fetchone()
        if row:
            profiles['slurm'] = json.loads(row[0])
        return profiles

    def save_slurm(self, profile):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('slurm_profile', ?)", (json.dumps(profile),))

    def directory(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT directory FROM jobs WHERE id=?', (job_id,)).fetchone()
        return Path(row['directory']) if row and row['directory'] else self.root / job_id

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def public(self, row):
        if row is None:
            return None
        return {**{k: row[k] for k in ('id', 'workflow', 'profile', 'state', 'created', 'scheduler_id', 'detail')},
                'values': json.loads(row['values_json'])}

    def list(self, owner):
        with self.connect() as db:
            return [self.public(r) for r in db.execute(
                'SELECT * FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 100', (owner,))]

    def get(self, owner, job_id):
        with self.connect() as db:
            return self.public(db.execute('SELECT * FROM jobs WHERE id=? AND owner=?',
                                          (job_id, owner)).fetchone())

    def submit(self, identity, body):
        if not isinstance(body, dict) or set(body) != {'workflow', 'values', 'profile', 'request_id'}:
            raise ValueError('Expected workflow, values, profile and request_id')
        workflow, values, profile = body['workflow'], body['values'], body['profile']
        profiles = self.profiles
        if not isinstance(workflow, str) or workflow not in WORKFLOWS:
            raise ValueError('Unknown workflow')
        if not isinstance(profile, str) or profile not in profiles:
            raise ValueError('Choose an available execution profile')
        from cryosparc_2d_projection.web_execution import validate_profiles
        validate_profiles({profile: profiles[profile]})
        if not isinstance(body['request_id'], str):
            raise ValueError('Invalid request ID')
        try:
            request_id = str(uuid.UUID(body['request_id']))
        except (ValueError, AttributeError):
            raise ValueError('Invalid request ID') from None
        defaults = default_values(workflow)
        if not isinstance(values, dict) or set(values) - (set(defaults) - {'url'}):
            raise ValueError('Unknown parameter. The CryoSPARC server is configured by the administrator.')
        for key, value in values.items():
            if type(value) is not type(defaults[key]) or (isinstance(value, str) and len(value) > 2048):
                raise ValueError(f'Invalid value: {key}')
        validated = dict(defaults, **values, url=self.config['cryosparc_url'])
        # argparse validation captures process-global stderr; serialize that short boundary.
        with VALIDATION_LOCK:
            argv = build_arguments(workflow, validated)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT * FROM jobs WHERE owner=? AND request_id=?',
                               (identity['owner'], request_id)).fetchone()
            if prior:
                if prior['workflow'] != workflow or prior['profile'] != profile or json.loads(prior['values_json']) != validated:
                    raise ValueError('Request ID already used with different settings')
                return self.public(prior)
            active = db.execute("SELECT owner FROM jobs WHERE state NOT IN ('completed','failed','interrupted')").fetchall()
            if len(active) >= 32 or sum(r['owner'] == identity['owner'] for r in active) >= 8:
                raise ValueError('Queue limit reached. Wait for a job to finish.')
            job_id = uuid.uuid4().hex
            execution = dict(profiles[profile])
            execution.setdefault('python', sys.executable)
            directory = Path(execution.get('work_dir', self.root)) / job_id
            directory.mkdir(mode=0o700)
            auth_dir = directory / 'config' / 'cryosparc-tools'
            auth_dir.mkdir(parents=True, mode=0o700)
            (directory / 'config').chmod(0o700)
            expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
            private_json(auth_dir / 'auth.json', {self.config['cryosparc_url']: {identity['email']: {
                'token': {'access_token': identity['token'], 'token_type': 'bearer'}, 'expires': expires}}})
            private_json(directory / 'request.json', {'workflow': workflow, 'argv': argv,
                                                       'email': identity['email']})
            db.execute('INSERT INTO jobs (id,owner,request_id,workflow,profile,values_json,state,created,directory,execution_json) VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (job_id, identity['owner'], request_id, workflow, profile,
                        json.dumps(validated), 'queued', datetime.now(timezone.utc).isoformat(),
                        str(directory), json.dumps(execution)))
        return self.get(identity['owner'], job_id)

    def update(self, job_id, state, detail='', scheduler_id=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET state=?,detail=?,scheduler_id=COALESCE(?,scheduler_id) WHERE id=?',
                       (state, detail, scheduler_id, job_id))
        if state in TERMINAL:
            (self.directory(job_id) / 'config' / 'cryosparc-tools' / 'auth.json').unlink(missing_ok=True)

    def log(self, owner, job_id):
        if not self.get(owner, job_id):
            return None
        path = self.directory(job_id) / 'output.log'
        if not path.exists():
            return ''
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size - 65536))
            return source.read(65536).decode('utf-8', errors='replace')
