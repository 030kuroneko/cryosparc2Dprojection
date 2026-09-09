"""Durable job records and private per-job credentials for the web launcher."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from time import time
import uuid

TERMINAL = ('completed', 'failed', 'interrupted')
VALIDATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExecutionObservation:
    """Execution evidence supplied by the Local or Slurm adapter."""
    scheduler_state: str | None = None
    detail: str = ''
    process_exit_code: int | None = None
    process_alive: bool = False


def private_json(path, value):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
        json.dump(value, out)


def _remove_credentials(directory):
    try:
        (directory / 'config' / 'cryosparc-tools' / 'auth.json').unlink(missing_ok=True)
    except OSError:
        return False
    return True


def record_worker_completion(directory, exit_code, detail):
    """Persist execution evidence even when job-scoped credential cleanup fails."""
    directory = Path(directory)
    _remove_credentials(directory)
    temporary = directory / 'result.tmp'
    temporary.write_text(json.dumps({'exit_code': exit_code, 'detail': detail}), encoding='utf-8')
    temporary.replace(directory / 'result.json')


class JobStore:
    def __init__(self, config, *, clock=time):
        self.config = config
        self.clock = clock
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
            if 'cleanup_pending' not in columns:
                db.execute('ALTER TABLE jobs ADD COLUMN cleanup_pending INTEGER NOT NULL DEFAULT 0')
                db.execute('UPDATE jobs SET cleanup_pending=1 WHERE state IN (?,?,?)', TERMINAL)
            for name in ('cleanup_attempts', 'cleanup_after'):
                if name not in columns:
                    db.execute(f'ALTER TABLE jobs ADD COLUMN {name} NUMERIC NOT NULL DEFAULT 0')
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
                'cleanup_pending': bool(row['cleanup_pending']),
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
        from cryosparc_2d_projection.workflow_config import WORKFLOWS, build_arguments, default_values
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
            active = db.execute('SELECT owner FROM jobs WHERE state NOT IN (?,?,?)', TERMINAL).fetchall()
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

    def active_jobs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT id,scheduler_id FROM jobs WHERE state NOT IN (?,?,?,?) ORDER BY created',
                ('queued', *TERMINAL))]

    def claim_next(self):
        """Durably claim one queued job only when no execution is outstanding."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM jobs WHERE state NOT IN (?,?,?,?) LIMIT 1',
                          ('queued', *TERMINAL)).fetchone():
                return None
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET state='submitting' WHERE id=?", (row['id'],))
        profile = json.loads(row['execution_json']) if row['execution_json'] else self.profiles.get(row['profile'])
        if profile is None:
            self.update(row['id'], 'failed', 'Execution profile was removed. Submit again with a current profile.')
            return None
        return {'id': row['id'], 'directory': self.directory(row['id']), 'profile': profile}

    def _complete_from_worker(self, job_id):
        marker = self.directory(job_id) / 'result.json'
        try:
            result = json.loads(marker.read_text())
        except FileNotFoundError:
            return False
        else:
            self.update(job_id, 'completed' if result['exit_code'] == 0 else 'failed',
                        result.get('detail', ''))
            return True

    def reconcile(self, job_id, observe):
        """Resolve execution evidence; the completion record takes precedence."""
        if self._complete_from_worker(job_id):
            return True
        try:
            observation = observe()
        except (OSError, subprocess.SubprocessError):
            if self._complete_from_worker(job_id):
                return True
            self.update(job_id, 'unknown', 'Slurm status unavailable; will retry without resubmitting.')
            return False
        # Scheduler/process inspection can take time; completion may arrive during it.
        if self._complete_from_worker(job_id):
            return True
        if observation.scheduler_state is not None:
            state, detail = observation.scheduler_state, observation.detail
            if state == 'completed':
                state, detail = 'failed', 'Slurm ended without a workflow completion record. Check bootstrap.log.'
            self.update(job_id, state, detail)
            return state in TERMINAL
        if observation.process_exit_code is not None:
            self.update(job_id, 'failed', f'Worker exited ({observation.process_exit_code}) without a completion record. Check bootstrap.log.')
            return True
        if not observation.process_alive:
            self.update(job_id, 'unknown', 'Service restarted during execution/submission. Waiting for completion; administrator may need to reconcile.')
        return False

    def update(self, job_id, state, detail='', scheduler_id=None):
        with self.connect() as db:
            changed = db.execute('UPDATE jobs SET state=?,detail=?,scheduler_id=COALESCE(?,scheduler_id) '
                                 'WHERE id=? AND state NOT IN (?,?,?)',
                                 (state, detail, scheduler_id, job_id, *TERMINAL)).rowcount
            if not changed:
                return
            if state in TERMINAL:
                db.execute('UPDATE jobs SET cleanup_pending=1 WHERE id=?', (job_id,))
        if state in TERMINAL:
            self._cleanup(job_id)

    def _cleanup(self, job_id):
        if not _remove_credentials(self.directory(job_id)):
            with self.connect() as db:
                attempts = db.execute('SELECT cleanup_attempts FROM jobs WHERE id=?', (job_id,)).fetchone()[0]
                delay = min(300, 5 * 2 ** min(attempts, 6))
                db.execute('UPDATE jobs SET cleanup_attempts=cleanup_attempts+1,cleanup_after=? WHERE id=?',
                           (self.clock() + delay, job_id))
            return
        with self.connect() as db:
            db.execute('UPDATE jobs SET cleanup_pending=0 WHERE id=?', (job_id,))

    def retry_cleanup(self):
        with self.connect() as db:
            jobs = db.execute('SELECT id FROM jobs WHERE cleanup_pending=1 AND cleanup_after<=?',
                              (self.clock(),)).fetchall()
        for job in jobs:
            self._cleanup(job['id'])

    def log(self, owner, job_id):
        if not self.get(owner, job_id):
            return None
        path = self.directory(job_id) / 'output.log'
        if not path.exists():
            return ''
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size - 65536))
            return source.read(65536).decode('utf-8', errors='replace')
