"""Single dispatcher, isolated processes, and explicit Slurm state reconciliation."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading

def worker_command(profile, directory):
    return [profile.get('python', sys.executable), '-u', '-m',
            'cryosparc_2d_projection.web_worker', str(directory)]


def worker_environment(cpus=1):
    # Never inherit the administrator's CryoSPARC credentials or instance selectors.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('CRYOSPARC_', 'SBATCH_', 'SLURM_'))}
    env.update(MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(cpus)
    return env


def validate_profiles(profiles, *, defer_slurm=False):
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError('At least one execution profile is required')
    for name, profile in profiles.items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', name) or not isinstance(profile, dict):
            raise ValueError('Invalid execution profile')
        if profile.get('backend') not in ('local', 'slurm'):
            raise ValueError('Profile backend must be local or slurm')
        if defer_slurm and profile['backend'] == 'slurm':
            continue
        if not Path(profile.get('python', sys.executable)).is_absolute():
            raise ValueError('Profile python must be an absolute path accessible on execution nodes')
        for key in ('cpus', 'memory_mb', 'time_minutes'):
            value = profile.get(key, 1)
            if type(value) is not int or not 1 <= value <= 1000000:
                raise ValueError(f'Invalid profile resource: {key}')
        for key in ('partition', 'account', 'qos'):
            if key in profile and not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', profile[key]):
                raise ValueError(f'Invalid profile: {key}')


class SlurmBackend:
    def __init__(self, profile, *, run=subprocess.run):
        self.profile, self.run = profile, run

    def submit(self, directory):
        directory = Path(directory)
        script = directory / 'submit.sh'
        content = '#!/bin/sh\numask 077\nexec ' + shlex.join(worker_command(self.profile, directory)) + '\n'
        with open(os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
            out.write(content)
        argv = ['sbatch', '--parsable', '--no-requeue', '--export=NONE',
                '--job-name=projection-' + directory.name, '--chdir=' + str(directory),
                '--output=' + str(directory / 'bootstrap.log'),
                '--error=' + str(directory / 'bootstrap.log'),
                '--nodes=1', '--ntasks=1',
                '--cpus-per-task=' + str(self.profile.get('cpus', 1)),
                '--mem=' + str(self.profile.get('memory_mb', 4096)) + 'M',
                '--time=' + str(self.profile.get('time_minutes', 60))]
        for key in ('partition', 'account', 'qos'):
            if self.profile.get(key):
                argv.append('--' + key + '=' + self.profile[key])
        result = self.run([*argv, str(script)], capture_output=True, text=True,
                          timeout=30, env=worker_environment(), check=True)
        job_id = result.stdout.strip().split(';')[0]
        if not re.fullmatch(r'[1-9][0-9]*', job_id):
            raise RuntimeError('Ambiguous Slurm submission response')
        return job_id

    def status(self, job_id):
        if not re.fullmatch(r'[1-9][0-9]*', job_id):
            raise ValueError('Invalid Slurm job ID')
        try:
            queued = self.run(['squeue', '--noheader', '--jobs=' + job_id, '--format=%T'],
                              capture_output=True, text=True, timeout=15, check=True)
            states = queued.stdout.strip().splitlines()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Completed/evicted job IDs may make squeue fail. Accounting is authoritative.
            states = []
        if states:
            value = states[0].strip()
            return ('pending', '') if value == 'PENDING' else ('running', value)
        completed = self.run(['sacct', '--noheader', '--parsable2', '--allocations',
                              '--jobs=' + job_id, '--format=JobIDRaw,State,ExitCode'],
                             capture_output=True, text=True, timeout=15, check=True)
        for line in completed.stdout.splitlines():
            parts = line.strip().split('|')
            if len(parts) < 3 or parts[0] != job_id:
                continue
            state = parts[1].split()[0].rstrip('+')
            if state == 'COMPLETED' and parts[2] == '0:0':
                return 'completed', ''
            if state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                         'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED') or state == 'COMPLETED':
                return 'failed', f'Slurm {state}; exit {parts[2]}'
            return 'running', state
        return 'unknown', 'Waiting for Slurm accounting; not resubmitting.'


class Dispatcher:
    """One active computation across both backends. Browser sessions are independent."""
    def __init__(self, store):
        self.store = store
        validate_profiles(store.profiles, defer_slurm=True)
        self.stop_event = threading.Event()
        self.processes = {}
        self.thread = None
        self.lock_file = None

    def start(self):
        import fcntl
        self.lock_file = (self.store.root / 'dispatcher.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError('Another dispatcher already owns this data directory') from None
        self.thread = threading.Thread(target=self._loop, daemon=True, name='projection-dispatcher')
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=65)
        if self.lock_file and (not self.thread or not self.thread.is_alive()):
            self.lock_file.close()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:
                # Preserve durable state; never infer failure and duplicate an uncertain submission.
                import logging
                logging.getLogger(__name__).error('Dispatcher check failed; existing jobs will be rechecked.')
            self.stop_event.wait(5)

    def tick(self):
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM jobs WHERE state NOT IN ('queued','completed','failed','interrupted') ORDER BY created").fetchall()
        for row in rows:
            self._refresh(row)
        with self.store.connect() as db:
            count = db.execute("SELECT count(*) FROM jobs WHERE state NOT IN ('queued','completed','failed','interrupted')").fetchone()[0]
            if count:
                return
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
        if not row:
            return
        job_id = row['id']
        profile = json.loads(row['execution_json']) if row['execution_json'] else self.store.profiles.get(row['profile'])
        if profile is None:
            self.store.update(job_id, 'failed', 'Execution profile was removed. Submit again with a current profile.')
            return
        directory = self.store.directory(job_id)
        self.store.update(job_id, 'submitting')
        try:
            if profile['backend'] == 'slurm':
                scheduler_id = SlurmBackend(profile).submit(directory)
                self.store.update(job_id, 'pending', scheduler_id=scheduler_id)
            else:
                # The worker owns scientific logging and its durable completion marker.
                with (directory / 'bootstrap.log').open('ab') as out:
                    process = subprocess.Popen(worker_command(profile, directory),
                                               stdout=out, stderr=out, cwd=directory,
                                               env=worker_environment(profile.get('cpus', 1)), start_new_session=True)
                self.processes[job_id] = process
                self.store.update(job_id, 'running')
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.store.update(job_id, 'failed', 'Submission failed. Check the configured executable and scheduler access.')
        except Exception:
            self.store.update(job_id, 'unknown', 'Submission outcome uncertain. Administrator reconciliation required; not resubmitting.')

    def _refresh(self, row):
        job_id = row['id']
        directory = self.store.directory(job_id)
        marker = directory / 'result.json'
        # A marker proves workflow completion even if the web service was restarted.
        if marker.exists():
            result = json.loads(marker.read_text())
            state = 'completed' if result['exit_code'] == 0 else 'failed'
            self.store.update(job_id, state, result.get('detail', ''))
            process = self.processes.pop(job_id, None)
            if process:
                process.wait(timeout=10)
            return
        if row['scheduler_id']:
            try:
                state, detail = SlurmBackend({}).status(row['scheduler_id'])
                if state == 'completed':
                    # Scheduler success is insufficient without our worker completion marker.
                    state, detail = 'failed', 'Slurm ended without a workflow completion record. Check bootstrap.log.'
                self.store.update(job_id, state, detail)
            except (OSError, subprocess.SubprocessError):
                self.store.update(job_id, 'unknown', 'Slurm status unavailable; will retry without resubmitting.')
        elif job_id in self.processes:
            code = self.processes[job_id].poll()
            if code is not None:
                del self.processes[job_id]
                self.store.update(job_id, 'failed', f'Worker exited ({code}) without a completion record. Check bootstrap.log.')
        else:
            self.store.update(job_id, 'unknown', 'Service restarted during execution/submission. Waiting for completion; administrator may need to reconcile.')
