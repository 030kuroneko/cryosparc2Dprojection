"""Filesystem, package-manager and systemd setup for the optional Web service."""
from dataclasses import dataclass
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import sqlite3
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request


ACCOUNT = 'cryosparc2d'
UNIT = 'cryosparc2d.service'


@dataclass(frozen=True)
class Layout:
    root: Path = Path('/')

    @property
    def application(self):
        return self.root / 'opt/cryosparc2d'

    @property
    def python(self):
        return self.application / 'venv/bin/python'

    @property
    def config(self):
        return self.root / 'etc/cryosparc2d/web.json'

    @property
    def unit(self):
        return self.root / 'etc/systemd/system' / UNIT

    @property
    def data(self):
        return self.root / 'var/lib/cryosparc2d'


@contextmanager
def setup_lock(layout):
    import fcntl
    layout.config.parent.mkdir(parents=True, exist_ok=True)
    with open(os.open(layout.config.parent / '.setup.lock', os.O_WRONLY | os.O_CREAT, 0o600), 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another installation or configuration is already in progress.') from None
        yield


def atomic_write(path, content, mode=0o640, *, gid=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as out:
            os.fchmod(out.fileno(), mode)
            if gid is not None:
                os.fchown(out.fileno(), -1, gid)
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ask_config(layout, old, prompt):
    config = json.loads(json.dumps(old))

    def ask(label, default=''):
        answer = prompt(f'{label} [{default}]: ').strip()
        return default if not answer else '' if answer == '-' else answer

    print('Press Enter to keep the displayed value; use - to clear an optional value.')
    config['cryosparc_url'] = ask('CryoSPARC URL', old.get('cryosparc_url', '')).rstrip('/')
    config['host'] = ask('Listen address', old.get('host', '0.0.0.0'))
    config['port'] = int(ask('Listen port', str(old.get('port', 40000))))
    public = ask('Public URL (optional; HTTPS reverse proxy or fixed origin)', old.get('public_url', ''))
    config.pop('allow_http', None)
    if public:
        config['public_url'] = public.rstrip('/')
        parsed = urlsplit(public)
        if (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')
                and config['host'] in ('localhost', '127.0.0.1', '::1')):
            config['allow_http'] = True
    else:
        config.pop('public_url', None)
    config['data_dir'] = ask('Private state directory', old.get('data_dir', str(layout.data)))
    config['work_dir'] = ask('Private work directory', old.get('work_dir', config['data_dir']))
    profiles = config.setdefault('profiles', {})
    local = profiles.setdefault('local', {'backend': 'local', 'label': 'Local · sequential'})
    local['python'] = ask('Local computation Python', local.get('python', str(layout.python)))
    return config


def validate_settings(config):
    def url(value):
        address = urlsplit(value)
        _ = address.port
        if (address.scheme not in ('http', 'https') or not address.hostname or
                address.username is not None or address.password is not None or
                address.query or address.fragment or any(c.isspace() for c in value)):
            raise ValueError('Use an HTTP(S) URL without credentials, query or fragment.')
        return address

    url(config['cryosparc_url'])
    if config.get('public_url'):
        public = url(config['public_url'])
        if public.path not in ('', '/') or public.hostname in ('0.0.0.0', '::'):
            raise ValueError('Public URL must be an origin using the real server IP or hostname.')
    if type(config['port']) is not int or not 1024 <= config['port'] <= 65535:
        raise ValueError('Use an unprivileged listen port between 1024 and 65535.')
    if not config['host'] or any(c.isspace() for c in config['host']):
        raise ValueError('Enter a valid listen address.')
    for name in ('data_dir', 'work_dir'):
        if not Path(config[name]).is_absolute():
            raise ValueError(f'{name} must be an absolute path.')
    if not Path(config['profiles']['local']['python']).is_absolute():
        raise ValueError('Local computation Python must be an absolute path.')


def check_connection(config, opener):
    try:
        with opener(config['cryosparc_url'], timeout=10):
            pass
    except HTTPError as error:
        if error.code not in (401, 403):
            raise ValueError(f'CryoSPARC URL returned HTTP {error.code}. Check the URL.') from error
    except (URLError, OSError) as error:
        raise ValueError(f'Cannot reach the CryoSPARC URL: {error}') from error
    print('CryoSPARC HTTP endpoint is reachable. Each user signs in through the Web GUI.')


def private_directory(directory, account):
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, mode=0o700)
        os.chown(directory, account.pw_uid, account.pw_gid)
    stat = directory.stat()
    if directory.is_symlink() or stat.st_uid != account.pw_uid or stat.st_mode & 0o077:
        raise ValueError(f'{directory} must be private (0700) and owned by {ACCOUNT}; '
                         'existing directory ownership is not changed automatically.')


def validate_runtime(layout, config_path, config, run):
    # Exercise the same configuration and store validation as the service, as
    # its actual account, without starting a dispatcher or a scientific job.
    script = (
        'import json, os, subprocess, sys; '
        'from cryosparc_2d_projection.web import create_app; '
        'c=json.load(open(sys.argv[1])); '
        'p=c["profiles"]["local"]["python"]; '
        'assert os.path.isfile(p) and os.access(p, os.X_OK), "Computation Python is not executable"; '
        'r=subprocess.run([p, "-c", '
        '"import cryosparc_2d_projection.cli; import cryosparc_2d_projection.axis_cli; print(\'cryosparc2d-ready\')"], '
        'check=True, capture_output=True, text=True, timeout=30); '
        'assert r.stdout.strip() == "cryosparc2d-ready", "Computation Python cannot load the scientific CLIs"; '
        'create_app(c)'
    )
    result = run([shutil.which('runuser'), '-u', ACCOUNT, '--', str(layout.python), '-c',
                  script, str(config_path)], check=False, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise ValueError('Settings validation failed: ' + (detail[-1] if detail else 'check service account access.'))


def wait_ready(config, opener):
    host = config['host']
    if host == '0.0.0.0':
        host = '127.0.0.1'
    elif host == '::':
        host = '::1'
    if ':' in host:
        host = '[' + host + ']'
    address = f'http://{host}:{config["port"]}'
    headers = {}
    if config.get('public_url'):
        headers['Host'] = urlsplit(config['public_url']).netloc
    deadline = time.monotonic() + 30
    while True:
        try:
            with opener(Request(address + '/api/session', headers=headers), timeout=2) as response:
                body = json.load(response)
            if isinstance(body, dict) and body.get('csrf'):
                return
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise ValueError('Web service did not become ready. Run cryosparc2d-service status '
                             'and journalctl -u cryosparc2d.service.')
        time.sleep(0.25)


def install_controls(layout, run):
    command = layout.root / 'usr/local/bin/cryosparc2d-service'
    target = layout.application / 'venv/bin/cryosparc2d-service'
    command.parent.mkdir(parents=True, exist_ok=True)
    if command.exists() or command.is_symlink():
        if not command.is_symlink() or command.readlink() != target:
            raise ValueError(f'Refusing to replace existing command: {command}')
    else:
        command.symlink_to(target)
    deployer = os.environ.get('SUDO_USER')
    if not deployer or deployer == 'root':
        return
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_-]*', deployer):
        raise ValueError('Unsupported deployer account name for a narrow sudoers rule.')
    pwd.getpwnam(deployer)
    systemctl = shutil.which('systemctl')
    commands = [f'{systemctl} {action} {UNIT}'
                for action in ('start', 'stop', 'restart', 'enable', 'disable')]
    commands.append(f'{systemctl} status --no-pager {UNIT}')
    rule = f'{deployer} ALL=(root) NOPASSWD: ' + ', '.join(commands) + '\n'
    destination = layout.root / 'etc/sudoers.d/cryosparc2d'
    candidate = destination.with_name('.cryosparc2d-candidate')
    atomic_write(candidate, rule, 0o440)
    try:
        run([shutil.which('visudo'), '-cf', str(candidate)], check=True)
        candidate.replace(destination)
    finally:
        candidate.unlink(missing_ok=True)


def install(layout, source, uv, run, prompt, opener):
    marker = layout.application / 'installation.json'
    state = json.loads(marker.read_text()) if marker.exists() else {}
    if layout.unit.exists() and state.get('status') == 'ready':
        print('Web service is already installed. Use cryosparc2d-service configure to change settings.')
        return
    if layout.unit.exists() and state.get('status') != 'installing':
        raise ValueError('An unmanaged cryosparc2d.service already exists; it will not be replaced.')
    run([shutil.which('systemctl'), 'show', '--property=Version', '--value'], check=True,
        capture_output=True, text=True)
    for name in ('useradd', 'runuser', 'visudo'):
        if not shutil.which(name):
            raise ValueError(f'{name} is required for managed Web installation.')
    source = source.resolve()
    for name in ('pyproject.toml', 'uv.lock', 'src'):
        if not (source / name).exists():
            raise ValueError('Managed installation requires the source checkout. '
                             'Run bash install.sh from the repository first.')
    old = json.loads(layout.config.read_text()) if layout.config.exists() else {}
    if old:
        require_idle(old)
    config = ask_config(layout, old, prompt)
    validate_settings(config)
    check_connection(config, opener)
    try:
        account = pwd.getpwnam(ACCOUNT)
    except KeyError:
        run([shutil.which('useradd'), '--system', '--user-group', '--no-create-home',
             '--home-dir', str(layout.data), '--shell', '/usr/sbin/nologin', ACCOUNT], check=True)
        account = pwd.getpwnam(ACCOUNT)
    if account.pw_uid == 0:
        raise ValueError('The service account must not be root.')

    # A root-managed runtime avoids making a shared service depend on the
    # deployer's private home or a mutable personal CLI environment.
    layout.application.mkdir(parents=True, exist_ok=True, mode=0o755)
    atomic_write(marker, json.dumps({'status': 'installing'}) + '\n', 0o644)
    if layout.unit.exists():
        run([shutil.which('systemctl'), 'stop', UNIT], check=True)
        require_idle(old)
    copied_source = layout.application / 'source'
    copied_source.mkdir(exist_ok=True)
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copyfile(source / name, copied_source / name)
    shutil.copytree(source / 'src', copied_source / 'src', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    environment = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(layout.application / 'venv'),
                       UV_PYTHON_INSTALL_DIR=str(layout.application / 'python'),
                       UV_CACHE_DIR=str(layout.application / 'cache'))
    previous_umask = os.umask(0o022)
    try:
        run([uv, 'sync', '--project', str(copied_source), '--locked', '--no-default-groups',
             '--no-editable', '--python', '3.12', '--managed-python', '--extra', 'web'],
            check=True, env=environment)
    finally:
        os.umask(previous_umask)
    for directory in {config['data_dir'], config['work_dir']}:
        private_directory(directory, account)
    layout.config.parent.mkdir(parents=True, exist_ok=True)
    layout.config.parent.chmod(0o750)
    os.chown(layout.config.parent, -1, account.pw_gid)
    atomic_write(layout.config, json.dumps(config, indent=2) + '\n', gid=account.pw_gid)
    validate_runtime(layout, layout.config, config, run)

    unit = f'''[Unit]
Description=CryoSPARC 2D Projection Web GUI
After=network-online.target
Wants=network-online.target

[Service]
User={ACCOUNT}
Group={ACCOUNT}
UMask=0077
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=MPLBACKEND=Agg
ExecStart={layout.application}/venv/bin/cryosparc2d --config {layout.config}
Restart=on-failure
KillMode=process
TimeoutStopSec=75
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
'''
    atomic_write(layout.unit, unit, 0o644)
    install_controls(layout, run)
    systemctl = shutil.which('systemctl')
    run([systemctl, 'daemon-reload'], check=True)
    run([systemctl, 'enable', '--now', UNIT], check=True)
    wait_ready(config, opener)
    atomic_write(marker, json.dumps({'status': 'ready'}) + '\n', 0o644)
    host = config['host'] if config['host'] not in ('0.0.0.0', '::') else 'SERVER-IP'
    print('Web GUI ready: ' + (config.get('public_url') or f'http://{host}:{config["port"]}'))
    print('Use cryosparc2d-service configure to change settings later.')


def require_idle(config):
    """Read existing queue state without creating or migrating a job database."""
    database = Path(config['data_dir']) / 'jobs.sqlite3'
    if not database.exists():
        return
    from cryosparc_2d_projection.web_jobs import TERMINAL
    try:
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
            busy = db.execute('SELECT 1 FROM jobs WHERE state NOT IN (?,?,?) LIMIT 1', TERMINAL).fetchone()
            columns = {row[1] for row in db.execute('PRAGMA table_info(jobs)')}
            cleanup = ('cleanup_pending' in columns and
                       db.execute('SELECT 1 FROM jobs WHERE cleanup_pending=1 LIMIT 1').fetchone())
    except sqlite3.Error as error:
        raise ValueError('Cannot verify existing jobs; the configuration was not changed.') from error
    if busy:
        raise ValueError('There are queued or running jobs. Let them finish before changing '
                         'the CryoSPARC connection, directories or computation environment.')
    if cleanup:
        raise ValueError('Credential cleanup is pending. Start the service and let cleanup '
                         'finish before changing the connection, directories or computation environment.')


def configure(layout, run, prompt, opener):
    old_text = layout.config.read_text()
    old = json.loads(old_text)
    config = ask_config(layout, old, prompt)
    validate_settings(config)
    check_connection(config, opener)
    if config == old:
        print('Settings unchanged.')
        return
    critical = any(config.get(key) != old.get(key)
                   for key in ('cryosparc_url', 'data_dir', 'work_dir', 'profiles'))
    if critical:
        require_idle(old)
        require_idle(config)
    account = pwd.getpwnam(ACCOUNT)
    for directory in {config['data_dir'], config['work_dir']}:
        private_directory(directory, account)
    candidate = layout.config.with_name('.web-candidate.json')
    atomic_write(candidate, json.dumps(config, indent=2) + '\n', gid=account.pw_gid)
    systemctl = shutil.which('systemctl')
    was_active = False
    replaced = False
    try:
        validate_runtime(layout, candidate, config, run)
        was_active = run([systemctl, 'is-active', '--quiet', UNIT], check=False).returncode == 0
        if was_active:
            run([systemctl, 'stop', UNIT], check=True)
        # Recheck after stopping dispatch: a job may have arrived during the
        # prompts or validation. Running workers still finish independently.
        if critical:
            require_idle(old)
            require_idle(config)
        atomic_write(layout.config.with_suffix('.previous.json'), old_text, gid=account.pw_gid)
        candidate.replace(layout.config)
        replaced = True
        if was_active:
            run([systemctl, 'start', UNIT], check=True)
            wait_ready(config, opener)
    except Exception:
        if replaced:
            atomic_write(layout.config, old_text, gid=account.pw_gid)
        if was_active:
            run([systemctl, 'restart', UNIT], check=False)
        raise
    finally:
        candidate.unlink(missing_ok=True)
    print('Settings saved.' + (' Service restarted.' if was_active else ' Service remains stopped.'))
