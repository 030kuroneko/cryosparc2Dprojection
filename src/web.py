"""Private multi-user HTTP launcher. Run through cryosparc2d, not flask run."""
import argparse
from collections import defaultdict, deque
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

from flask import Flask, g, jsonify, request, send_from_directory

from cryosparc_2d_projection.workflow_config import WORKFLOWS, actions, default_values, validate_url
from cryosparc_2d_projection.workflow_fields import BASIC, LABELS, TITLES, DESCRIPTIONS, FIELD_HELP, SYMMETRY_HELP, PLACEHOLDERS, HELP_SOURCES
from cryosparc_2d_projection.web_jobs import JobStore

WEB_LABELS = dict(LABELS, render_grid_size='Surface sampling grid size',
                  render_size='Camera View Render size (px)', surface_level='Surface Level',
                  roll_coarse_step='Coarse in-plane step (°)', roll_refine_step='Fine in-plane step (°)',
                  tilt_coarse_step='Coarse tilt step (°)', tilt_refine_step='Fine tilt step (°)',
                  axis_cone_degrees='Near-Axis cone (°)')


def cryosparc_login(url, email, password):
    """Use the SDK's public login API; never save the password or global auth."""
    from cryosparc import __version__
    from cryosparc.api import APIClient
    from cryosparc.constants import API_SUFFIX
    # The web proxy preserves bearer auth only for the official Tools user agent.
    client = APIClient(url + API_SUFFIX, timeout=30,
                       headers={'User-Agent': f'cryosparc-tools/{__version__}'})
    token = client.login(grant_type='password', username=email,
                         password=sha256(password.encode()).hexdigest())
    # APIClient may return plain JSON when the server schema is not registered.
    access_token = token.get('access_token') if isinstance(token, dict) else token.access_token
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValueError('CryoSPARC returned an invalid authentication token')
    client(auth=access_token)
    user = client.users.me()
    owner = user.get('_id', user.get('id')) if isinstance(user, dict) else user.id
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError('CryoSPARC returned an invalid user identity')
    return {'owner': owner, 'email': email, 'token': access_token}


def create_app(config, *, authenticate=cryosparc_login, start_dispatcher=False):
    config = dict(config)
    config['cryosparc_url'] = validate_url(config['cryosparc_url'].rstrip('/'))
    public = urlsplit(validate_url(config['public_url'].rstrip('/')))
    if public.hostname in ('0.0.0.0', '::'):
        raise ValueError('public_url must use the server IP or hostname, not a wildcard bind address')
    if public.path not in ('', '/'):
        raise ValueError('public_url must be an origin without a path')
    lan_http = public.scheme == 'http' and config.get('host') == '0.0.0.0'
    if lan_http and public.hostname in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Set --public-url to the actual server IP or hostname for LAN access')
    if public.scheme != 'https' and not (config.get('allow_http', False) or lan_http):
        raise ValueError('HTTPS public_url is required; allow_http is for loopback development only')
    if config.get('allow_http') and not lan_http and public.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('allow_http is restricted to loopback development')
    origin = f'{public.scheme}://{public.netloc}'
    trusted_hosts = [public.hostname]
    allowed_origins = {origin}
    loopback_http = public.scheme == 'http' and config.get('allow_http', False) and not lan_http
    if loopback_http:
        trusted_hosts = list(dict.fromkeys([public.hostname, 'localhost', '127.0.0.1']))
        port = f':{public.port}' if public.port else ''
        allowed_origins.update(f'http://{host}{port}' for host in ('localhost', '127.0.0.1'))
    app = Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=65536, TRUSTED_HOSTS=trusted_hosts)
    store = JobStore(config)
    admin_token = config.get('admin_token')
    if not admin_token:
        token_path = store.state_root / 'admin-token'
        try:
            with open(os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
                out.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
        if token_path.stat().st_mode & 0o077 or token_path.stat().st_uid != os.getuid():
            raise ValueError('admin-token must be private (chmod 600) and owned by the service account')
        admin_token = token_path.read_text().strip()
        if not admin_token:
            raise ValueError('admin-token is empty')
    from cryosparc_2d_projection.web_execution import Dispatcher, validate_profiles
    validate_profiles(store.profiles, defer_slurm=True)
    if start_dispatcher:
        dispatcher = Dispatcher(store)
        dispatcher.start()
        app.extensions['dispatcher'] = dispatcher
    sessions, attempts = {}, defaultdict(deque)
    lock = threading.RLock()
    secure = public.scheme == 'https'

    def new_session(identity=None):
        with lock:
            now = time.time()
            for sid in list(sessions):
                if sessions[sid]['expires'] <= now:
                    del sessions[sid]
            if len(sessions) >= 2048:
                raise ValueError('Too many sessions; try again later')
            sid = secrets.token_urlsafe(32)
            sessions[sid] = dict(identity or {}, csrf=secrets.token_urlsafe(32),
                                 expires=now + (28800 if identity else 1800))
            g.new_sid = sid
            g.identity = sessions[sid]
            return sessions[sid]

    @app.before_request
    def protect():
        with lock:
            identity = sessions.get(request.cookies.get('projection_session'))
            g.identity = identity if identity and identity['expires'] > time.time() else None
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            if (request.headers.get('Origin') not in allowed_origins or
                    (loopback_http and request.headers.get('Origin') != request.host_url.rstrip('/')) or
                    not g.identity or
                    not secrets.compare_digest(request.headers.get('X-CSRF-Token', ''),
                                               g.identity['csrf'])):
                return jsonify(error='Session expired or request verification failed. Reload and sign in.'), 403
        if request.path.startswith('/api/') and request.path not in ('/api/session', '/api/login'):
            if not g.identity or 'owner' not in g.identity:
                return jsonify(error='Sign in to continue.'), 401
        if request.path.startswith('/api/admin/') and request.path != '/api/admin/unlock':
            if not g.identity.get('admin'):
                return jsonify(error='Administrator unlock required.'), 403

    @app.after_request
    def response_headers(response):
        response.headers.update({
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Referrer-Policy': 'no-referrer',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        })
        if secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000'
        if getattr(g, 'new_sid', None):
            response.set_cookie('projection_session', g.new_sid, httponly=True,
                                secure=secure, samesite='Strict', max_age=28800, path='/')
        return response

    @app.get('/api/session')
    def session():
        identity = g.identity or new_session()
        return jsonify(csrf=identity['csrf'], email=identity.get('email'),
                       cryosparc_url=config['cryosparc_url'])

    @app.post('/api/login')
    def login():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Email and password are required.'), 400
        email, password = body.get('email'), body.get('password')
        if not isinstance(email, str) or not isinstance(password, str) or not email.strip() or not password:
            return jsonify(error='Email and password are required.'), 400
        # Bounded, per-IP throttling. A proxy must not supply untrusted forwarding headers.
        with lock:
            now = time.time()
            for key in list(attempts):
                if not attempts[key] or attempts[key][-1] < now - 300:
                    del attempts[key]
            recent = attempts[request.remote_addr]
            while recent and recent[0] < now - 300:
                recent.popleft()
            if len(recent) >= 10 or len(attempts) > 4096:
                return jsonify(error='Too many sign-in attempts. Try again in five minutes.'), 429
            recent.append(now)
        try:
            identity = authenticate(config['cryosparc_url'], email.strip(), password)
        except Exception:
            return jsonify(error='Sign-in failed. Check your credentials and server availability.'), 401
        with lock:
            sessions.pop(request.cookies.get('projection_session'), None)
            identity = new_session(identity)
        return jsonify(csrf=identity['csrf'], email=identity['email'])

    @app.post('/api/logout')
    def logout():
        with lock:
            sessions.pop(request.cookies.get('projection_session'), None)
        identity = new_session()
        return jsonify(csrf=identity['csrf'])

    @app.get('/api/jobs')
    def jobs():
        return jsonify(jobs=store.list(g.identity['owner']))

    @app.get('/api/request-id')
    def request_id():
        # LAN HTTP browsers may not expose crypto.randomUUID().
        return jsonify(request_id=str(uuid.uuid4()))

    @app.post('/api/admin/unlock')
    def unlock_admin():
        body = request.get_json(silent=True)
        token = body.get('token') if isinstance(body, dict) else None
        if not isinstance(token, str) or not secrets.compare_digest(token.encode(), admin_token.encode()):
            return jsonify(error='Invalid administrator key.'), 403
        with lock:
            g.identity['admin'] = True
        return jsonify(ok=True)

    @app.get('/api/admin/slurm')
    def slurm_settings():
        profile = store.profiles.get('slurm', {})
        return jsonify(settings=dict(work_dir=profile.get('work_dir', ''),
            python=profile.get('python', sys.executable), partition=profile.get('partition', ''),
            account=profile.get('account', ''), qos=profile.get('qos', ''),
            cpus=profile.get('cpus', 4), memory_mb=profile.get('memory_mb', 16384),
            time_minutes=profile.get('time_minutes', 120), shared_confirmed=False))

    @app.post('/api/admin/slurm')
    def configure_slurm():
        body = request.get_json(silent=True)
        expected = {'work_dir', 'python', 'partition', 'account', 'qos', 'cpus',
                    'memory_mb', 'time_minutes', 'shared_confirmed'}
        try:
            if not isinstance(body, dict) or set(body) != expected:
                raise ValueError('Provide the complete Slurm settings form.')
            if body['shared_confirmed'] is not True:
                raise ValueError('Confirm that this directory and Python are accessible at the same paths on compute nodes.')
            for key in ('work_dir', 'python', 'partition', 'account', 'qos'):
                if not isinstance(body[key], str) or len(body[key]) > 2048 or '\x00' in body[key]:
                    raise ValueError('Invalid setting: ' + key)
            directory, executable = Path(body['work_dir']), Path(body['python'])
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError('Choose an existing absolute shared directory, separate from CryoSPARC project folders.')
            if directory.stat().st_mode & 0o077 or directory.stat().st_uid != os.getuid():
                raise ValueError('Shared directory must be private (chmod 700) and owned by the service account.')
            if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError('Python must be an absolute executable path, also available on compute nodes.')
            profile = {k: v for k, v in body.items() if k != 'shared_confirmed' and v != ''}
            profile.update(backend='slurm', label='Slurm', work_dir=str(directory.resolve()))
            validate_profiles({'slurm': profile})
            missing = [name for name in ('sbatch', 'squeue', 'sacct') if not shutil.which(name)]
            if missing:
                raise ValueError('Slurm is not available on this server PATH: ' + ', '.join(missing))
            store.save_slurm(profile)
        except (OSError, ValueError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(ok=True)

    @app.post('/api/jobs')
    def submit():
        try:
            return jsonify(store.submit(g.identity, request.get_json(silent=True))), 201
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.get('/api/jobs/<job_id>')
    def job(job_id):
        item = store.get(g.identity['owner'], job_id)
        return (jsonify(item), 200) if item else (jsonify(error='Job not found'), 404)

    @app.get('/api/jobs/<job_id>/log')
    def log(job_id):
        content = store.log(g.identity['owner'], job_id)
        return (jsonify(log=content), 200) if content is not None else (jsonify(error='Job not found'), 404)

    @app.get('/api/schema')
    def schema():
        workflows = {}
        for name in WORKFLOWS:
            defaults = default_values(name)
            fields = []
            for action in actions(name):
                key = action.dest
                if key == 'url' or key.endswith('_output'):
                    continue
                group = ('connection' if key in ('project', 'workspace') else
                         'basic' if key in BASIC[name] else
                         'rendering' if key.startswith(('render_', 'surface_', 'comparison_', 'preview_', 'auto_crop')) or key == 'axis_roll' else
                         'search')
                label = key.replace('_', ' ').replace('resolution A', 'resolution (Å)')
                hint, help_text = SYMMETRY_HELP[name] if key == 'symmetry' else FIELD_HELP[key]
                fields.append({'key': key, 'label': WEB_LABELS.get(key, label[0].upper() + label[1:]),
                               'hint': hint, 'help': help_text, 'placeholder': PLACEHOLDERS.get(key, ''),
                               'help_url': HELP_SOURCES.get(key, ''), 'required': action.required,
                               'default': defaults[key], 'group': group,
                               'type': 'boolean' if isinstance(action, argparse._StoreTrueAction) else 'text',
                               'choices': list(action.choices) if action.choices else []})
            workflows[name] = {'title': TITLES[name], 'description': DESCRIPTIONS[name], 'fields': fields}
        profiles = [{'id': key, 'label': value.get('label', key), 'backend': value['backend']}
                    for key, value in store.profiles.items()]
        return jsonify(workflows=workflows, profiles=profiles)

    @app.get('/')
    def index():
        return send_from_directory(Path(__file__).parent / 'web_assets', 'index.html')

    @app.get('/assets/<name>')
    def asset(name):
        if name not in ('app.js', 'app.css', 'theme.js', 'help.js'):
            return jsonify(error='Not found'), 404
        return send_from_directory(Path(__file__).parent / 'web_assets', name)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog='cryosparc2d', description='Serve the multi-user 2D Projection launcher. Configure Slurm in the web UI.')
    parser.add_argument('--config', help='Optional administrator JSON; flags override its values')
    parser.add_argument('--url', dest='cryosparc_url', help='CryoSPARC server URL')
    parser.add_argument('--host', help='Bind address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, help='Listen port (default: 40000)')
    parser.add_argument('--public-url', help='User-facing origin; HTTP LAN access requires --host 0.0.0.0 and the actual server IP/hostname')
    parser.add_argument('--data-dir', help='Private local state directory (default: ~/.local/state/cryosparc2d)')
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError('Configuration must be a JSON object')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    config.update({key: value for key, value in vars(args).items() if key != 'config' and value is not None})
    if args.public_url:
        config['allow_http'] = False
    config.setdefault('host', '127.0.0.1')
    config.setdefault('port', 40000)
    config.setdefault('data_dir', str(Path.home() / '.local/state/cryosparc2d'))
    if not config.get('cryosparc_url'):
        parser.error('--url is required unless cryosparc_url is set in --config')
    if type(config['port']) is not int or not 1 <= config['port'] <= 65535:
        parser.error('--port must be between 1 and 65535')
    if 'public_url' not in config:
        if config['host'] == '0.0.0.0':
            parser.error('--host 0.0.0.0 requires --public-url http://SERVER-IP:40000 (or your HTTPS origin)')
        config['public_url'] = f"http://127.0.0.1:{config['port']}"
        config['allow_http'] = True
    if config.get('allow_http') and config.get('host', '127.0.0.1') not in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
        parser.error('HTTP development mode must bind only to loopback')
    from waitress import serve
    try:
        app = create_app(config, start_dispatcher=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print('2D Projection web service: ' + config['public_url'], flush=True)
    if config['host'] == '0.0.0.0' and urlsplit(config['public_url']).scheme == 'http':
        print('WARNING: LAN HTTP is unencrypted. Restrict access to your trusted lab network/VPN.', flush=True)
    print('Slurm administrator key file: ' + str(Path(config['data_dir']).resolve() / 'admin-token'), flush=True)
    try:
        serve(app, host=config['host'], port=config['port'], threads=8,
              max_request_body_size=65536, channel_timeout=60)
    finally:
        app.extensions['dispatcher'].close()


if __name__ == '__main__':
    main()
