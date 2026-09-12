"""Administrator HTTP workflows for publishing execution lanes."""
from hashlib import sha256
import os
from pathlib import Path
import secrets
import shutil
import sys

from flask import g, jsonify, request

from cryosparc_2d_projection.web_execution import SlurmBackend, validate_profiles
from cryosparc_2d_projection.slurm_templates import DEFAULT_TEMPLATE, BUILTIN_VARIABLES, validate_variables


SLURM_DEFAULTS = dict(backend='slurm', label='New Slurm lane', work_dir='', python=sys.executable,
                      partition='', account='', qos='', cpus=4, gpus=0, memory_mb=16384,
                      time_minutes=120, max_concurrent=1, enabled=True, template_path='', variables={})


def validate_shared_slurm_profile(name, profile, shared_confirmed):
    """Validate the shared execution contract for both administrator interfaces."""
    if shared_confirmed is not True:
        raise ValueError('Confirm that the directory and Python are accessible on compute nodes.')
    for key in ('work_dir', 'python', 'partition', 'account', 'qos'):
        value = profile.get(key, '')
        if not isinstance(value, str) or len(value) > 2048 or '\x00' in value:
            raise ValueError('Invalid setting: ' + key)
    directory, executable = Path(profile.get('work_dir', '')), Path(profile.get('python', ''))
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError('Choose an existing absolute shared directory, separate from CryoSPARC project folders.')
    if directory.stat().st_mode & 0o077 or directory.stat().st_uid != os.getuid():
        raise ValueError('Shared directory must be private (chmod 700), owned by the service account.')
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError('Python must be an absolute executable path available on compute nodes.')
    profile = {k: v for k, v in profile.items() if k not in ('partition', 'account', 'qos') or v}
    profile['work_dir'] = str(directory.resolve())
    validate_profiles({name: profile})
    missing = [command for command in ('sbatch', 'squeue', 'sacct') if not shutil.which(command)]
    if missing:
        raise ValueError('Slurm is not available on this server PATH: ' + ', '.join(missing))
    return profile


def _validate_settings(name, settings, current, reload_template, template_source=None):
    if not isinstance(settings, dict):
        raise ValueError('Provide lane settings.')
    allowed = set(SLURM_DEFAULTS) | {'id', 'revision', 'shared_confirmed', 'template_text', 'template_sha256', 'lane_id', 'copy_from'}
    if set(settings) - allowed:
        raise ValueError('Unknown lane setting.')
    if current and settings.get('revision') != current['revision']:
        raise ValueError('Lane already exists or changed. Select it and reload its settings.')
    backend = settings.get('backend')
    if current and backend != current['backend']:
        raise ValueError('A lane backend cannot be changed.')
    if backend == 'local':
        if not current:
            raise ValueError('Configure the existing Local lane.')
        profile = dict(current, max_concurrent=settings.get('max_concurrent', 1))
        if profile.get('label') in (None, 'Local · sequential'):
            profile['label'] = 'Local'
    else:
        profile = dict(SLURM_DEFAULTS, **{k: v for k, v in settings.items() if k in SLURM_DEFAULTS})
        profile = validate_shared_slurm_profile(name, profile, settings.get('shared_confirmed'))
        template_current = template_source or current
        template_path = profile['template_path']
        if not isinstance(template_path, str) or len(template_path) > 2048 or any(c in template_path for c in '\x00\r\n'):
            raise ValueError('Invalid template path.')
        if template_path and not Path(template_path).is_absolute():
            raise ValueError('Template path must be absolute on the launcher server.')
        if reload_template and template_path:
            with Path(template_path).open('rb') as source:
                content = source.read(32769)
            if len(content) > 32768:
                raise ValueError('Submission template exceeds 32 KiB.')
            profile['template_text'] = content.decode('utf-8')
        elif template_path != template_current.get('template_path', ''):
            raise ValueError('Reload the external template before applying a new path.')
        else:
            profile['template_text'] = template_current.get('template_text', DEFAULT_TEMPLATE) if template_path else DEFAULT_TEMPLATE
        profile['template_sha256'] = sha256(profile['template_text'].encode()).hexdigest()
        profile['lane_id'] = name
        validate_variables(profile['variables'])
    if not isinstance(profile['label'], str) or not profile['label'].strip() or len(profile['label']) > 100:
        raise ValueError('Provide a lane name of 1 to 100 characters.')
    validate_profiles({name: profile})
    return profile


def register_lane_routes(app, store):
    @app.get('/api/admin/lanes')
    def lanes():
        return jsonify(lanes=[dict(value, id=name) for name, value in store.profiles.items()],
                       defaults=SLURM_DEFAULTS, example_template=DEFAULT_TEMPLATE, builtin_variables=BUILTIN_VARIABLES)

    @app.post('/api/admin/lanes/preview')
    def preview_lane():
        body = request.get_json(silent=True)
        try:
            if not isinstance(body, dict) or set(body) != {'id', 'settings', 'reload_template'}:
                raise ValueError('Provide a lane ID, settings and reload choice.')
            name = body['id']
            if not isinstance(name, str):
                raise ValueError('Invalid lane ID.')
            if type(body['reload_template']) is not bool:
                raise ValueError('Invalid template reload choice.')
            profiles = store.profiles
            current = profiles.get(name, {})
            settings = body['settings']
            source = None
            if isinstance(settings, dict) and settings.get('copy_from'):
                source_id = settings['copy_from']
                if not isinstance(source_id, str):
                    raise ValueError('Invalid source lane.')
                source = profiles.get(source_id)
                if current or not source or source['backend'] != 'slurm':
                    raise ValueError('Duplicate a Slurm lane using a new lane ID.')
            profile = _validate_settings(name, settings, current, body['reload_template'], source)
            preview = (SlurmBackend(profile).preview(Path(profile['work_dir']) / 'preview-job')
                       if profile['backend'] == 'slurm' else {'script': '', 'command': []})
            token = secrets.token_urlsafe(32)
            g.identity['lane_preview'] = dict(token=token, id=name, profile=profile,
                                              revision=current.get('revision', 0))
        except (OSError, ValueError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(preview_token=token, preview=preview['script'], command=preview['command'],
                       settings=dict(profile, id=name))

    @app.post('/api/admin/lanes')
    def apply_lane():
        body = request.get_json(silent=True)
        preview = g.identity.get('lane_preview')
        if (not isinstance(body, dict) or set(body) != {'preview_token'} or not preview
                or body['preview_token'] != preview['token']):
            return jsonify(error='Preview these settings before applying them.'), 400
        try:
            lane = store.save_lane(preview['id'], preview['profile'], expected_revision=preview['revision'])
        except ValueError as error:
            return jsonify(error=str(error)), 409
        g.identity.pop('lane_preview', None)
        return jsonify(lane=lane)
