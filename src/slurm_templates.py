"""Render administrator-owned batch templates without evaluating template code."""
import math
import re
import shlex

DEFAULT_TEMPLATE = '''#!/bin/bash
# Lane resources are filled by the launcher.
# Add cluster-specific module loads or environment setup below.
exec {{ run_cmd }}
'''

BUILTIN_VARIABLES = ('cpus', 'gpus', 'memory_mb', 'time_minutes', 'partition', 'account', 'qos',
                     'python', 'run_cmd', 'job_dir', 'log_path', 'job_id', 'job_name',
                     'lane_id', 'project_uid', 'workspace_uid', 'workflow')
_PLACEHOLDER = re.compile(r'{{\s*([A-Za-z_][A-Za-z0-9_]*)(\s*\|\s*quote)?\s*}}')
# These settings are owned by the lane/worker contract, including GPU aliases
# which otherwise accumulate with --gpus rather than overriding it.
_MANAGED_LONG = {
    'cpus-per-task', 'cpus-per-gpu', 'gpus', 'gres', 'gpus-per-node', 'gpus-per-task',
    'gpus-per-socket', 'mem', 'mem-per-cpu', 'mem-per-gpu', 'time', 'partition', 'account',
    'qos', 'nodes', 'ntasks', 'ntasks-per-node', 'ntasks-per-core', 'ntasks-per-socket',
    'ntasks-per-gpu', 'job-name', 'chdir', 'output', 'error', 'export', 'export-file',
    'requeue', 'no-requeue', 'array', 'wrap',
}
_MANAGED_SHORT = set('cGtpAqNnJDoea')
_NO_VALUE = {'requeue', 'no-requeue'}


def validate_variables(variables):
    if not isinstance(variables, dict) or len(variables) > 32:
        raise ValueError('Custom variables must be an object with at most 32 entries.')
    for name, value in variables.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', name) or name in BUILTIN_VARIABLES:
            raise ValueError('Invalid or reserved custom variable: ' + name)
        if (type(value) not in (str, int, float, bool) or
                (type(value) is float and not math.isfinite(value)) or
                len(str(value)) > 2048 or any(c in str(value) for c in '\r\n\x00')):
            raise ValueError('Custom variables must be single-line scalar values: ' + name)


def _remaining_directives(line):
    tokens = shlex.split(line, comments=True)
    kept = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        key, _, _value = token.partition('=')
        managed = None
        if key.startswith('--'):
            matches = {name for name in _MANAGED_LONG if name.startswith(key[2:])}
            if matches:
                managed = key[2:]
        elif token.startswith('-') and len(token) > 1 and token[1] in _MANAGED_SHORT:
            managed = token[1]
        if managed is not None:
            takes_value = managed not in _NO_VALUE
            attached = '=' in token or (not token.startswith('--') and len(token) > 2)
            if takes_value and not attached and i + 1 < len(tokens):
                i += 1
        else:
            kept.append(token)
        i += 1
    return shlex.join(kept)


def render_submission(profile, directory, run_cmd, options):
    template = profile.get('template_text', DEFAULT_TEMPLATE)
    if not isinstance(template, str) or not template or len(template.encode('utf-8')) > 32768 or '\x00' in template:
        raise ValueError('Submission template must be nonempty UTF-8 text, at most 32 KiB.')
    variables = profile.get('variables', {})
    validate_variables(variables)
    job = profile.get('job_context', {})
    values = dict(variables, **{name: profile.get(name, '') for name in BUILTIN_VARIABLES})
    values.update(cpus=profile.get('cpus', 1), gpus=profile.get('gpus', 0),
                  memory_mb=profile.get('memory_mb', 4096), time_minutes=profile.get('time_minutes', 60),
                  run_cmd=run_cmd, job_dir=str(directory), log_path=str(directory / 'bootstrap.log'),
                  job_id=directory.name, job_name='projection-' + directory.name,
                  **{key: job.get(key, '') for key in ('project_uid', 'workspace_uid', 'workflow')})
    seen = set()
    def substitute(match):
        name = match[1]
        if name not in values:
            raise ValueError('Unknown template variable: ' + name)
        seen.add(name)
        value = str(values[name])
        return shlex.quote(value) if match[2] else value
    rendered = _PLACEHOLDER.sub(substitute, template)
    if '{{' in rendered or '{%' in rendered or '}}' in rendered:
        raise ValueError('Use {{ variable }} or {{ variable | quote }} placeholders; template code is not supported.')
    command_lines = '\n'.join(line for line in template.splitlines() if not line.lstrip().startswith('#'))
    if 'run_cmd' not in seen or not re.search(r'{{\s*run_cmd\s*}}', command_lines):
        raise ValueError('Template must include {{ run_cmd }} to start the workflow.')
    lines = rendered.splitlines()
    if not lines or not lines[0].startswith('#!'):
        raise ValueError('Submission template must start with a shell shebang, such as #!/bin/bash.')
    result = [lines[0]]
    result.extend('#SBATCH ' + shlex.quote(option) for option in options)
    result.append('umask 077')
    started = False
    directive_position = len(options) + 1
    for line in lines[1:]:
        if line.lstrip().startswith('#SBATCH'):
            if started:
                raise ValueError('Place #SBATCH directives before the first shell command.')
            extra = _remaining_directives(line.lstrip()[7:].strip())
            if extra:
                # Keep all scheduler directives before the generated umask command.
                result.insert(directive_position, '#SBATCH ' + extra)
                directive_position += 1
        else:
            if line.strip() and not line.lstrip().startswith('#'):
                started = True
            result.append(line)
    return '\n'.join(result) + '\n'
