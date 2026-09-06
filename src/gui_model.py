"""Display-independent configuration and subprocess boundary for the launcher."""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
from urllib.parse import urlsplit

from cryosparc_2d_projection import cli, axis_cli
from cryosparc_2d_projection.axis_search import AxisSearchConfig, AxisProximityConfig
from cryosparc_2d_projection.axis_presentation import parse_axis_rolls
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.scoring import BandLimitedScoreConfig
from cryosparc_2d_projection.surface_render import ClassRenderOptions

WORKFLOWS = {'orientation': cli, 'axis': axis_cli}
SHARED = ('url', 'project', 'workspace')


def actions(workflow):
    return [a for a in WORKFLOWS[workflow].build_parser()._actions if a.dest != 'help']


def default_values(workflow):
    return {a.dest: (False if isinstance(a, argparse._StoreTrueAction) else
                     '' if a.default is None or a.default == [] else str(a.default))
            for a in actions(workflow)}


def validate_url(value):
    try:
        url = urlsplit(value)
        _ = url.port  # Validate port syntax and range.
    except ValueError as error:
        raise ValueError('Enter a valid CryoSPARC HTTP(S) URL.') from error
    if (url.scheme not in ('http', 'https') or not url.hostname or
            url.username is not None or url.password is not None or url.query or url.fragment or
            any(c.isspace() for c in value)):
        raise ValueError('Use a CryoSPARC HTTP(S) URL without credentials, query or fragment.')
    return value


def build_arguments(workflow, values):
    """Validate completely before a subprocess can connect or create a job."""
    argv = []
    for action in actions(workflow):
        value = values.get(action.dest, default_values(workflow)[action.dest])
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f'{action.dest} must be a checkbox value')
            if value:
                argv.append(action.option_strings[0])
            continue
        value = str(value).strip()
        if not value:
            if action.required:
                raise ValueError(f'{action.dest.replace("_", " ")} is required')
            continue
        parts = value.split(';') if isinstance(action, argparse._AppendAction) else [value]
        for part in parts:
            argv.extend((action.option_strings[0], part.strip()))
    error_text = io.StringIO()
    try:
        with contextlib.redirect_stderr(error_text):
            args = WORKFLOWS[workflow].build_parser().parse_args(argv)
    except SystemExit as error:
        raise ValueError(error_text.getvalue().split('error:')[-1].strip()) from error
    validate_url(args.url)
    for name, prefix in [('project', 'P'), ('workspace', 'W'), ('select_job', 'J'),
                         ('refinement_job' if workflow == 'orientation' else 'volume_job', 'J')]:
        if not re.fullmatch(prefix + r'[1-9]\d*', getattr(args, name)):
            raise ValueError(f'{name.replace("_", " ")} must look like {prefix}1')
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    ClassRenderOptions(args.surface_level, args.render_map, args.render_background,
                       args.render_size, args.render_grid_size)
    ComparisonRenderOptions(args.comparison_dpi, args.preview_page_size, args.auto_crop_2d)
    if workflow == 'orientation':
        BandLimitedScoreConfig(args.diagnostic_low_resolution_A, args.diagnostic_high_resolution_A,
                               args.diagnostic_mask_radius_fraction, args.diagnostic_mask_edge_fraction)
    else:
        AxisSearchConfig(low_resolution_A=args.low_resolution_A, high_resolution_A=args.high_resolution_A,
                         mask_radius_fraction=args.mask_radius_fraction, mask_edge_fraction=args.mask_edge_fraction,
                         roll_coarse_step_degrees=args.roll_coarse_step, roll_refine_step_degrees=args.roll_refine_step,
                         shift_bound_fraction=args.shift_bound_fraction, top_n=args.top_n,
                         mirror_warning_margin=args.mirror_warning_margin)
        AxisProximityConfig(args.axis_cone_degrees, args.tilt_coarse_step, args.tilt_refine_step)
        parse_axis_rolls(args.axis_roll)
    return argv


def clean_settings(pages):
    if not isinstance(pages, dict) or set(pages) != set(WORKFLOWS):
        raise ValueError('Settings must contain both workflow pages.')
    result = {}
    for name in WORKFLOWS:
        if not isinstance(pages[name], dict):
            raise ValueError('Invalid workflow settings')
        result[name] = defaults = default_values(name)
        for key in defaults:
            value = pages[name].get(key, defaults[key])
            if type(value) is not type(defaults[key]):
                raise ValueError(f'Invalid setting: {key}')
            defaults[key] = value
        if defaults['url']:
            validate_url(defaults['url'])
    if any(result['orientation'][key] != result['axis'][key] for key in SHARED):
        raise ValueError('Connection settings must be the same on both pages.')
    return result


def save_settings(path, pages):
    payload = {'version': 1, 'pages': clean_settings(pages)}
    Path(path).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def load_settings(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError('Unsupported GUI settings version')
    return clean_settings(payload.get('pages'))


class JobRunner:
    """One child at a time; all GUI updates are delivered through a queue."""
    def __init__(self):
        self.events = queue.Queue()
        self.thread = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, command):
        if self.running:
            raise RuntimeError('A job is already running')
        self.thread = threading.Thread(target=self._run, args=(command,), daemon=False)
        self.thread.start()

    def _run(self, command):
        try:
            env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', env=env) as process:
                for line in process.stdout:
                    self.events.put(('log', line))
                code = process.wait()
        except Exception as error:
            self.events.put(('log', f'Could not launch job: {error}\n'))
            code = 1
        self.events.put(('finished', code))

    def drain(self, limit=500):
        events = []
        for _ in range(limit):
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        return events
