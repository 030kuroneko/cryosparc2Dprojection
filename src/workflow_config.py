"""Shared workflow defaults and argument validation for launchers."""
import argparse
import contextlib
import io
import math
import re
from urllib.parse import urlsplit

from cryosparc_2d_projection import cli, axis_cli
from cryosparc_2d_projection.axis_search import AxisSearchConfig, AxisProximityConfig
from cryosparc_2d_projection.axis_presentation import parse_axis_rolls
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.scoring import BandLimitedScoreConfig
from cryosparc_2d_projection.surface_render import ClassRenderOptions

WORKFLOWS = {'orientation': cli, 'axis': axis_cli}


def actions(workflow):
    return [a for a in WORKFLOWS[workflow].build_parser()._actions if a.dest != 'help']


def default_values(workflow):
    return {a.dest: (False if isinstance(a, argparse._StoreTrueAction) else
                     '' if a.default is None or a.default == [] else str(a.default))
            for a in actions(workflow)}


def validate_url(value):
    try:
        url = urlsplit(value)
        _ = url.port
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
        parse_axis_rolls(args.axis_roll, symmetry=args.symmetry)
    return argv
