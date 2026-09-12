"""Shared workflow defaults and argument validation for launchers."""
import argparse
from dataclasses import dataclass
import contextlib
import io
import math
import re
from urllib.parse import urlsplit

from cryosparc_2d_projection import cli, axis_cli
from cryosparc_2d_projection.external_job_adapter import ExternalJobSource
from cryosparc_2d_projection.axis_search import AxisSearchConfig, AxisProximityConfig
from cryosparc_2d_projection.axis_presentation import parse_axis_rolls
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.scoring import BandLimitedScoreConfig
from cryosparc_2d_projection.surface_render import ClassRenderOptions

WORKFLOWS = {'orientation': cli, 'axis': axis_cli}


@dataclass(frozen=True)
class WorkflowField:
    """Launcher-facing field metadata without parser implementation details."""

    key: str
    default: str | bool
    required: bool
    type: str
    choices: tuple


def _actions(workflow):
    return [a for a in WORKFLOWS[workflow].build_parser()._actions if a.dest != 'help']


def workflow_fields(workflow):
    fields = []
    for action in _actions(workflow):
        boolean = isinstance(action, argparse._StoreTrueAction)
        default = (False if boolean else
                   '' if action.default is None or action.default == [] else str(action.default))
        fields.append(WorkflowField(
            key=action.dest, default=default, required=action.required,
            type='boolean' if boolean else 'text', choices=tuple(action.choices or ()),
        ))
    return tuple(fields)


def default_values(workflow):
    return {field.key: field.default for field in workflow_fields(workflow)}


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
    defaults = default_values(workflow)
    for action in _actions(workflow):
        value = values.get(action.dest, defaults[action.dest])
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
            prepare_workflow(workflow, argv)
    except SystemExit as error:
        raise ValueError(error_text.getvalue().split('error:')[-1].strip()) from error
    return argv


@dataclass(frozen=True)
class PreparedWorkflow:
    """Validated connection selectors and ready-to-run workflow settings."""

    url: str
    project: str
    options: dict


def prepare_workflow(workflow, argv=None):
    """Resolve every workflow setting before any CryoSPARC connection is made."""
    args = WORKFLOWS[workflow].build_parser().parse_args(argv)
    validate_url(args.url)
    for name, prefix in [('project', 'P'), ('workspace', 'W'), ('select_job', 'J'),
                         ('refinement_job' if workflow == 'orientation' else 'volume_job', 'J')]:
        if not re.fullmatch(prefix + r'[1-9]\d*', getattr(args, name)):
            raise ValueError(f'{name.replace("_", " ")} must look like {prefix}1')
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    options = dict(
        workspace_uid=args.workspace,
        symmetry=args.symmetry,
        render_options=ClassRenderOptions(
            surface_level=args.surface_level,
            map_name=args.render_map,
            background=args.render_background,
            image_size=args.render_size,
            grid_size=args.render_grid_size,
        ),
        comparison_options=ComparisonRenderOptions(
            dpi=args.comparison_dpi,
            page_size=args.preview_page_size,
            auto_crop_2d=args.auto_crop_2d,
        ),
    )
    if workflow == 'orientation':
        options.update(
            select_2d_source=ExternalJobSource(args.select_job, args.select_output),
            select_templates_source=ExternalJobSource(args.select_job, args.templates_output),
            refinement_source=ExternalJobSource(args.refinement_job, args.refinement_particles_output),
            volume_source=ExternalJobSource(args.refinement_job, args.volume_output),
            interactive_class_numbers=args.classes or (),
            diagnostic_score_config=BandLimitedScoreConfig(
                low_resolution_A=args.diagnostic_low_resolution_A,
                high_resolution_A=args.diagnostic_high_resolution_A,
                mask_radius_fraction=args.diagnostic_mask_radius_fraction,
                mask_edge_fraction=args.diagnostic_mask_edge_fraction,
            ),
        )
    else:
        options.update(
            templates_source=ExternalJobSource(args.select_job, args.select_output),
            volume_source=ExternalJobSource(args.volume_job, args.volume_output),
            families=args.axis_family,
            config=AxisSearchConfig(
                low_resolution_A=args.low_resolution_A,
                high_resolution_A=args.high_resolution_A,
                mask_radius_fraction=args.mask_radius_fraction,
                mask_edge_fraction=args.mask_edge_fraction,
                roll_coarse_step_degrees=args.roll_coarse_step,
                roll_refine_step_degrees=args.roll_refine_step,
                shift_bound_fraction=args.shift_bound_fraction,
                top_n=args.top_n,
                mirror_warning_margin=args.mirror_warning_margin,
            ),
            proximity_config=AxisProximityConfig(
                cone_degrees=args.axis_cone_degrees,
                coarse_step_degrees=args.tilt_coarse_step,
                refine_step_degrees=args.tilt_refine_step,
            ),
            axis_rolls=parse_axis_rolls(args.axis_roll, symmetry=args.symmetry),
            refine_near_axis=args.refine_near_axis,
        )
    return PreparedWorkflow(args.url, args.project, options)
