"""Launcher contracts: no live server or display required."""

import pytest

from cryosparc_2d_projection.workflow_config import build_arguments, default_values


def test_orientation_launcher_exposes_validated_image_fallback_settings():
    from cryosparc_2d_projection.workflow_config import prepare_workflow

    values = default_values("orientation")
    values.update(url="http://localhost:39000", project="P1", workspace="W1",
                  select_job="J1", refinement_job="J2", fallback_quality="fine",
                  fallback_device="cpu", fallback_batch_size="4")
    prepared = prepare_workflow("orientation", build_arguments("orientation", values))
    config = prepared.options["fallback_config"]
    assert config.quality == "fine"
    assert config.device == "cpu"
    assert config.batch_size == 4
    assert prepared.options["volume_source"].job_uid == "J2"


def configured(workflow):
    values = default_values(workflow)
    values.update(url='https://cryo.example', project='P1', workspace='W2', select_job='J3')
    values['refinement_job' if workflow == 'orientation' else 'volume_job'] = 'J4'
    return values


@pytest.mark.parametrize('workflow', ['orientation', 'axis'])
def test_defaults_round_trip_through_real_cli(workflow):
    from cryosparc_2d_projection import cli, axis_cli
    values = configured(workflow)
    args = build_arguments(workflow, values)
    parsed = (cli if workflow == 'orientation' else axis_cli).build_parser().parse_args(args)
    assert parsed.project == 'P1'
    assert parsed.render_grid_size is None
    assert parsed.comparison_dpi == 100


def test_axis_roll_and_boolean_forwarding():
    values = configured('axis')
    values.update(axis_roll='2fold=90;3fold=30', refine_near_axis=True)
    args = build_arguments('axis', values)
    assert args.count('--axis-roll') == 2
    assert '--refine-near-axis' in args


@pytest.mark.parametrize('workflow,field,value', [
    ('orientation', 'url', 'file:///tmp/x'),
    ('orientation', 'url', 'https://user:password@host'),
    ('orientation', 'project', 'J1'),
    ('orientation', 'select_job', ''),
    ('orientation', 'comparison_dpi', '0'),
    ('orientation', 'surface_level', 'nan'),
    ('orientation', 'diagnostic_high_resolution_A', '100'),
    ('orientation', 'symmetry', 'I2'),
    ('orientation', 'classes', '1,1'),
    ('axis', 'url', 'file:///tmp/x'),
    ('axis', 'project', 'J1'),
    ('axis', 'top_n', '0'),
    ('axis', 'axis_cone_degrees', '91'),
    ('axis', 'axis_roll', '2fold=nan'),
    ('axis', 'render_size', '1'),
    ('axis', 'surface_level', 'nan'),
    ('axis', 'comparison_dpi', '0'),
])
def test_web_and_cli_reject_invalid_settings_before_connecting(workflow, field, value):
    from cryosparc_2d_projection import cli, axis_cli

    values = configured(workflow)
    argv = build_arguments(workflow, values)
    values[field] = value
    with pytest.raises(ValueError):
        build_arguments(workflow, values)

    def unexpected_connection(_url):
        pytest.fail('Invalid settings must not create a CryoSPARC client')

    entry = cli if workflow == 'orientation' else axis_cli
    with pytest.raises((ValueError, SystemExit)):
        entry.main([*argv, '--' + field.replace('_', '-'), value],
                   client_factory=unexpected_connection)
