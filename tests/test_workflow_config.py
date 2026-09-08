"""Launcher contracts: no live server or display required."""

import pytest

from cryosparc_2d_projection.workflow_config import build_arguments, default_values


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


@pytest.mark.parametrize('field,value', [
    ('url', 'file:///tmp/x'), ('url', 'https://user:password@host'),
    ('project', 'J1'), ('select_job', ''), ('comparison_dpi', '0'),
    ('surface_level', 'nan'), ('diagnostic_high_resolution_A', '100'),
    ('symmetry', 'I2'), ('classes', '1,1'),
])
def test_invalid_input_fails_before_launch(field, value):
    values = configured('orientation')
    values[field] = value
    with pytest.raises(ValueError):
        build_arguments('orientation', values)


@pytest.mark.parametrize('field,value', [('top_n', '0'), ('axis_cone_degrees', '91'),
                                         ('axis_roll', '2fold=nan'), ('render_size', '1')])
def test_axis_validation_before_connecting(field, value):
    values = configured('axis')
    values[field] = value
    with pytest.raises(ValueError):
        build_arguments('axis', values)
