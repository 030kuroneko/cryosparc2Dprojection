"""Launcher contracts: no live server or display required."""
import json
import sys
import time

import pytest

from cryosparc_2d_projection.gui_model import (
    JobRunner, build_arguments, default_values, load_settings, save_settings,
)


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


def test_settings_are_versioned_and_do_not_keep_unknown_secrets(tmp_path):
    path = tmp_path / 'settings.json'
    pages = {name: configured(name) for name in ('orientation', 'axis')}
    pages['axis']['password'] = 'secret'
    save_settings(path, pages)
    assert 'secret' not in path.read_text()
    assert load_settings(path)['orientation']['project'] == 'P1'
    path.write_text(json.dumps({'version': 999, 'pages': {}}))
    with pytest.raises(ValueError):
        load_settings(path)


def test_runner_streams_failure_and_can_retry():
    runner = JobRunner()
    runner.start([sys.executable, '-u', '-c', 'print("stage one"); raise SystemExit(3)'])
    events = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        events.extend(runner.drain())
        if any(kind == 'finished' for kind, _ in events):
            break
        time.sleep(.01)
    assert ('log', 'stage one\n') in events
    assert ('finished', 3) in events
    assert not runner.running
    runner.start([sys.executable, '-c', 'pass'])
    while runner.running and time.monotonic() < deadline:
        time.sleep(.01)
    assert not runner.running


def test_runner_rejects_double_submission():
    runner = JobRunner()
    runner.start([sys.executable, '-c', 'import time; time.sleep(.2)'])
    with pytest.raises(RuntimeError):
        runner.start([sys.executable, '-c', 'pass'])
    runner.thread.join(timeout=5)
    assert not runner.running


def test_worker_dispatches_to_existing_workflow(monkeypatch):
    from cryosparc_2d_projection import gui
    received = []
    monkeypatch.setattr(gui.WORKFLOWS['axis'], 'main', lambda args: received.append(args) or 0)
    assert gui.main(['--worker', 'axis', '--project', 'P1']) == 0
    assert received == [['--project', 'P1']]


def test_launch_failure_is_reported():
    runner = JobRunner()
    runner.start(['/nonexistent/cryosparc-gui-test-executable'])
    runner.thread.join(timeout=5)
    events = runner.drain()
    assert ('finished', 1) in events
    assert any(kind == 'log' and 'Could not launch job' in value for kind, value in events)


@pytest.mark.parametrize('field,value', [('top_n', '0'), ('axis_cone_degrees', '91'),
                                         ('axis_roll', '2fold=nan'), ('render_size', '1')])
def test_axis_validation_before_connecting(field, value):
    values = configured('axis')
    values[field] = value
    with pytest.raises(ValueError):
        build_arguments('axis', values)


def test_desktop_pages_and_shared_connection():
    """Run under a desktop or xvfb-run; skip explicitly on headless machines."""
    tk = pytest.importorskip('tkinter')
    from cryosparc_2d_projection.gui import Launcher
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('A desktop display is required for the Tk smoke test')
    try:
        app = Launcher(root)
        root.update()
        assert len(app.notebook.tabs()) == 2
        app.variables['orientation']['project'].set('P9')
        assert app.variables['axis']['project'].get() == 'P9'
        app.notebook.select(1)
        root.update()
        assert app.selected() == 'axis'
    finally:
        root.destroy()
