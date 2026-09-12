"""Service commands exercised against filesystem and operating-system boundaries."""
from pathlib import Path
import io
import json
import os
import sys
import uuid
from types import SimpleNamespace
import subprocess

import pytest


def test_stop_does_not_disable_boot_startup(tmp_path, monkeypatch):
    from cryosparc_2d_projection.service_cli import main
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr('platform.system', lambda: 'Linux')
    monkeypatch.setattr('os.geteuid', lambda: 0)
    monkeypatch.setattr('shutil.which', lambda name: f'/usr/bin/{name}')
    unit = tmp_path / 'etc/systemd/system/cryosparc2d.service'
    unit.parent.mkdir(parents=True)
    unit.write_text('[Service]\n')

    assert main(['stop'], root=tmp_path, run=run) == 0
    assert calls == [['/usr/bin/systemctl', 'stop', 'cryosparc2d.service']]


@pytest.mark.parametrize('retry_after_failure', [False, True])
def test_install_adds_gui_and_private_service_without_changing_basic_environment(tmp_path, monkeypatch, retry_after_failure):
    from cryosparc_2d_projection.service_cli import main
    calls = []
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    monkeypatch.setattr('os.geteuid', lambda: 0)
    monkeypatch.setattr('shutil.which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr('pwd.getpwnam', lambda name: SimpleNamespace(
        pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name=name))
    monkeypatch.setenv('SUDO_USER', 'lab-deployer')

    def run(argv, **kwargs):
        calls.append(argv)
        if 'sync' in argv:
            bin_dir = Path(kwargs['env']['UV_PROJECT_ENVIRONMENT']) / 'bin'
            bin_dir.mkdir(parents=True, exist_ok=True)
            for name in ('python', 'cryosparc2d', 'cryosparc2d-service'):
                (bin_dir / name).write_text('placeholder')
        if retry_after_failure and 'enable' in argv and sum('enable' in c for c in calls) == 1:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    basic = tmp_path / 'basic/venv/keep'
    basic.parent.mkdir(parents=True)
    basic.write_text('CLI stays available')
    answers = iter(['https://cryo.example', '', '', '', '', '', ''])
    previous_umask = os.umask(0o077)
    try:
        result = main(['install', '--source', str(Path(__file__).resolve().parents[1]),
                       '--uv', '/usr/bin/uv'], root=tmp_path, run=run,
                      prompt=lambda message: next(answers),
                      opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}'))
        if retry_after_failure:
            assert result == 1
            answers = iter(['', '', '', '', '', '', ''])
            result = main(['install', '--source', str(Path(__file__).resolve().parents[1]),
                           '--uv', '/usr/bin/uv'], root=tmp_path, run=run,
                          prompt=lambda message: next(answers),
                          opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}'))
    finally:
        os.umask(previous_umask)

    assert result == 0
    config = json.loads((tmp_path / 'etc/cryosparc2d/web.json').read_text())
    assert config['cryosparc_url'] == 'https://cryo.example'
    assert config['host'] == '0.0.0.0' and config['port'] == 40000
    assert basic.read_text() == 'CLI stays available'
    unit = (tmp_path / 'etc/systemd/system/cryosparc2d.service').read_text()
    assert 'User=cryosparc2d' in unit and 'KillMode=process' in unit
    assert 'sync' in next(c for c in calls if '--extra' in c)
    assert ['--extra', 'web'] == next(c for c in calls if '--extra' in c)[-2:]
    assert ['/usr/bin/systemctl', 'enable', '--now', 'cryosparc2d.service'] in calls
    if retry_after_failure:
        assert sum('enable' in c for c in calls) == 2
    assert (Path(config['data_dir']).stat().st_mode & 0o777) == 0o700
    rule = (tmp_path / 'etc/sudoers.d/cryosparc2d').read_text()
    assert 'lab-deployer ALL=(root) NOPASSWD:' in rule
    assert '/usr/bin/systemctl stop cryosparc2d.service' in rule
    assert '*' not in rule and 'configure' not in rule
    assert any(c[0] == '/usr/bin/visudo' for c in calls)
    assert (tmp_path / 'usr/local/bin/cryosparc2d-service').is_symlink()
    assert (tmp_path / 'opt/cryosparc2d').stat().st_mode & 0o005 == 0o005
    assert (tmp_path / 'etc/cryosparc2d').stat().st_mode & 0o050 == 0o050


@pytest.fixture
def installed_service(tmp_path, monkeypatch):
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    monkeypatch.setattr('os.geteuid', lambda: 0)
    monkeypatch.setattr('shutil.which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr('pwd.getpwnam', lambda name: SimpleNamespace(
        pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name=name))
    from cryosparc_2d_projection.service_setup import Layout
    layout = Layout(tmp_path)
    layout.unit.parent.mkdir(parents=True)
    layout.unit.write_text('[Service]\n')
    config = {'cryosparc_url': 'https://cryo.example', 'host': '0.0.0.0', 'port': 40000,
              'data_dir': str(layout.data), 'work_dir': str(layout.data),
              'profiles': {'local': {'backend': 'local', 'python': sys.executable}}}
    layout.config.parent.mkdir(parents=True)
    layout.config.write_text(json.dumps(config))
    layout.data.mkdir(parents=True, mode=0o700)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    return layout, config, calls, run


def test_configure_retains_values_and_applies_validated_port_change(installed_service):
    from cryosparc_2d_projection.service_cli import main
    layout, config, calls, run = installed_service
    answers = iter(['', '', '41000', '', '', '', ''])

    assert main(['configure'], root=layout.root, run=run,
                prompt=lambda message: next(answers),
                opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}')) == 0

    changed = json.loads(layout.config.read_text())
    assert changed == dict(config, port=41000)
    assert ['/usr/bin/systemctl', 'stop', 'cryosparc2d.service'] in calls
    assert ['/usr/bin/systemctl', 'start', 'cryosparc2d.service'] in calls


def test_configure_rejects_connection_change_with_queued_job(installed_service, capsys):
    from cryosparc_2d_projection.service_cli import main
    from cryosparc_2d_projection.web_jobs import JobStore
    layout, config, calls, run = installed_service
    store = JobStore(config)
    store.submit({'owner': 'alice', 'email': 'alice@example.test', 'token': 'test-token'}, {
        'workflow': 'orientation', 'profile': 'local', 'request_id': str(uuid.uuid4()),
        'values': {'project': 'P1', 'workspace': 'W1', 'select_job': 'J1', 'refinement_job': 'J2'},
    })
    answers = iter(['https://other.example', '', '', '', '', '', ''])

    result = main(['configure'], root=layout.root, run=run,
                  prompt=lambda message: next(answers),
                  opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}'))

    assert result == 1
    assert json.loads(layout.config.read_text()) == config
    assert 'queued or running' in capsys.readouterr().err
    assert store.list('alice')[0]['state'] == 'queued'


def test_failed_restart_restores_previous_configuration(installed_service):
    from cryosparc_2d_projection.service_cli import main
    layout, config, calls, normal_run = installed_service

    def run(argv, **kwargs):
        if argv[1:3] == ['start', 'cryosparc2d.service']:
            raise subprocess.CalledProcessError(1, argv)
        return normal_run(argv, **kwargs)

    answers = iter(['', '', '41000', '', '', '', ''])
    result = main(['configure'], root=layout.root, run=run,
                  prompt=lambda message: next(answers),
                  opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}'))

    assert result == 1
    assert json.loads(layout.config.read_text()) == config
    assert ['/usr/bin/systemctl', 'restart', 'cryosparc2d.service'] in calls


def test_configure_does_not_start_a_service_the_user_stopped(installed_service):
    from cryosparc_2d_projection.service_cli import main
    layout, config, calls, normal_run = installed_service

    def run(argv, **kwargs):
        result = normal_run(argv, **kwargs)
        if 'is-active' in argv:
            result.returncode = 3
        return result

    answers = iter(['', '', '41000', '', '', '', ''])
    assert main(['configure'], root=layout.root, run=run,
                prompt=lambda message: next(answers),
                opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}')) == 0
    assert not any('start' in c or 'restart' in c or 'enable' in c for c in calls)


def test_enter_keeps_existing_loopback_http_settings(installed_service):
    from cryosparc_2d_projection.service_cli import main
    layout, config, calls, run = installed_service
    config.update(host='127.0.0.1', public_url='http://127.0.0.1:40000', allow_http=True)
    layout.config.write_text(json.dumps(config))

    assert main(['configure'], root=layout.root, run=run, prompt=lambda message: '',
                opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}')) == 0
    assert json.loads(layout.config.read_text()) == config
    assert calls == []


def test_two_configuration_sessions_cannot_overwrite_each_other(installed_service, capsys):
    import fcntl
    from cryosparc_2d_projection.service_cli import main
    layout, config, calls, run = installed_service
    with (layout.config.parent / '.setup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = main(['configure'], root=layout.root, run=run, prompt=lambda message: '',
                      opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}'))
    assert result == 1
    assert 'already in progress' in capsys.readouterr().err
    assert json.loads(layout.config.read_text()) == config


def test_configure_validates_the_computation_environment_before_saving(installed_service):
    from cryosparc_2d_projection.service_cli import main
    layout, config, calls, normal_run = installed_service

    def run(argv, **kwargs):
        if argv[0] == '/usr/bin/runuser':
            # Run the real validation in a subprocess under the current test
            # account; only account switching is replaced on this host.
            return subprocess.run([sys.executable, *argv[5:]], **kwargs)
        return normal_run(argv, **kwargs)

    answers = iter(['', '', '', '', '', '', '/usr/bin/true'])
    result = main(['configure'], root=layout.root, run=run,
                  prompt=lambda message: next(answers),
                  opener=lambda *args, **kwargs: io.BytesIO(b'{"csrf":"ready"}'))
    assert result == 1
    assert json.loads(layout.config.read_text()) == config
    assert not any('stop' in c for c in calls)


def test_install_finds_uv_in_miniforge_environment_without_activation(tmp_path, monkeypatch):
    from cryosparc_2d_projection.service_cli import main
    executable = tmp_path / 'bin/python'
    executable.parent.mkdir()
    uv = executable.with_name('uv')
    uv.touch()
    monkeypatch.setattr('sys.executable', str(executable))
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/systemctl' if name == 'systemctl' else None)
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    monkeypatch.setattr('os.geteuid', lambda: 1000)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    assert main(['install'], root=tmp_path, run=run) == 0
    assert calls[0][0] == 'sudo'
    assert calls[0][-2:] == ['--uv', str(uv)]
