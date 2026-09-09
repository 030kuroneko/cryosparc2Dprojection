"""Installed command and server launch contracts."""
import sys
from pathlib import Path
import tomllib

import pytest

pytest.importorskip('flask')
from cryosparc_2d_projection.web import main


def test_simple_launch_uses_port_40000_without_slurm_or_json(tmp_path, monkeypatch, capsys):
    import waitress
    observed = {}
    def serve(app, **options):
        observed.update(options)
        assert app.test_client().get('/api/session', base_url='http://127.0.0.1:40000').status_code == 200
    monkeypatch.setattr(waitress, 'serve', serve)
    main(['--url', 'https://cryo.example', '--data-dir', str(tmp_path)])
    assert observed['host'] == '127.0.0.1'
    assert observed['port'] == 40000
    assert 'http://127.0.0.1:40000' in capsys.readouterr().out


def test_all_project_commands_use_cryosparc2d_prefix():
    metadata = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    commands = metadata['project']['scripts']
    assert commands['cryosparc2d'] == 'cryosparc_2d_projection.web:main'
    assert all(name.startswith('cryosparc2d') for name in commands)


@pytest.mark.parametrize('flags, message', [
    (['--port', '0'], '65535'),
    (['--public-url', 'http://lab.example'], 'HTTPS'),
    (['--slurm'], 'unrecognized'),
])
def test_bad_launch_flags_fail_with_actionable_errors(tmp_path, flags, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(['--url', 'https://cryo.example', '--data-dir', str(tmp_path), *flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_flags_override_optional_configuration(tmp_path, monkeypatch):
    import json
    import waitress
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'cryosparc_url': 'https://old.example', 'port': 40001,
                                 'data_dir': str(tmp_path / 'state')}))
    def serve(app, **options):
        assert options['port'] == 40002
        response = app.test_client().get('/api/session', base_url='http://127.0.0.1:40002')
        assert response.json['cryosparc_url'] == 'https://new.example'
    monkeypatch.setattr(waitress, 'serve', serve)
    main(['--config', str(config), '--url', 'https://new.example', '--port', '40002'])


def test_public_url_flag_replaces_development_http_policy(tmp_path, monkeypatch):
    import json
    import waitress
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'cryosparc_url': 'https://cryo.example',
        'public_url': 'http://127.0.0.1:40000', 'allow_http': True,
        'data_dir': str(tmp_path / 'state')}))
    def serve(app, **options):
        response = app.test_client().get('/api/session', base_url='https://lab.example')
        assert response.status_code == 200
        assert 'Secure' in response.headers['Set-Cookie']
    monkeypatch.setattr(waitress, 'serve', serve)
    main(['--config', str(config), '--public-url', 'https://lab.example'])


def test_explicit_lan_http_listens_on_all_interfaces_with_exact_origin(tmp_path, monkeypatch, capsys):
    import waitress
    def serve(app, **options):
        assert options['host'] == '0.0.0.0'
        assert options['port'] == 40000
        client = app.test_client()
        response = client.get('/api/session', base_url='http://192.168.1.20:40000')
        assert response.status_code == 200
        assert 'Secure;' not in response.headers['Set-Cookie']
        assert client.get('/api/session', base_url='http://evil.example:40000').status_code == 400
        assert client.post('/api/login', base_url='http://192.168.1.20:40000',
            json={'email': 'test', 'password': 'test'}, headers={
                'Origin': 'http://evil.example:40000', 'X-CSRF-Token': response.json['csrf']}).status_code == 403
    monkeypatch.setattr(waitress, 'serve', serve)
    main(['--url', 'https://cryo.example', '--host', '0.0.0.0',
          '--public-url', 'http://192.168.1.20:40000', '--data-dir', str(tmp_path)])
    assert 'unencrypted' in capsys.readouterr().out


def test_host_alone_starts_lan_server_without_public_url(tmp_path, monkeypatch, capsys):
    import waitress
    def serve(app, **options):
        assert options['host'] == '0.0.0.0'
        assert options['port'] == 40000
        client = app.test_client()
        for host in ('192.168.1.20', '10.2.3.4', '127.0.0.1', 'localhost'):
            assert client.get('/api/session', base_url=f'http://{host}:40000').status_code == 200
    monkeypatch.setattr(waitress, 'serve', serve)
    main(['--url', 'https://cryo.example', '--host', '0.0.0.0', '--data-dir', str(tmp_path)])
    output = capsys.readouterr().out
    assert 'unencrypted' in output
    assert 'http://SERVER-IP:40000' in output


def test_web_help_works_without_retired_desktop_modules():
    import subprocess
    result = subprocess.run([sys.executable, '-c', """
import sys
sys.modules['cryosparc_2d_projection.gui'] = None
sys.modules['cryosparc_2d_projection.gui_model'] = None
from cryosparc_2d_projection.web import main
main(['--help'])
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--port' in result.stdout
