"""Installer contracts at the shell/package-manager boundary."""
import json
import os
from pathlib import Path
import subprocess
import builtins
import importlib
import tomllib
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arguments', [[], ['--manager', 'uv']])
def test_basic_install_exposes_commands_without_web_or_development_packages(tmp_path, arguments):
    commands = tmp_path / 'commands'
    commands.mkdir()
    uv = commands / 'uv'
    uv.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, pathlib, sys\n'
        'with open(os.environ["INSTALL_LOG"], "a") as out:\n'
        '    out.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'target = pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"]) / "bin"\n'
        'target.mkdir(parents=True, exist_ok=True)\n'
        'for name in ("cryosparc2d-projection", "cryosparc2d-axis-search", "cryosparc2d-service"):\n'
        '    script = target / name\n'
        '    script.write_text("#!/bin/sh\\necho ready\\n")\n'
        '    script.chmod(0o755)\n'
    )
    uv.chmod(0o755)
    bin_dir = tmp_path / 'user bin'
    environment = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
                       CRYOSPARC2D_HOME=str(tmp_path / 'installation'),
                       CRYOSPARC2D_BIN_DIR=str(bin_dir), INSTALL_LOG=str(tmp_path / 'log'))

    result = subprocess.run(['bash', str(ROOT / 'install.sh'), *arguments], cwd=ROOT,
                            env=environment, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    installed = subprocess.run([str(bin_dir / 'cryosparc2d-projection'), '--help'],
                               capture_output=True, text=True)
    assert installed.stdout.strip() == 'ready'
    calls = [json.loads(line) for line in (tmp_path / 'log').read_text().splitlines()]
    assert any(call[0] == 'sync' and '--locked' in call and '--no-default-groups' in call
               for call in calls)
    assert all('--extra' not in call and 'sudo' not in call for call in calls)
    assert 'cryosparc2d-service install' in result.stdout


@pytest.mark.parametrize("existing_environment", [False, True])
def test_miniforge_installs_into_a_dedicated_environment(tmp_path, existing_environment):
    if existing_environment:
        (tmp_path / 'app/venv/conda-meta').mkdir(parents=True)
    prefix = tmp_path / 'miniforge'
    (prefix / 'bin').mkdir(parents=True)
    conda = prefix / 'bin/conda'
    conda.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, pathlib, sys\n'
        'with open(os.environ["INSTALL_LOG"], "a") as out: out.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'target=pathlib.Path(sys.argv[sys.argv.index("--prefix")+1])/"bin"\n'
        'target.mkdir(parents=True, exist_ok=True)\n'
        'for name in ("python", "cryosparc2d-projection", "cryosparc2d-service"):\n'
        '    p=target/name\n'
        '    p.write_text("#!/bin/sh\\necho ready\\n")\n'
        '    p.chmod(0o755)\n'
    )
    conda.chmod(0o755)
    environment = dict(os.environ, CRYOSPARC2D_HOME=str(tmp_path / 'app'),
                       CRYOSPARC2D_BIN_DIR=str(tmp_path / 'bin'),
                       INSTALL_LOG=str(tmp_path / 'log'))
    result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--manager', 'miniforge',
                             '--miniforge-prefix', str(prefix)], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'log').read_text().splitlines()]
    assert calls[0][0] == ('install' if existing_environment else 'create')
    assert 'python=3.12' in calls[0] and 'pip' in calls[0]
    assert '--override-channels' in calls[0] and 'conda-forge' in calls[0]
    assert subprocess.check_output([str(tmp_path / 'bin/cryosparc2d-projection')], text=True).strip() == 'ready'


def test_basic_install_explains_how_to_add_gui_when_web_packages_are_absent(monkeypatch, capsys):
    entry = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['scripts']['cryosparc2d']
    module, name = entry.split(':')
    real_import = builtins.__import__

    def without_web(name, *args, **kwargs):
        if name in ('flask', 'waitress'):
            raise ModuleNotFoundError(f'No module named {name}', name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_web)
    command = getattr(importlib.import_module(module), name)

    assert command([]) == 1
    assert 'cryosparc2d-service install' in capsys.readouterr().err
