import json
from pathlib import Path
import subprocess
import tomllib

import pytest

from cryosparc_2d_projection.update_cli import main


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _initialized_repository(tmp_path):
    repository = tmp_path / "project"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.email", "test@example.test")
    _git(repository, "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("original\n")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    return repository


def test_update_command_refuses_tracked_changes(tmp_path, capsys):
    repository = _initialized_repository(tmp_path)
    (repository / "tracked.txt").write_text("locally changed\n")

    exit_code = main([], repository=repository)

    assert exit_code == 1
    assert "tracked files have local changes" in capsys.readouterr().err


def test_update_command_requires_an_upstream_branch(tmp_path, capsys):
    repository = _initialized_repository(tmp_path)

    exit_code = main([], repository=repository)

    assert exit_code == 1
    assert "current branch has no upstream" in capsys.readouterr().err


def test_update_command_refuses_detached_head(tmp_path, capsys):
    repository = _initialized_repository(tmp_path)
    _git(repository, "checkout", "--detach")

    exit_code = main([], repository=repository)

    assert exit_code == 1
    assert "detached HEAD" in capsys.readouterr().err


def test_update_command_fast_forwards_installs_and_tests_with_current_python(
    tmp_path, monkeypatch, capsys
):
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(remote)],
        check=True,
        capture_output=True,
    )
    repository = _initialized_repository(tmp_path)
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "--set-upstream", "origin", "main")

    publisher = tmp_path / "publisher"
    subprocess.run(
        ["git", "clone", str(remote), str(publisher)],
        check=True,
        capture_output=True,
    )
    _git(publisher, "config", "user.email", "publisher@example.test")
    _git(publisher, "config", "user.name", "Publisher")
    (publisher / "upstream.txt").write_text("new release\n")
    _git(publisher, "add", "upstream.txt")
    _git(publisher, "commit", "-m", "release")
    _git(publisher, "push")

    untracked = repository / "notes.txt"
    untracked.write_text("keep me\n")
    command_log = tmp_path / "python-commands.jsonl"
    python = tmp_path / "current-conda-python"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['UPDATE_COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    monkeypatch.setenv("UPDATE_COMMAND_LOG", str(command_log))

    exit_code = main([], repository=repository, python_executable=python)

    assert exit_code == 0
    assert (repository / "upstream.txt").read_text() == "new release\n"
    assert untracked.read_text() == "keep me\n"
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    assert commands == [
        ["-m", "pip", "install", "-e", f"{repository}[dev]"],
        ["-m", "pytest", "-q"],
    ]
    assert "Update complete." in capsys.readouterr().out


def test_update_command_help_does_not_require_a_repository(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "editable project checkout" in capsys.readouterr().out


def test_project_metadata_installs_update_command_and_test_dependencies():
    metadata = tomllib.loads(
        (Path(__file__).parent.parent / "pyproject.toml").read_text()
    )

    assert metadata["project"]["scripts"]["cryosparc-update"] == (
        "cryosparc_2d_projection.update_cli:main"
    )
    assert metadata["project"]["optional-dependencies"]["dev"] == [
        "pytest>=8,<9"
    ]


def test_update_command_requires_an_editable_git_checkout(tmp_path, capsys):
    exit_code = main([], repository=tmp_path)

    assert exit_code == 1
    assert "not an editable Git checkout" in capsys.readouterr().err


def test_update_command_reports_when_git_is_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", "")

    exit_code = main([], repository=tmp_path)

    assert exit_code == 1
    assert "Git executable was not found" in capsys.readouterr().err
