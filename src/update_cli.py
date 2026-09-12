"""Update the editable project checkout from its configured Git upstream."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def build_parser():
    return argparse.ArgumentParser(
        description="Update the editable project checkout and current Python environment."
    )


def main(argv=None, *, repository=None, python_executable=None):
    build_parser().parse_args(argv)
    repository = (
        Path(repository)
        if repository is not None
        else Path(__file__).resolve().parent.parent
    )
    python_executable = str(python_executable or sys.executable)

    if shutil.which("git") is None:
        print("Update aborted: Git executable was not found.", file=sys.stderr)
        return 1

    checkout = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        checkout.returncode != 0
        or Path(checkout.stdout.strip()).resolve() != repository.resolve()
    ):
        print(
            "Update aborted: installation source is not an editable Git checkout.",
            file=sys.stderr,
        )
        return 1

    working_tree_changed = subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet"],
        check=False,
        capture_output=True,
    ).returncode
    index_changed = subprocess.run(
        ["git", "-C", str(repository), "diff", "--cached", "--quiet"],
        check=False,
        capture_output=True,
    ).returncode
    if working_tree_changed or index_changed:
        print("Update aborted: tracked files have local changes.", file=sys.stderr)
        return 1

    branch = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "--quiet", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if branch.returncode != 0:
        print("Update aborted: repository is in detached HEAD state.", file=sys.stderr)
        return 1

    upstream = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if upstream.returncode != 0:
        print("Update aborted: current branch has no upstream.", file=sys.stderr)
        return 1

    try:
        environment = Path(python_executable).parent.parent
        venv_config = environment / 'pyvenv.cfg'
        managed = (venv_config.exists() and
                   any(line.startswith('uv =') for line in venv_config.read_text().splitlines()))
        uv = shutil.which('uv') if managed else None
        if managed and not uv:
            raise FileNotFoundError('uv is required to update this installation; add uv to PATH.')
        modes = set()
        if managed:
            probe = subprocess.run([
                python_executable, '-c',
                'import importlib.util as u; '
                'print(",".join(name for name, enabled in '
                '[("web", any(u.find_spec(m) for m in ("flask", "waitress"))), '
                '("dev", u.find_spec("pytest"))] if enabled) or "cli")',
            ], capture_output=True, text=True, check=True)
            modes = set(probe.stdout.strip().split(','))
        print(f"Updating {branch.stdout.strip()} from {upstream.stdout.strip()}...")
        subprocess.run(
            ["git", "-C", str(repository), "pull", "--ff-only"],
            check=True,
        )
        print(f"Installing into {python_executable}...")
        if managed:
            command = [uv, 'sync', '--project', str(repository), '--locked', '--no-default-groups']
            if 'web' in modes:
                command.extend(['--extra', 'web'])
            if 'dev' in modes:
                command.extend(['--group', 'dev'])
            subprocess.run(command, check=True, cwd=repository,
                           env=dict(os.environ, UV_PROJECT_ENVIRONMENT=str(environment)))
            print('Checking CLI entry points...')
            for module in ('cli', 'axis_cli', *(['web_launcher'] if 'web' in modes else [])):
                subprocess.run([python_executable, '-m', 'cryosparc_2d_projection.' + module, '--help'],
                               check=True, capture_output=True, text=True, cwd=repository)
        else:
            subprocess.run(
                [
                    python_executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    f"{repository}[dev]",
                ],
                check=True,
                cwd=repository,
            )
        if not managed or 'dev' in modes:
            print("Running tests...")
            subprocess.run(
                [python_executable, "-m", "pytest", "-q"],
                check=True,
                cwd=repository,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Update failed: {error}", file=sys.stderr)
        return 1

    print("Update complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
