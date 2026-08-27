"""Update the editable project checkout from its configured Git upstream."""

import argparse
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
        print(f"Updating {branch.stdout.strip()} from {upstream.stdout.strip()}...")
        subprocess.run(
            ["git", "-C", str(repository), "pull", "--ff-only"],
            check=True,
        )
        print(f"Installing into {python_executable}...")
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
