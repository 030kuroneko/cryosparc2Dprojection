# Build and publish packages

The **Build and publish packages** GitHub Actions workflow produces a Python
wheel and source distribution, plus Conda packages. It installs each distribution
in a clean environment and starts the CLI outside the checkout before publishing.
A manual run defaults to build/test only; no publishing accounts are needed to
download its artifacts.

## One-time configuration

1. Merge `.github/workflows/packages.yml` and the packaging files into the default
   branch so **Actions → Build and publish packages → Run workflow** appears.
2. In PyPI, create a pending trusted publisher for `cryosparc-2d-projection`:
   owner `030kuroneko`, repository `cryosparc2Dprojection`, workflow `packages.yml`,
   environment `pypi`. Create the matching `pypi` GitHub environment.
3. Create an Anaconda.org account/channel. In the GitHub environment `anaconda`,
   add variable `ANACONDA_USER` with that account name and secret
   `ANACONDA_API_TOKEN` with upload access to that channel. Do not put the token
   in this repository or in chat. Replace `YOUR_CHANNEL` in the installation
   examples once the channel is chosen.

The repository workflow does not create these accounts or publish to
conda-forge. Conda-forge submission is a separate review process. The dependency
`cryosparc-tools` is built from its pinned upstream source and uploaded alongside
this application, because it is not currently available in conda-forge.

## Each release

1. Update `version` in `pyproject.toml` and refresh `uv.lock` with `uv lock`.
2. Push the changes, then run the workflow with **publish unchecked** to inspect
   the results. Both build jobs must pass.
3. Publish a GitHub Release tagged `v<version>` (for example `v0.1.0`). This builds,
   tests and publishes to both package indexes. Alternatively, manually run the
   workflow with **publish checked** on the intended branch.

PyPI versions are immutable: increment the version for a new release. A rerun
of an already-published version may fail on PyPI; Conda uploads skip existing
files. When only a Conda recipe changes, increase its build number. Check both
publish jobs, as one index can succeed while the other fails.

## Installation after publication

```bash
uv tool install --python 3.12 cryosparc-2d-projection
```

```bash
conda create -n cryosparc2d --override-channels -c YOUR_CHANNEL -c conda-forge cryosparc-2d-projection
conda activate cryosparc2d
```

The Conda build targets Linux x86-64 and macOS ARM64, Python 3.12. Other platforms
are not covered by this workflow. Basic installation does not require sudo.
Web dependencies are optional on PyPI (`cryosparc-2d-projection[web]`); Conda users
can add `flask` and `waitress` from conda-forge. The managed systemd installer
still needs uv and a source checkout passed with `--source`; package installation
alone does not install or configure a system service.

## References

- [GitHub manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- [Conda recipe metadata](https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html)
- [Anaconda package installation](https://www.anaconda.com/docs/anaconda-org/installing-packages)
