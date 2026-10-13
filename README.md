# cryosparc2Dprojection

Match CryoSPARC 2D classes to views of a 3D map and find classes along symmetry axes. Compare class averages, matched projections, and 3D renders. Run from the command line, or use the optional Web browser interface to run jobs, select classes, and export selections to CryoSPARC.

## Install

Clone the repository:

```bash
git clone https://github.com/030kuroneko/cryosparc2Dprojection.git
cd cryosparc2Dprojection
```

Choose **one** method for installation:

**Option 1: uv**

Install the command-line tools:

```bash
uv sync
```

**Optional: install the Web browser interface** to run jobs, select classes with checkboxes, and export them to CryoSPARC:

```bash
uv sync --extra web
```

**Option 2: Conda / Miniforge**

Create an environment and install the command-line tools:

```bash
conda create -n cryosparc2d --override-channels -c conda-forge python=3.12 pip
conda activate cryosparc2d
python -m pip install -e .
```

**Optional: install the Web browser interface** to run jobs, select classes with checkboxes, and export them to CryoSPARC:

```bash
python -m pip install -e '.[web]'
```

## Features

- **Class Orientation**: match classes to map views.
- **Axis Search**: find classes along symmetry axes.
- **Select & Export**: select classes after Class Orientation and export to CryoSPARC using the Web checkbox panel.

## Quick start

From the repository folder, install Web dependencies and start the browser interface (replace the URL with your CryoSPARC server):

```bash
uv run --extra web cryosparc2d --url https://your-cryosparc-server --port 40000
```

Open [localhost:40000](http://127.0.0.1:40000), sign in, and choose a workflow.

With Conda, install the optional Web dependencies above, then omit `uv run --extra web`.

**CLI**: replace the URL, project/workspace IDs, job IDs, and symmetry with your own. With Conda, omit `uv run`.

Sign in before your first CLI run:

```bash
uv run python -m cryosparc.tools login --url https://your-cryosparc-server
```

**Class Orientation**: match Select 2D classes (`J100`) using NU or Local Refinement (`J200`):

```bash
uv run cryosparc2d-projection \
  --url https://your-cryosparc-server \
  --project P1 --workspace W1 \
  --select-job J100 --refinement-job J200 --symmetry I
```

**Axis Search**: find symmetry-axis classes using Select 2D templates (`J100`) and a volume (`J200`):

```bash
uv run cryosparc2d-axis-search \
  --url https://your-cryosparc-server \
  --project P1 --workspace W1 \
  --select-job J100 --volume-job J200 --symmetry I
```

Both commands create an External Job in the specified workspace. See [output settings](docs/usage.md) if your input output names differ. CLI has no checkbox panel; selection/export commands are not yet implemented.

## Compatibility & limits

- CryoSPARC **5.0.6**; Python **3.10–3.12**.
- Class Orientation: Select 2D + NU or Local Refinement.
- Axis Search: 2D templates + volume.

## Documentation

- [Usage reference](docs/usage.md): CLI examples, parameters, outputs, GPU setup, updates, and interpretation.
- [Web deployment](docs/web-launcher.md): authentication, shared service, and Slurm configuration.
- [Configuration example](docs/web-config.example.json): launcher settings.
- [Symmetry conventions](docs/research/cryosparc-point-groups.md): supported groups, coordinates, and sources.
- [Image-only matching](docs/research/reference-based-class-orientation.md): method, confidence, and validation limits.
