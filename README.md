# cryosparc2Dprojection

Create a CryoSPARC **v5.0.6 External Job** that maps Select 2D classes to 3D viewing directions using particle poses from Non-uniform (NU) or Local Refinement.

The job:

1. matches Select 2D and refinement particles by CryoSPARC `uid`;
2. uses only particles present in both datasets;
3. converts `alignments3D/pose` Rodrigues vectors using the same convention as pyem;
4. folds symmetry-equivalent directions before averaging;
5. calculates a representative direction and RMS angular spread for every class;
6. projects the refinement volume from each representative direction;
7. displays a projection preview in the CryoSPARC job log.

## Compatibility

- CryoSPARC server: **v5.0.6**
- CryoSPARC Tools: `~=5.0.0`, following the official minor-version matching rule
- Python: 3.10–3.12
- Refinement source: NU Refinement or Local Refinement
- Symmetry: `C<n>`, `D<n>`, `T`, `O`, `I`, `I1`, or `I2`

This uses CryoSPARC's supported External Job API. Running the command creates a job inside the selected workspace, but does not permanently register a new built-in job type in Job Builder.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Log in using CryoSPARC Tools' v5 token flow. Use the same URL you use in the browser:

```bash
uv run python -m cryosparc.tools login --url https://cryosparc.example.org
```

The token is stored by CryoSPARC Tools. This project does not store your email or password.

## Run with NU Refinement

```bash
uv run cryosparc-2d-projection \
  --url https://cryosparc.example.org \
  --project P1 \
  --workspace W1 \
  --select-job J100 \
  --select-output particles_selected \
  --refinement-job J200 \
  --refinement-particles-output particles \
  --volume-output volume \
  --symmetry I
```

## Run with Local Refinement

Use the Local Refinement job as `--refinement-job`. Change the two output names if that job exposes different names in its CryoSPARC Outputs panel:

```bash
uv run cryosparc-2d-projection \
  --url https://cryosparc.example.org \
  --project P1 \
  --workspace W1 \
  --select-job J100 \
  --refinement-job J300 \
  --refinement-particles-output particles \
  --volume-output volume \
  --symmetry C1
```

## Outputs

The created External Job contains:

- `class_orientations.json`: class number, overlapping particle count, view vector, and angular spread;
- `class_projections.mrcs`: one simulated projection per matched 2D class, preserving the volume pixel size;
- job-log projection preview: up to the first 25 matched classes.

CryoSPARC stores class IDs from zero. The JSON includes both `class_id` (zero-based) and `class_number` (one-based, matching the human-readable class number).

## Important interpretation

- Select 2D and refinement particles must share CryoSPARC UID lineage. Reordering and subsetting are safe; unrelated datasets are rejected.
- Local Refinement may contain only a subset. Only overlapping particles contribute.
- The direction comes from refinement particle poses, not from cross-correlating the 2D class average against an exhaustive projection library.
- The projection uses a canonical in-plane rotation. It shows the correct viewing axis, but is not yet in-plane aligned to the displayed 2D class average.
- Symmetry folding prevents equivalent directions from cancelling during averaging. Always pass the symmetry used by refinement.

## Development

The repository follows test-driven development:

```bash
uv run pytest -q
```
