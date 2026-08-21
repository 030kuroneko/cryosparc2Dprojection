# cryosparc2Dprojection

Create a CryoSPARC **v5.0.6 External Job** that maps Select 2D classes to 3D viewing directions using particle poses from Non-uniform (NU) or Local Refinement.

The job:

1. matches Select 2D and refinement particles by CryoSPARC `uid`;
2. uses only particles present in both datasets;
3. converts `alignments3D/pose` Rodrigues vectors using the same convention as pyem;
4. folds symmetry-equivalent directions before averaging;
5. folds complete symmetry-equivalent camera rotations before averaging;
6. refines the viewing direction and in-plane rotation against the actual selected
   class average;
7. resamples the class and map to a common physical grid (at most 128 pixels);
8. extracts a solid triangular isosurface from the Rendering Map;
9. displays the class average, Matched Projection, Camera View Render, and
   Oblique Inspection Render in pages of ten classes.

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

Or in the already-created conda environment:

```bash
python -m pip install -e .
python -m pytest -q
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

With the P1/W9 test data and the conda installation:

```bash
cryosparc-2d-projection \
  --url http://localhost:39000 \
  --project P1 \
  --workspace W9 \
  --select-job J1025 \
  --templates-output templates_selected \
  --select-output particles_selected \
  --refinement-job J1083 \
  --refinement-particles-output particles \
  --volume-output volume \
  --symmetry I \
  --classes 1,13,26 \
  --render-map map
```

`--classes` is optional. The job analyzes every selected class either way; this
option only creates extra rotated CryoSPARC Volume Viewer outputs for the listed
one-based class numbers.

Use the sharpened result only for visualization with:

```bash
--render-map sharpened
```

Camera matching still uses the unsharpened `map`. Rendering options are:

```text
--surface-level FLOAT          Raw contour value; default is mean + 1.5 sigma
--render-background dark|light
--oblique-tilt-degrees FLOAT   Default: 20
--render-size INTEGER          Camera View Render and Oblique Inspection Render PNG size; default: 1024
--render-grid-size INTEGER     Maximum surface grid; default: 192
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

- `matched_projections`: a registered CryoSPARC template stack;
- `rendering_map`: the selected unsharpened or sharpened Rendering Map in the
  standard CryoSPARC `map` slot;
- `class_NNN_volume`: camera-rotated volumes requested with `--classes`;
- `class_orientations.json`: full rotation matrix, quaternion, direction, roll,
  shift, match score, confidence, angular spread, and nearest symmetry axis;
- `class_projections.mrcs`: one matched simulated projection per class;
- `renders/class_NNN_exact.png`: exact orthographic Camera View Render;
- `renders/class_NNN_oblique.png`: 20-degree Oblique Inspection Render;
- `renders/class_NNN_comparison.png`: four-column Class Result;
- `chimerax/all_classes.cxc` and one ChimeraX script per class;
- job-log preview pages: ten four-column Class Results per page.

CryoSPARC stores class IDs from zero. The JSON includes both `class_id`
(zero-based) and `class_number` (one-based, matching the UI). Selected classes
keep their original `blob/idx`, including gaps; they are never renumbered.

## Important interpretation

- Select 2D and refinement particles must share CryoSPARC UID lineage. Reordering and subsetting are safe; unrelated datasets are rejected.
- Local Refinement may contain only a subset. Only overlapping particles contribute.
- Refinement particle poses provide the seed. A bounded local projection search
  then refines both viewing direction and in-plane rotation against the actual
  selected class average.
- `match_confidence` remains visible when low; it is not silently discarded.
- Symmetry folding prevents equivalent directions from cancelling during averaging. Always pass the symmetry used by refinement.
- The Camera View Render uses the solved orthographic camera. The Oblique
  Inspection Render is deliberately tilted for depth perception and is not
  another camera solution.
- The automatic Surface Level starts at mean + 1.5 sigma, lowers it when that
  contour is unusable, records the chosen value, and removes density islands
  smaller than 1% of the main component. Use `--surface-level` to reproduce a
  raw contour value chosen in CryoSPARC Volume Viewer.
- CryoSPARC's public External Job API can register the original and rotated
  volumes, but cannot preset the integrated Volume Viewer camera. Use a requested
  `class_NNN_volume` or the generated ChimeraX scripts for the exact camera.

## Development

The repository follows test-driven development:

```bash
uv run pytest -q
```
