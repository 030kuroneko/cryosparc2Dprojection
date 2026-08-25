# cryosparc2Dprojection

Create a CryoSPARC **v5.0.6 External Job** that maps Select 2D classes to reproducible Class Camera Orientations using particle poses from Non-uniform (NU) or Local Refinement.

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
9. displays the class average, Matched Projection, and Camera View Render in
   pages of ten classes.

## Compatibility

- CryoSPARC server: **v5.0.6**
- CryoSPARC Tools: `~=5.0.0`, following the official minor-version matching rule
- Python: 3.10–3.12
- Refinement source: NU Refinement or Local Refinement
- Symmetry: `C1` or CryoSPARC `I`

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
--render-size INTEGER          Camera View Render PNG size; default: automatic
--render-grid-size INTEGER     Optional maximum surface grid; default: native map
--comparison-dpi INTEGER       All three-column Class Results; default: 100
--preview-page-size INTEGER    Classes per CryoSPARC preview page; default: 10
```

When `--render-size` is omitted, its effective value is
`max(1024, 3 * comparison_dpi)`. An explicit smaller value is respected with a
warning because the third column may be upscaled. DPI has no hard maximum;
values above 600 report the estimated page dimensions and RGBA memory in both
the terminal and CryoSPARC Job Log.

When `--render-grid-size` is omitted, surface extraction uses the complete
native grid of the selected Rendering Map. An explicit value of at least 2
reduces the grid proportionally without upsampling or distorting non-cubic
maps; it has no fixed software maximum. The terminal, Job Log, and JSON record
the original and effective shapes, downsampling state, and a lower-bound memory
estimate. Estimates above 1 GiB warn but continue. A memory failure never
silently lowers quality and reports an explicit smaller retry value.

For a high-quality export with one class per Event Log page:

```bash
--comparison-dpi 600 --preview-page-size 1
```

This changes only Class Result raster presentation. It does not alter the Class
Average, Search Projection, native-grid Matched Projection, Surface Sampling
Grid, camera search, or matching scores.

Camera selection and the raw search score use a bounded Search Projection of
at most 128 pixels per side. After selecting the camera, the job regenerates a
Matched Projection from the unsharpened Matching Map at the native Class
Average box and pixel size and re-optimizes only XY translation. The
Diagnostic Band-Limited Score is calculated from this native displayed pair.

The comparison preview also reports a Diagnostic Band-Limited Score. Its
defaults can be overridden without changing camera selection:

```text
--diagnostic-low-resolution-A FLOAT        Default: 80
--diagnostic-high-resolution-A FLOAT       Default: 15
--diagnostic-mask-radius-fraction FLOAT    Default: 0.45 of box width
--diagnostic-mask-edge-fraction FLOAT      Default: 0.05 of box width
```

The effective frequency limits are clipped to the matching box extent and
Nyquist resolution and recorded in `class_orientations.json`.

To adjust the 3D density threshold explicitly, pass the raw Surface Level used
by the CryoSPARC Volume Viewer, for example:

```bash
--surface-level 0.12
```

Omit this option to keep automatic selection at mean + 1.5 sigma with fallback
to lower usable levels.

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

- `matched_projections`: native Class Average-grid projections used in Class
  Results;
- `search_projections`: bounded projections that produced the raw camera-search
  scores;
- `rendering_map`: the selected unsharpened or sharpened Rendering Map in the
  standard CryoSPARC `map` slot;
- `class_NNN_volume`: camera-rotated volumes requested with `--classes`;
- `class_orientations.json`: full rotation matrix, quaternion, direction, roll,
  shift, raw match score, Diagnostic Band-Limited Score with reproducibility
  metadata, presentation resolution, warnings, confidence, angular spread, and
  Surface Level;
- `class_projections.mrcs`: one native-grid matched projection per class;
- `search_projections.mrcs`: one bounded scored projection per class;
- `renders/class_NNN_exact.png`: exact orthographic Camera View Render;
- `renders/class_NNN_comparison.png`: three-column Class Result;
- `chimerax/all_classes.cxc` and one ChimeraX script per class;
- job-log preview pages: configurable three-column Class Results per page.

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
- `match_score` remains the raw score used to select the camera from bounded
  Search Projections. The separately reported Diagnostic Band-Limited Score
  checks the native-grid Class Average and Matched Projection inside a physical
  frequency band and soft circular mask; it does not rerank cameras, define a
  second-best margin, or represent a probability.
- Symmetry folding prevents equivalent directions from cancelling during averaging. Always pass the symmetry used by refinement.
- Version 0.1 rejects symmetry conventions other than `C1` and CryoSPARC `I`.
  Support for `C<n>`, `D<n>`, and `T` is deferred until convention integration
  tests are available.
- Static Class Results use the same vertical display orientation as the
  CryoSPARC UI. Matching arrays, registered MRCS data, scores, and Camera
  Metadata are not flipped.
- The text-free Camera View Render uses the solved orthographic camera.
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
