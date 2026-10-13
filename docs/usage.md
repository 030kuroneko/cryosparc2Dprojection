# Usage reference

[Installation](../README.md#install) · [Web deployment](web-launcher.md)

Examples use uv. With Conda, omit `uv run`. Replace URLs and job IDs with your own.

## Sign in

Save a CryoSPARC login token for CLI use:

```bash
uv run python -m cryosparc.tools login --url https://cryosparc.example.org
```

## Run with NU Refinement

Match Select 2D classes to a refinement map:

```bash
uv run cryosparc2d-projection \
  --url https://cryosparc.example.org \
  --project P1 --workspace W1 \
  --select-job J100 --refinement-job J200 --symmetry I
```

Defaults: `particles_selected` and `templates_selected` from Select 2D;
`particles` and `volume` from refinement. Use the refinement's symmetry.

## Run with Local Refinement

Use the Local Refinement job as `--refinement-job`. If needed, change
`--refinement-particles-output` and `--volume-output` to its output names.

## Symmetry-Axis Class Search

Find classes near symmetry axes using templates and a volume; no particle poses are needed:

```bash
uv run cryosparc2d-axis-search \
  --url https://cryosparc.example.org \
  --project P1 --workspace W1 \
  --select-job J100 --select-output templates_selected \
  --volume-job J200 --volume-output volume --symmetry I
```

- `--axis-family 2fold,5fold`: search selected families; omit to search all.
- `--refine-near-axis`: add local refinement around each exact axis.
- `--axis-roll 2fold=10`: rotate presentation only; repeat for other families.

C1 is unsupported. See [axis families and coordinates](research/cryosparc-point-groups.md#axis-search).

## Select and export

After a Web Class Orientation run, use **Select classes**, then **Export selection**.
Choices save automatically. At least one class is required.

A new External Job contains `particles_selected`, `templates_selected`,
`particles_excluded`, and `templates_excluded`. Selecting all produces empty
excluded datasets. Original class numbers, fields and source images are preserved.

This panel does not apply to Axis Search or older runs without selection artifacts.
CLI selection/export commands are not implemented. Failed exports can retry the
recorded job; uncertain job creation requires administrator reconciliation before retrying.

## Common options

| Option | Purpose |
| --- | --- |
| `--render-map sharpened` | Use the sharpened map for rendering only |
| `--surface-level FLOAT` | Set the raw density contour |
| `--render-background dark\|light` | Choose render background |
| `--render-size INTEGER` | Set Camera View PNG size |
| `--render-grid-size INTEGER` | Limit surface grid size |
| `--comparison-dpi INTEGER` | Set preview resolution; default 100 |
| `--preview-page-size INTEGER` | Set classes per preview page; default 10 |
| `--auto-crop-2d` | Frame 2D panels for display only |

Class Orientation also accepts `--classes 1,3,8` to create extra camera-rotated
volumes. It still analyzes every class; this option does not select particles for export.

Use either workflow's `--help` for all options. Presentation settings do not change
scientific arrays or camera selection. Matching uses the unsharpened map.

## Image-only fallback and GPU

Classes without usable overlapping poses use image-only matching automatically.
`--fallback-quality fine` increases search density. `--fallback-device` accepts
`auto`, `cpu`, or `cuda`; `--fallback-batch-size` defaults to 16.

Install optional CUDA 12 support on a Linux worker:

```bash
uv sync --extra gpu
```

Add `--extra web` if using Web. Other CUDA versions require the appropriate
[CuPy package](https://docs.cupy.dev/en/stable/install.html).
For Slurm, select a GPU lane with **GPUs = 1** and CuPy installed in its Python environment.

`auto` uses an available GPU or CPU. GPU memory failures reduce the batch, then
retry on CPU. See [method and confidence](research/reference-based-class-orientation.md).

## Outputs and interpretation

- **Class Orientation**: Matched/Search Projections, comparison images,
  `class_orientations.json`, rendering map, and ChimeraX scripts.
- **Axis Search**: ranked classes, Exact-Axis projections, preview pages,
  `axis_search_results.json`, and optional Near-Axis projections.
- Outputs and previews appear in the selected CryoSPARC workspace as External Jobs.

Class numbers match the UI (one-based); JSON also contains zero-based `class_id`.
Pose matching requires shared particle UID lineage. Only usable overlapping poses
contribute; other classes use image-only fallback.

Search scores select cameras on a bounded grid. The Diagnostic Band-Limited Score
checks the native displayed pair and does not rerank cameras. Confidence is not a
probability. Use symmetry-aligned maps and the correct [coordinate convention](research/cryosparc-point-groups.md).

## Update an editable installation

Update a Git-based installation:

```bash
cryosparc2d-update
```

Requires a branch with an upstream and no tracked local changes. It updates the
checkout and dependencies; failures are not automatically rolled back. The managed
Web service is separate and is not updated or restarted by this command.
