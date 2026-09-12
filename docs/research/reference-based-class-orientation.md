# Reference-based Class Camera Orientation fallback

Research and accepted design, 2026-09-12. Implementation is local; live
CryoSPARC and physical CUDA validation remain pending.

## Scope and decision

Keep Select 2D Classes + Refinement as the input combination. Use existing
particle-pose search for classes with finite overlapping 2D/3D poses. For each
other selected class, estimate a camera directly from the class image and the
same Matching Map. No reference generation and no low-score-triggered recovery
are included. See [ADR 0020](../adr/0020-add-image-only-class-camera-fallback.md).

The user initially approved investigating native Auto Select outputs, then chose
direct implementation after comparing the two integration paths. The native
pose export contract was not established and is not a dependency of this design.

## Primary sources

- [CryoSPARC Reference Based Auto Select 2D guide](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/particle-curation/job-reference-based-auto-select-2d-beta): class averages and a reference volume are required, particles are optional; the job aligns class images to a volume and scores the corresponding projections. Its documented outputs do not specify a reusable complete-pose field.
- [Official CryoSPARC tools session API](https://github.com/cryoem-uoft/cryosparc-tools/blob/main/cryosparc/tools.py) and [job controller](https://github.com/cryoem-uoft/cryosparc-tools/blob/main/cryosparc/controllers/job.py): a native integration would discover the server's registry, create/connect/queue a separate job, and read its outputs. The local 5.0.3 SDK does not bundle the native Auto Select algorithm or job-specific pose schema.
- [CryoSPARC symmetry relaxation](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-symmetry-relaxation): distinguishes symmetry-related poses and the significance of asymmetric features. Similar images alone do not prove symmetry equivalence.
- [CuPy affine transform](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.affine_transform.html) and [installation](https://docs.cupy.dev/en/stable/install.html): support a CUDA implementation of the spatial projection path. This establishes feasibility, not a measured speedup.

Sources accessed 2026-09-12. The implementation does not claim algorithmic
equivalence to CryoSPARC Reference Based Auto Select 2D.

## Implementation contract

`solve_class_cameras_from_images` is the image-only numerical boundary;
`run_external_orientation_job` publishes the complete mixed-source class result
set. Existing native-grid rendering remains independent of camera selection.

| Setting | Standard | Fine |
| --- | --- | --- |
| Maximum selection box | 32 pixels | 48 pixels |
| Coarse Euler spacing | 20° plus cardinal angles | 12° plus cardinal angles |
| Distinct coarse seeds | 8 | 12 |
| Local increments | 10°, 5°, 2°, 1° | 6°, 3°, 1°, 0.5° |

Both settings cover the full viewing sphere and in-plane rotations; poles avoid
redundant azimuth enumeration. Coarse projections are shared across classes.
Each local increment runs two 3-axis neighborhood sweeps. These are practical
sampling presets, not a guarantee of global optimality or angular accuracy.

Search uses a cosine-edged circular mask (full weight to 40% of box width, zero
at 50%), weighted zero-mean normalized cross-correlation and bounded integer XY
shifts up to 10% of selection box width. Density normalization prevents absolute
density scale from determining validity. Downsampling uses anti-alias smoothing
and preserves the physical center. This search score is distinct from the
existing diagnostic band-limited score; diagnostic settings do not change it.

Full camera rotations are compared modulo the declared proper symmetry
operators. Final candidates within 5° under that distance are consolidated;
other refined groups are preserved. Confidence is heuristic: score at least 0.5
and a measured distinct-group margin at least 0.03. A missing competitor does
not establish confidence. Particle-derived spread and direction are null for
fallback classes, and usable overlapping particle count is zero. Camera metadata
contains the estimated direction separately. Blank or invalid images fail
explicitly rather than inventing an orientation.

The actual selection projections and their pixel shifts are preserved on the
bounded selection grid, with its physical pixel size recorded. Mixed pose/image
search grids are published in separate `search_projections` /
`search_projections_NNN` stacks. Every class records its output name and zero-based
index; no replacement reprojection is substituted for the scored image.
The selection box/shift units are recorded separately from the input search
grid and native rendered grid. Native reprojection still performs its existing
translation alignment without changing the selected rotation.

## GPU and scheduling

`auto` tries CuPy/CUDA and reports CPU execution when unavailable. `cpu` forces
CPU. `cuda` requires a GPU at startup. GPU allocation failure halves the batch
until one projection remains; continued failure restarts all image-only classes
on CPU. Other runtime failures propagate. Initialization memory failures that
cannot be addressed by projection batching also restart on CPU.

The CUDA 12 extra is optional and Linux-only. One GPU is used, honoring the
worker's visible CUDA devices. Slurm profiles/admin settings accept `gpus: 0`
or `gpus: 1`, with zero preserving the prior CPU-only resource request.

## Validation

- Independent array-axis projections test full camera recovery without poses.
- Analytic Gaussian projections test an off-grid camera and XY shift.
- Mixed and zero-overlap External Jobs test preservation of all selected classes;
  non-finite particle poses also take the fallback path.
- C2-equivalent candidates are consolidated before ambiguity is assessed;
  indistinguishable non-equivalent views remain low-confidence.
- CPU stand-ins for the external CUDA API exercise batch reduction, CPU restart
  and propagation of other failures. They are **not** CUDA numerical validation.
- A separate hardware-dependent CPU/CUDA comparison skips when CUDA is absent.
- Launcher and Slurm tests verify argument propagation and GPU resource requests.

No live server jobs were created or run during this work. Real-data projection
comparison, target-version convention checks and GPU throughput/accuracy remain
unverified. Local timing artifacts, when present, only describe their stated
synthetic inputs and host.

Final local verification: 541 tests passed, one hardware CUDA test skipped
(`uv run pytest -q`). Regression coverage includes actual selection-grid output,
mixed-grid publication and CUDA discovery error propagation. Both Standards and
Spec reviews of `f14e707...3b03dd5` reported no remaining actionable findings.
The generated fallback three-column preview was visually checked for readable
labels and aligned image panels.

[Local timing record](../fixtures/results/image_camera_local.json): on this
arm64 macOS host, one synthetic 17-pixel identity-camera class took 1.651 s with
standard search (5,368 evaluations) and 4.853 s with fine search (18,016
evaluations). These tiny CPU cases do not predict real-data or GPU throughput.
