# Image-only Class Orientation

Classes with usable overlapping particle poses use pose-based search. Other
selected classes are matched directly to the same reference map. No Auto Select
job is required; a low score alone does not trigger fallback.

## Search and confidence

| Setting | Standard | Fine |
| --- | --- | --- |
| Maximum search box | 32 pixels | 48 pixels |
| Coarse angle spacing | 20° | 12° |
| Coarse seeds | 8 | 12 |
| Local increments | 10°, 5°, 2°, 1° | 6°, 3°, 1°, 0.5° |

Search samples viewing directions and in-plane rotations, then refines candidates
using masked normalized correlation and bounded XY shifts. These presets do not
guarantee a global optimum or a particular angular accuracy.

Candidates are compared under the declared symmetry; cameras within 5° are
grouped. Confidence requires a score of at least 0.5 and a distinct-group margin
of at least 0.03. It is a heuristic, not a probability. No competitor does not
establish confidence; particle-derived spread is unavailable for fallback classes.

Scored Search Projections are saved on their search grids. Native-grid Matched
Projections are regenerated at the chosen rotation and aligned in translation.
The diagnostic band-limited score does not change camera selection.

## GPU

`--fallback-device auto` uses available CUDA or CPU; `cpu` forces CPU and `cuda`
requires a GPU at startup. Memory failures reduce batches, then restart on CPU;
other computation errors remain visible. One GPU is used.

For installation and Slurm settings, see [GPU setup](../usage.md#image-only-fallback-and-gpu).
GPU support does not imply a measured speedup.

## Sources

- [CryoSPARC Reference Based Auto Select 2D](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/particle-curation/job-reference-based-auto-select-2d-beta): reference-based class matching; this tool does not claim algorithmic equivalence.
- [CryoSPARC symmetry relaxation](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-symmetry-relaxation): symmetry-related poses and asymmetric features.
- [CuPy affine transform](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.affine_transform.html): spatial transformation support used by the GPU path.
- [Coordinate conventions](cryosparc-point-groups.md): supported groups and source-based embeddings.
