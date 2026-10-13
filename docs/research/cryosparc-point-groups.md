# Point-group coordinate conventions

Class Orientation accepts `Cn`, `Dn` (positive integer n), `T`, `O`, and `I`.
Names are case-insensitive. `I1`, `I2`, helical and reflection groups are unsupported.
Inputs must already use symmetry-aligned map coordinates.

## Coordinates and sources

| Group | Convention used by this tool |
| --- | --- |
| Cn | Principal axis Z |
| Dn | Principal axis Z; dyad Y |
| T | Z threefold; tilted dyad `(0, sqrt(2/3), sqrt(1/3))` |
| O | Fourfold axes X, Y, Z; threefold `(1,1,1)` |
| I | Existing [icosahedral convention](cryosparc-icosahedral-conventions.md) |

CryoSPARC documents the [supported families](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-symmetry-expansion),
[principal-axis alignment](https://discuss.cryosparc.com/t/how-does-cryosparc-align-to-symmetry/476),
and [D/T orientation](https://discuss.cryosparc.com/t/tetrahedral-symmetry-orientation/12482).
The exact T embedding is inferred from those descriptions and
[RELION's generators](https://github.com/3dem/relion/blob/master/src/symmetries.cpp).
O uses the standard embedding in RELION and
[CCTBX](https://github.com/cctbx/cctbx_project/blob/master/mmtbx/ncs/ncs.py).
T/O embeddings are source-based choices, not measured operator exports.

The implementation adapts [SciPy rotation groups](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.create_group.html)
to these coordinates for both direction and full-camera symmetry folding.

## Axis Search

Axis Search accepts the same groups except C1; default symmetry is I.
Families are derived from the declared operators, grouping symmetry-equivalent
axis directions. Opposite poles can belong to separate families.

| Symmetry | Search records |
| --- | --- |
| Cn, n ≥ 2 | nfold, nfold-2 |
| D1 | 2fold, 2fold-2 |
| D2 | 2fold, 2fold-2, 2fold-3 |
| Dn, n > 2 | 2fold, 2fold-2, nfold |
| T | 2fold, 3fold, 3fold-2 |
| O | 2fold, 3fold, 4fold |
| I | 2fold, 3fold, 5fold |

Omit `--axis-family` to search all records. `3fold` selects only that record,
not `3fold-2`; suffixes distinguish direction groups, not symmetry order.
Axis Search does not align arbitrary maps to a symmetry convention.
