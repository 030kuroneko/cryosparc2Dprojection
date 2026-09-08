# Point-group coordinate conventions

The Class Orientation workflow accepts `Cn`, `Dn` (positive integer `n`, including
`C1` and `D1`), `T`, `O`, and the existing `I`. Names are case-insensitive;
surrounding whitespace is ignored. `I1`, `I2`, reflection groups, helical symmetry,
zero orders, and leading-zero orders are rejected.

## Sources and implementation

CryoSPARC lists these five point-group families in its
[Symmetry Expansion documentation](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-symmetry-expansion).
Inputs must already use the refinement's symmetry-aligned map coordinates.

- `Cn`: the principal axis is Z, as explained by CryoSPARC in its
  [symmetry-alignment discussion](https://discuss.cryosparc.com/t/how-does-cryosparc-align-to-symmetry/476).
- `Dn`: the principal axis is Z and the dyad is Y. CryoSPARC staff explicitly
  distinguish this from RELION's X dyad in the
  [tetrahedral-orientation discussion](https://discuss.cryosparc.com/t/tetrahedral-symmetry-orientation/12482).
  The adapter conjugates SciPy's Z-axis dihedral operators by a 90-degree Z
  rotation. `D1` therefore contains identity and the Y half-turn. Odd orders such
  as D7 distinguish this embedding from the unconverted SciPy group.
- `T`: CryoSPARC staff confirm a Z threefold and describe the convention as
  consistent with RELION in the same discussion. The exact embedding here is an
  **inference** using the generators in
  [RELION's symmetry definitions](https://github.com/3dem/relion/blob/master/src/symmetries.cpp):
  a Z threefold and a half-turn about `(0, sqrt(2/3), sqrt(1/3))`.
  It also agrees with the dyad in Phenix's `T (c)` operator set
  ([CCTBX source, `tetrahedral_b`](https://github.com/cctbx/cctbx_project/blob/master/mmtbx/ncs/ncs.py)).
  The adapter rotates SciPy's diagonal threefold onto Z, and its X dyad onto that
  tilted axis; it does not pass through SciPy's default T embedding.
- `O`: uses fourfold axes on X, Y, Z, and a threefold on `(1,1,1)`, the standard
  embedding also present in RELION's definitions and Phenix's `O (a)` set.
  This is the selected convention; a full operator export from the installed
  CryoSPARC version remains necessary to verify it independently.
- `I`: unchanged; see [the existing icosahedral research](cryosparc-icosahedral-conventions.md).

[SciPy's group API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.create_group.html)
provides the finite proper-rotation groups. The shared adapter is used by both
View Direction folding and complete Class Camera Orientation folding.

## Validation and limits

Local tests cover group cardinalities, unique proper rotations, closure,
principal axes, T/O vertex preservation, independent generator mates with
non-identity cameras, direction folding, input validation, launcher submission,
and External Job result metadata. These are mathematical and local workflow
checks, **not a live CryoSPARC integration pass**. The T azimuth and O embedding
are source-based choices rather than measurements from an installed worker.

Before declaring installed-version convention validation complete, export all
operators from that CryoSPARC worker for C3, D1, D7, T, O, and I, and compare full
sets against `symmetry_operators`. Then compare symmetry-expanded non-identity
particle poses/projections against the tool's `camera @ operator` folding rule.
The existing live fixture result is skipped and does not establish these facts.


## Axis Search

Axis Search uses the same operator adapter for `Cn` (`n >= 2`), `Dn`, `T`, `O`,
and `I`. C1 has no non-identity rotations, hence no nontrivial symmetry axes;
use Class Orientation instead. The default remains I.

Axis coverage is **derived mathematically from the declared operator sets**,
not from a live CryoSPARC run. Extract both poles of every non-identity rotation
axis and partition them under the action of the proper-rotation group. Each
directed orbit becomes one search record. The number of operators fixing its
direction determines the maximal order. Thus an O fourfold axis is not also
reported as an independent twofold family. See the
[SciPy group API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.create_group.html)
and the coordinate conversions above.

Names sort by order, then decreasing representative `(z, y, x)` (rounded for
stable ordering). The first orbit is `Nfold`; additional orbits are `Nfold-2`,
`Nfold-3`, etc. The suffix is an orbit identifier, not a change in order. Blank
family selection searches all records; `3fold` selects only that record.

| Symmetry | Search records |
| --- | --- |
| Cn, n >= 2 | nfold, nfold-2 (opposite poles) |
| D1 | 2fold, 2fold-2 (opposite Y poles) |
| D2 | 2fold, 2fold-2, 2fold-3 (Z, Y, X axes) |
| Dn, n > 2 | 2fold, 2fold-2, nfold |
| T | 2fold, 3fold, 3fold-2 |
| O | 2fold, 3fold, 4fold |
| I | 2fold, 3fold, 5fold (existing cameras unchanged) |

For odd Dn the dyad records cover two directed orbits of the same undirected
axes; for even Dn they cover two inequivalent sets of undirected axes. New
cameras fix display roll by projecting Cartesian X onto the image plane (Y
when X is nearly parallel to the view) and placing it horizontally. I retains
its existing cross-family presentation rule.

Local tests cover all rotation-axis poles, disjoint orbits, valid roll periods,
proper cameras, ranking, near-axis refinement, pre-submission validation, UI
submission, and External Job metadata. Inputs must already be symmetry-aligned;
Axis Search does not align arbitrary maps. Live convention validation, including
the inferred T/O embeddings, remains pending.
