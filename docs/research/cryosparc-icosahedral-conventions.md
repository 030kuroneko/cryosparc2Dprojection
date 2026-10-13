# CryoSPARC icosahedral coordinate conventions

Research target: CryoSPARC 5.0.6, with SciPy 1.18.0 as locked for Python 3.12 in this repository.

## Decision summary

- Advertise **`I` only** for now. CryoSPARC's current public job documentation names `I`, not `I1` or `I2`, as the supported icosahedral string. [`I1` and `I2` exist in older CryoSPARC implementation evidence](https://discuss.cryosparc.com/t/illegal-memory-access-with-icosahedral-symmetry-ab-initio/2804/2), but they are not a documented CryoSPARC 5.0.6 API contract.
- `I` and `I1` are exact aliases in that implementation evidence. `I2` is the same abstract 60-element rotation group in a different Cartesian embedding.
- `scipy.spatial.transform.Rotation.create_group("I")` matches CryoSPARC's `I`/`I1` embedding, **not** its `I2` embedding. This is an inference from the two projects' published generators, confirmed numerically; neither project states the equivalence directly.
- Remove the current `I1`/`I2` → SciPy `I` aliasing until `I2` is explicitly converted and covered by convention tests. Pose/symmetry multiplication order also needs a CryoSPARC integration test because the public sources do not fully specify it.

## 1. Accepted strings and axis definitions

CryoSPARC's current public refinement documentation lists the allowed point groups as `Cn`, `Dn`, `T`, `O`, and `I`; Symmetry Expansion and Volume Alignment Tools likewise show examples using `I`. None of these public pages advertises `I1` or `I2`. ([Homogeneous Refinement](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/3d-refinement/job-homogeneous-refinement), [Symmetry Expansion](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-symmetry-expansion), [Volume Alignment Tools](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-volume-alignment-tools))

The most authoritative public axis definitions found are from CryoSPARC co-founder Ali Punjani, quoting CryoSPARC's symmetry definitions: ([source](https://discuss.cryosparc.com/t/illegal-memory-access-with-icosahedral-symmetry-ab-initio/2804/2))

| String | 2-fold axis | 5-fold axis | 3-fold axis |
|---|---|---|---|
| `I` | `(0, 1, 0)` | `(-0.85065080702670, 0, 0.5257311142635)` | `(-0.9341723640, 0.3568220765, 0)` |
| `I1` | identical to `I` | identical to `I` | identical to `I` |
| `I2` | `(0, 0, 1)` | `(-1.618033989, -1, 0)` | `(-0.53934467, -1.4120227, 0)` |

Therefore `I` and `I1` are aliases, while `I2` has a distinct coordinate embedding. The official Guide confirms that CryoSPARC expects volumes to be aligned to its conventional symmetry axes, but does not publish those vectors. ([Symmetry Expansion](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/utilities/job-symmetry-expansion))

### Exact `I` → `I2` transform

**Inference from the published axes:** the active rotation

\[
Q = R_x(+90°) =
\begin{bmatrix}
1&0&0\\
0&0&-1\\
0&1&0
\end{bmatrix}
\]

maps the `I` 2-fold and 5-fold axes onto their `I2` counterparts. Conjugating every `I` operator as `Q G Qᵀ` reproduces the `I2` operator set (within the precision of the published decimal axes). This is a global coordinate change, consistent with Punjani's statement that the definitions correspond to global rotations, but the exact matrix above is derived rather than directly documented. Before using it in production, lock it down against operators emitted by the installed CryoSPARC 5.0.6 worker.

## 2. Which CryoSPARC convention SciPy uses

SciPy accepts only `I`, `O`, `T`, `Dn`, and `Cn`; it has no `I1` or `I2` selector. ([SciPy 1.18.0 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.create_group.html)) Its `I` group is constructed from a fixed set of scalar-last quaternions using the golden ratio. ([SciPy 1.18.0 source](https://github.com/scipy/scipy/blob/54ef5423f2e4376230ec3bfda6912a07a50958e3/scipy/spatial/transform/_rotation_groups.py#L5-L55))

**Inference, numerically checked from those quaternions:** SciPy's set contains rotations about all three published CryoSPARC `I`/`I1` generator axes at the specified orders (residuals below `1.3×10⁻⁸` using the rounded CryoSPARC axes). The `I2` 5-fold and 3-fold generators are not members of the untransformed SciPy set. Thus direct `Rotation.create_group("I")` is appropriate for CryoSPARC `I`/`I1`, while `I2` requires conjugation by `Q`.

## 3. Pose representation and composition gap

CryoSPARC stores `alignments3D/pose` as an axis-angle/rotation vector: direction is the rotation axis and vector norm is the angle in radians. A zero vector maps to identity. CryoSPARC staff also state that `alignments3D/{pose,shift}` are the actual alignments used for backprojection, including Local Refinement outputs. ([pose definition](https://discuss.cryosparc.com/t/cs-file-slot-field-explanation/17474/6), [zero/identity](https://discuss.cryosparc.com/t/null-rodrigues-vectors-question/11361/2), [Local Refinement fields](https://discuss.cryosparc.com/t/clarification-of-final-pose-for-local-refinement-jobs/6758/2))

pyem converts CryoSPARC pose vectors with its `expmap`, then converts the resulting matrix to RELION Euler angles. ([conversion call](https://github.com/asarnow/pyem/blob/8bca98395f987b6d77708126141455013cef0517/pyem/metadata/cryosparc2.py#L366-L372), [`expmap` implementation](https://github.com/asarnow/pyem/blob/8bca98395f987b6d77708126141455013cef0517/pyem/geom/convert.py#L157-L168)) This is useful first-party evidence for pyem behavior, but it does not define how CryoSPARC composes a pose with a symmetry operator.

CryoSPARC's Symmetry Relaxation guide says symmetry-related pose angles are computed analytically and evaluated, but does not state whether the operator is left- or right-multiplied, nor whether matrices are active or passive in this context. ([Symmetry Relaxation](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-symmetry-relaxation)) Therefore the repository's present `camera @ symmetry_operator` rule is **unverified**, not a sourced CryoSPARC convention.

## 4. Required tests before advertising `I1`/`I2`

1. **Installed-version acceptance test:** on CryoSPARC 5.0.6, ask the worker/API to generate operators for `I`, `I1`, and `I2`; record accepted strings and all 60 matrices. Public docs alone only justify `I`.
2. **Operator-set test:** show CryoSPARC `I == I1`; show `I2 == {Q G Qᵀ | G in I}` with a strict tolerance and determinant/order checks.
3. **SciPy bridge test:** show the 60 SciPy `I` matrices equal CryoSPARC `I`; implement an explicit `I2` adapter rather than aliasing the string.
4. **Composition integration test:** select a non-symmetric synthetic volume and one known pose; compare this tool's projections for both candidate compositions (`P G` and `G P`, including transpose alternatives) with CryoSPARC-generated symmetry-expanded particles/projections. This establishes multiplication side and active/passive convention.
5. **End-to-end convention test:** rotate the same synthetic asymmetric marker volume into `I` and `I2` coordinates, then require identical recovered physical camera views after conversion.

Until these pass, the safe public claim is: **CryoSPARC 5.0.6 `I` convention only; `I1`/`I2` unsupported**.
