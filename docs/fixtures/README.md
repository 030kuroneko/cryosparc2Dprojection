# Live fixture observations

`cryosparc_5_0_6_observation.template.json` is intentionally not executable
evidence: its placeholders must be replaced with arrays and camera matrices
exported independently from a CryoSPARC 5.0.6 worker/API or confirmed in the
CryoSPARC UI. The live fixture rejects placeholders, missing arrays, and
`independent: false`; it records `skipped` or `failed` rather than claiming a
pass without observations. The artifact must also bind the exact project UID,
source volume job/output, capture method, and SHA-256 of the C-contiguous
voxel-array bytes returned by `cryosparc.mrc.read`; the live run compares all
of these values.
