# CryoSPARC 5.0.6 convention fixture

This is an opt-in live check. It uses a real CryoSPARC Tools session and a
real source volume output; it does not mock the API or report unavailable
servers as passing.

```bash
export CRYOSPARC_RUN_LIVE_FIXTURE=1
export CRYOSPARC_FIXTURE_URL=https://cryosparc.example.org
export CRYOSPARC_FIXTURE_PROJECT=P1
export CRYOSPARC_FIXTURE_WORKSPACE=W1
export CRYOSPARC_FIXTURE_VOLUME_JOB=J200
export CRYOSPARC_FIXTURE_VOLUME_OUTPUT=volume
export CRYOSPARC_FIXTURE_OBSERVATION=/absolute/path/to/cryosparc_5_0_6_observation.json
export CRYOSPARC_FIXTURE_RESULT=tests/live/results/cryosparc_5_0_6.json
uv run cryosparc-506-convention-fixture
```

Authenticate first with the CryoSPARC Tools v5 token flow. The command creates
one supported External Job, verifies the server reports exactly `v5.0.6`,
checks all registered `I` cameras against an independent observation artifact
exported from a CryoSPARC 5.0.6 worker/API or confirmed in the CryoSPARC UI,
and verifies both raw and display arrays plus `flipud(raw)` before registering
the outputs. The JSON result records `passed`, `failed`, or `skipped` and
includes the reason and checks. A missing observation is skipped; it is never
silently generated from this package's own registry.

Use [the observation format template](../../docs/fixtures/cryosparc_5_0_6_observation.template.json)
and replace its placeholders with independently exported camera matrices and
raw/display reference arrays. Array paths are relative to the observation JSON.
The observation must also record the exact project/source volume job/output,
capture method, and SHA-256 of the C-contiguous voxel-array bytes returned by
`cryosparc.mrc.read`; mismatched provenance is a failed validation.

The pytest file is skipped unless `CRYOSPARC_RUN_LIVE_FIXTURE=1` is set.
