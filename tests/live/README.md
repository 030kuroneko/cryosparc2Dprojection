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
export CRYOSPARC_FIXTURE_RESULT=tests/live/results/cryosparc_5_0_6.json
uv run cryosparc-506-convention-fixture
```

Authenticate first with the CryoSPARC Tools v5 token flow. The command creates
one supported External Job, verifies the server reports exactly `v5.0.6`,
checks all registered `I` cameras, generates raw projections directly from the
selected volume, and verifies `display == flipud(raw)` before registering both
output stacks. The JSON result records `passed`, `failed`, or `skipped` and
includes the reason and checks.

The pytest file is skipped unless `CRYOSPARC_RUN_LIVE_FIXTURE=1` is set.
