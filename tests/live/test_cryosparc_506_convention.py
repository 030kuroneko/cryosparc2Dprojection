import os

import pytest

from cryosparc_2d_projection.live_fixture import (
    LiveFixtureConfig,
    LiveFixtureStatus,
    run_live_convention_fixture,
)


@pytest.mark.skipif(
    os.environ.get("CRYOSPARC_RUN_LIVE_FIXTURE", "").strip().lower()
    not in {"1", "true", "yes", "on"},
    reason="opt-in real CryoSPARC 5.0.6 fixture",
)
def test_cryosparc_506_convention_fixture_is_live_and_complete():
    config = LiveFixtureConfig.from_environment()
    if not config.is_configured:
        pytest.fail(
            "CRYOSPARC_RUN_LIVE_FIXTURE is enabled but fixture settings are "
            + ", ".join(config.missing_fields)
        )

    result = run_live_convention_fixture(config)

    assert result.status == LiveFixtureStatus.PASSED.value, result.reason or result.error
    assert result.executed is True
    assert result.server_version.removeprefix("v") == "5.0.6"
    assert all(
        all(check.values()) for check in (result.camera_checks or {}).values()
    )
    assert all(
        check["vertical_flip_matches_cryosparc"]
        for check in (result.display_checks or {}).values()
    )
