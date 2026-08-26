import json
from types import SimpleNamespace

import numpy as np

from cryosparc_2d_projection.axis_projection import AxisReferenceProjection
from cryosparc_2d_projection.axis_registry import axis_family_records, get_axis_family
from cryosparc_2d_projection.live_fixture import (
    LiveFixtureConfig,
    LiveFixtureStatus,
    record_fixture_result,
    run_live_convention_fixture,
    validate_live_convention,
)


_CAMERAS = {
    family.name: family.canonical_camera_matrix
    for family in axis_family_records("I")
}
_RAW = {
    "2fold": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    "3fold": np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float32),
    "5fold": np.array([[4.0, 1.0], [3.0, 2.0]], dtype=np.float32),
}


def _references():
    return [
        AxisReferenceProjection(
            family=family,
            projection=_RAW[family.name],
            display_projection=np.flipud(_RAW[family.name]),
            rotation_matrix=_CAMERAS[family.name],
            pixel_size_A=1.0,
        )
        for family in axis_family_records("I")
    ]


def test_live_fixture_configuration_requires_only_connection_and_volume_settings():
    config = LiveFixtureConfig.from_environment(
        {
            "CRYOSPARC_FIXTURE_URL": "https://cryosparc.example.test",
            "CRYOSPARC_FIXTURE_PROJECT": "P1",
            "CRYOSPARC_FIXTURE_WORKSPACE": "W1",
            "CRYOSPARC_FIXTURE_VOLUME_JOB": "J1",
        }
    )

    assert config.is_configured is True
    assert config.missing_fields == ()


def test_live_fixture_without_connection_settings_is_explicitly_skipped(tmp_path):
    config = LiveFixtureConfig.from_environment({})

    assert config.is_configured is False
    result_path = tmp_path / "cryosparc-5.0.6-convention.json"
    result = record_fixture_result(
        result_path,
        status=LiveFixtureStatus.SKIPPED,
        config=config,
        reason="missing CRYOSPARC_FIXTURE_URL",
    )

    assert result.status == "skipped"
    recorded = json.loads(result_path.read_text())
    assert recorded["schema"] == "cryosparc-5.0.6-convention-fixture/v1"
    assert recorded["status"] == "skipped"
    assert recorded["executed"] is False
    assert recorded["server_version"] is None


def test_live_fixture_validation_checks_server_camera_and_display_contract():
    validation = validate_live_convention("v5.0.6", _references())

    assert validation.passed is True
    assert validation.server_version_matches is True
    assert validation.camera_checks["2fold"]["orthogonal"] is True
    assert validation.camera_checks["2fold"]["determinant_plus_one"] is True
    assert validation.camera_checks["2fold"]["third_row_matches_direction"] is True
    assert validation.display_checks["2fold"]["vertical_flip_matches_cryosparc"] is True


def test_live_fixture_validation_reports_mismatch_without_claiming_pass():
    raw = _RAW["2fold"]
    corrupted = AxisReferenceProjection(
        family=get_axis_family("I", "2fold"),
        projection=raw,
        display_projection=np.fliplr(raw),
        rotation_matrix=np.eye(3),
        pixel_size_A=1.0,
    )

    validation = validate_live_convention("5.0.5", [corrupted])

    assert validation.passed is False
    assert validation.server_version_matches is False
    assert validation.camera_checks["2fold"]["third_row_matches_direction"] is False
    assert validation.display_checks["2fold"]["vertical_flip_matches_cryosparc"] is False
    assert validation.failures


def test_live_fixture_records_auth_unavailability_as_skip(tmp_path):
    config = LiveFixtureConfig(
        base_url="https://cryosparc.example.test",
        project_uid="P1",
        workspace_uid="W1",
        volume_job_uid="J1",
        result_path=tmp_path / "auth-skip.json",
    )

    def no_auth_client(*args, **kwargs):
        raise ValueError("CryoSPARC authentication not provided or expired")

    result = run_live_convention_fixture(config, client_factory=no_auth_client)

    assert result.status == LiveFixtureStatus.SKIPPED.value
    assert result.executed is False
    assert "authentication unavailable" in result.reason
    assert json.loads(config.result_path.read_text())["status"] == "skipped"


def test_live_fixture_rejects_a_reachable_non_506_server(tmp_path):
    config = LiveFixtureConfig(
        base_url="https://cryosparc.example.test",
        project_uid="P1",
        workspace_uid="W1",
        volume_job_uid="J1",
        result_path=tmp_path / "version-fail.json",
    )
    client = SimpleNamespace(
        test_connection=lambda: True,
        api=SimpleNamespace(config=SimpleNamespace(get_version=lambda: "v5.0.5")),
    )

    result = run_live_convention_fixture(
        config,
        client_factory=lambda *args, **kwargs: client,
    )

    assert result.status == LiveFixtureStatus.FAILED.value
    assert result.executed is False
    assert "requires CryoSPARC 5.0.6" in result.reason
    assert json.loads(config.result_path.read_text())["status"] == "failed"
