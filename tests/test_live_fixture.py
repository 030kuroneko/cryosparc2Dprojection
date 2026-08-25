import json
from types import SimpleNamespace

from cryosparc_2d_projection.live_fixture import (
    ConventionObservation,
    LiveFixtureConfig,
    LiveFixtureStatus,
    load_convention_observation,
    record_fixture_result,
    run_live_convention_fixture,
    validate_live_convention,
)
from cryosparc_2d_projection.axis_projection import (
    AxisReferenceProjection,
    project_axis_reference,
)
from cryosparc_2d_projection.axis_registry import axis_family_records, get_axis_family
import numpy as np


_OBSERVED_CAMERAS = {
    "2fold": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "3fold": np.array(
        [[0.356822089773090, 0.934172358962716, 0.0], [0.0, 0.0, -1.0], [-0.934172358962716, 0.356822089773090, 0.0]]
    ),
    "5fold": np.array(
        [[0.525731112119134, 0.0, 0.850650808352040], [0.0, 1.0, 0.0], [-0.850650808352040, 0.0, 0.525731112119134]]
    ),
}
_OBSERVED_RAW = {
    "2fold": np.array([[1.0, 2.0, 3.0], [4.0, 7.0, 5.0], [6.0, 8.0, 9.0]], dtype=np.float32),
    "3fold": np.array([[2.0, 4.0, 1.0], [8.0, 3.0, 6.0], [5.0, 9.0, 7.0]], dtype=np.float32),
    "5fold": np.array([[9.0, 1.0, 4.0], [2.0, 8.0, 6.0], [3.0, 5.0, 7.0]], dtype=np.float32),
}
_OBSERVED_MAP_SHA256 = "0" * 64


def _independent_observation():
    return ConventionObservation(
        server_version="v5.0.6",
        source="cryosparc_5.0.6_worker_export",
        independent=True,
        project_uid="P1",
        source_job_uid="J1",
        source_output_name="volume",
        matching_map_sha256=_OBSERVED_MAP_SHA256,
        capture_method="worker_reference_export",
        camera_matrices=_OBSERVED_CAMERAS,
        raw_projections=_OBSERVED_RAW,
        display_projections={name: np.flipud(image) for name, image in _OBSERVED_RAW.items()},
        display_orientation="cryosparc_vertical_flip",
    )


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
    assert "missing CRYOSPARC_FIXTURE_URL" in result.reason
    recorded = json.loads(result_path.read_text())
    assert recorded["schema"] == "cryosparc-5.0.6-convention-fixture/v1"
    assert recorded["status"] == "skipped"
    assert recorded["executed"] is False
    assert recorded["server_version"] is None


def test_live_fixture_validation_checks_server_camera_and_display_contract():
    references = [
        AxisReferenceProjection(
            family=family,
            projection=_OBSERVED_RAW[family.name],
            display_projection=np.flipud(_OBSERVED_RAW[family.name]),
            rotation_matrix=_OBSERVED_CAMERAS[family.name],
            pixel_size_A=1.0,
        )
        for family in axis_family_records("I")
    ]

    validation = validate_live_convention(
        "v5.0.6",
        references,
        _independent_observation(),
        project_uid="P1",
        source_job_uid="J1",
        source_output_name="volume",
        matching_map_sha256=_OBSERVED_MAP_SHA256,
    )

    assert validation.passed is True
    assert validation.server_version_matches is True
    assert validation.camera_checks["2fold"]["exact_camera_matches_observation"] is True
    assert validation.display_checks["2fold"]["vertical_flip_matches_cryosparc"] is True


def test_live_fixture_validation_reports_mismatch_without_claiming_pass():
    reference = AxisReferenceProjection(
        family=get_axis_family("I", "2fold"),
        projection=_OBSERVED_RAW["2fold"],
        display_projection=np.fliplr(_OBSERVED_RAW["2fold"]),
        rotation_matrix=np.eye(3),
        pixel_size_A=1.0,
    )
    corrupted = type(reference)(
        family=reference.family,
        projection=reference.projection,
        display_projection=reference.display_projection,
        rotation_matrix=np.eye(3),
        pixel_size_A=reference.pixel_size_A,
    )

    validation = validate_live_convention(
        "5.0.5",
        [corrupted],
        _independent_observation(),
        project_uid="P1",
        source_job_uid="J1",
        source_output_name="volume",
        matching_map_sha256=_OBSERVED_MAP_SHA256,
    )

    assert validation.passed is False
    assert validation.server_version_matches is False
    assert validation.camera_checks["2fold"]["exact_camera_matches_observation"] is False
    assert validation.display_checks["2fold"]["vertical_flip_matches_cryosparc"] is False
    assert validation.failures


def test_live_fixture_binds_observation_to_current_project_source_and_map():
    references = [
        AxisReferenceProjection(
            family=family,
            projection=_OBSERVED_RAW[family.name],
            display_projection=np.flipud(_OBSERVED_RAW[family.name]),
            rotation_matrix=_OBSERVED_CAMERAS[family.name],
            pixel_size_A=1.0,
        )
        for family in axis_family_records("I")
    ]

    validation = validate_live_convention(
        "v5.0.6",
        references,
        _independent_observation(),
        project_uid="P1",
        source_job_uid="J1",
        source_output_name="volume",
        matching_map_sha256=_OBSERVED_MAP_SHA256,
    )

    assert validation.provenance_matches is True


def test_live_fixture_rejects_observation_bound_to_another_source():
    references = [
        AxisReferenceProjection(
            family=family,
            projection=_OBSERVED_RAW[family.name],
            display_projection=np.flipud(_OBSERVED_RAW[family.name]),
            rotation_matrix=_OBSERVED_CAMERAS[family.name],
            pixel_size_A=1.0,
        )
        for family in axis_family_records("I")
    ]

    validation = validate_live_convention(
        "v5.0.6",
        references,
        _independent_observation(),
        project_uid="P9",
        source_job_uid="J9",
        source_output_name="volume",
        matching_map_sha256="1" * 64,
    )

    assert validation.provenance_matches is False
    assert any("provenance" in failure for failure in validation.failures)


def test_live_fixture_records_auth_unavailability_as_skip(tmp_path):
    config = LiveFixtureConfig(
        base_url="https://cryosparc.example.test",
        project_uid="P1",
        workspace_uid="W1",
        volume_job_uid="J1",
        observation_path=tmp_path / "observation.json",
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
        observation_path=tmp_path / "not-used.json",
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


def test_observation_loader_requires_independent_worker_or_ui_artifacts(tmp_path):
    raw = np.arange(9, dtype=np.float32).reshape(3, 3)
    display = np.flipud(raw)
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "display.npy", display)
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(
        json.dumps(
            {
                "schema": "cryosparc-5.0.6-convention-observation/v1",
                "server_version": "v5.0.6",
                "source": "cryosparc_5.0.6_worker_export",
                "independent": True,
                "project_uid": "P1",
                "source_job_uid": "J1",
                "source_output_name": "volume",
                "matching_map_sha256": _OBSERVED_MAP_SHA256,
                "capture_method": "worker_reference_export",
                "display_orientation": "cryosparc_vertical_flip",
                "camera_matrices": {
                    name: matrix.tolist() for name, matrix in _OBSERVED_CAMERAS.items()
                },
                "reference_projections": {
                    name: {"raw": "raw.npy", "display": "display.npy"}
                    for name in _OBSERVED_CAMERAS
                },
            }
        )
    )

    observation = load_convention_observation(observation_path)

    assert observation.independent is True
    assert observation.source == "cryosparc_5.0.6_worker_export"
    assert observation.source_job_uid == "J1"
    assert np.array_equal(observation.raw_projections["2fold"], raw)
    assert np.array_equal(observation.display_projections["2fold"], display)
