"""Runnable CryoSPARC 5.0.6 convention-validation fixture.

The fixture is optional in ordinary CI because it needs a real CryoSPARC
instance and a saved CryoSPARC Tools session.  When it is run, it creates a
real supported External Job, consumes a real volume output, registers raw and
CryoSPARC-display-oriented reference projections, and writes an auditable JSON
result.  No unavailable live run is reported as a pass.
"""

from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.axis_projection import project_axis_reference
from cryosparc_2d_projection.axis_registry import axis_family_records
from cryosparc_2d_projection.external_job import TARGET_CRYOSPARC_VERSION


FIXTURE_SCHEMA = "cryosparc-5.0.6-convention-fixture/v1"
OBSERVATION_SCHEMA = "cryosparc-5.0.6-convention-observation/v1"
SUPPORTED_CAPTURE_METHODS = frozenset(
    {"worker_reference_export", "worker_reference_job", "ui_reference_export"}
)


class LiveFixtureStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class LiveFixtureConfig:
    """Connection and source-output settings for a live fixture run."""

    base_url: str | None = None
    project_uid: str | None = None
    workspace_uid: str | None = None
    volume_job_uid: str | None = None
    volume_output: str = "volume"
    observation_path: Path | None = None
    result_path: Path | None = None
    timeout: int = 300
    required: bool = False

    @property
    def is_configured(self):
        return not self.missing_fields

    @property
    def missing_fields(self):
        fields = {
            "CRYOSPARC_FIXTURE_URL": self.base_url,
            "CRYOSPARC_FIXTURE_PROJECT": self.project_uid,
            "CRYOSPARC_FIXTURE_WORKSPACE": self.workspace_uid,
            "CRYOSPARC_FIXTURE_VOLUME_JOB": self.volume_job_uid,
            "CRYOSPARC_FIXTURE_OBSERVATION": self.observation_path,
        }
        return tuple(name for name, value in fields.items() if not value)

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        return cls(
            base_url=(
                environment.get("CRYOSPARC_FIXTURE_URL")
                or environment.get("CRYOSPARC_BASE_URL")
            ),
            project_uid=(
                environment.get("CRYOSPARC_FIXTURE_PROJECT")
                or environment.get("CRYOSPARC_PROJECT")
            ),
            workspace_uid=(
                environment.get("CRYOSPARC_FIXTURE_WORKSPACE")
                or environment.get("CRYOSPARC_WORKSPACE")
            ),
            volume_job_uid=(
                environment.get("CRYOSPARC_FIXTURE_VOLUME_JOB")
                or environment.get("CRYOSPARC_VOLUME_JOB")
            ),
            volume_output=environment.get("CRYOSPARC_FIXTURE_VOLUME_OUTPUT", "volume"),
            observation_path=(
                Path(environment["CRYOSPARC_FIXTURE_OBSERVATION"])
                if environment.get("CRYOSPARC_FIXTURE_OBSERVATION")
                else None
            ),
            result_path=(
                Path(environment["CRYOSPARC_FIXTURE_RESULT"])
                if environment.get("CRYOSPARC_FIXTURE_RESULT")
                else None
            ),
            required=_as_bool(environment.get("CRYOSPARC_FIXTURE_REQUIRED", "0")),
        )


@dataclass(frozen=True)
class ConventionObservation:
    """Independent 5.0.6 worker/UI observation consumed by the fixture."""

    server_version: str
    source: str
    independent: bool
    project_uid: str
    source_job_uid: str
    source_output_name: str
    matching_map_sha256: str
    capture_method: str
    camera_matrices: dict[str, np.ndarray]
    raw_projections: dict[str, np.ndarray]
    display_projections: dict[str, np.ndarray]
    display_orientation: str
    path: Path | None = None


def load_convention_observation(path):
    """Load an independently captured worker/UI convention artifact.

    The artifact must identify its CryoSPARC source and contain literal camera
    matrices plus raw/display reference arrays.  A template or an artifact
    generated solely from this package is rejected before a live job can pass.
    Array paths are resolved relative to the JSON file.
    """

    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError(f"observation schema must be {OBSERVATION_SCHEMA}")
    source = str(payload.get("source", ""))
    if not source.startswith("cryosparc_5.0.6_"):
        raise ValueError("observation source must identify CryoSPARC 5.0.6")
    if payload.get("independent") is not True:
        raise ValueError("observation must explicitly declare independent=true")
    capture_method = _required_observation_field(payload, "capture_method")
    if capture_method not in SUPPORTED_CAPTURE_METHODS:
        raise ValueError(f"unsupported observation capture method: {capture_method}")
    matching_map_sha256 = _required_observation_field(
        payload, "matching_map_sha256"
    )
    if len(matching_map_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in matching_map_sha256
    ):
        raise ValueError("matching_map_sha256 must be a 64-character hexadecimal digest")
    cameras = {
        str(name): np.asarray(matrix, dtype=float)
        for name, matrix in payload.get("camera_matrices", {}).items()
    }
    references = payload.get("reference_projections", {})
    raw = {}
    display = {}
    for family_name, paths in references.items():
        if not isinstance(paths, dict) or "raw" not in paths or "display" not in paths:
            raise ValueError(
                f"observation reference for {family_name} must provide raw and display paths"
            )
        raw[str(family_name)] = _load_observation_array(path.parent / paths["raw"])
        display[str(family_name)] = _load_observation_array(path.parent / paths["display"])
    return ConventionObservation(
        server_version=str(payload.get("server_version", "")),
        source=source,
        independent=True,
        project_uid=_required_observation_field(payload, "project_uid"),
        source_job_uid=_required_observation_field(payload, "source_job_uid"),
        source_output_name=_required_observation_field(payload, "source_output_name"),
        matching_map_sha256=matching_map_sha256,
        capture_method=capture_method,
        camera_matrices=cameras,
        raw_projections=raw,
        display_projections=display,
        display_orientation=str(payload.get("display_orientation", "")),
        path=path,
    )


def _required_observation_field(payload, name):
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"observation requires non-empty {name}")
    return value.strip()


def _load_observation_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix.lower() in {".mrc", ".mrcs"}:
        _, array = mrc.read(path)
        return np.asarray(array, dtype=np.float32)
    raise ValueError(f"unsupported observation array format: {path.suffix}")


@dataclass(frozen=True)
class LiveFixtureResult:
    status: str
    reason: str | None
    executed: bool
    server_version: str | None
    result_path: Path | None
    observation_path: Path | None = None
    job_uid: str | None = None
    output_names: tuple[str, ...] = ()
    camera_checks: dict[str, object] | None = None
    display_checks: dict[str, object] | None = None
    provenance: dict[str, object] | None = None
    error: str | None = None

    def as_dict(self):
        return {
            "schema": FIXTURE_SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "executed": self.executed,
            "server_version": self.server_version,
            "observation_path": (
                None if self.observation_path is None else str(self.observation_path)
            ),
            "job_uid": self.job_uid,
            "output_names": list(self.output_names),
            "camera_checks": self.camera_checks,
            "display_checks": self.display_checks,
            "provenance": self.provenance,
            "error": self.error,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class LiveConventionValidation:
    """Auditable checks performed by the live convention fixture."""

    server_version: str
    server_version_matches: bool
    provenance_matches: bool
    camera_checks: dict[str, dict[str, bool]]
    display_checks: dict[str, dict[str, object]]
    failures: tuple[str, ...]

    @property
    def passed(self):
        return not self.failures


def validate_live_convention(
    server_version,
    references,
    observation,
    *,
    project_uid,
    source_job_uid,
    source_output_name,
    matching_map_sha256,
):
    """Compare live output with an independent CryoSPARC observation.

    Registry constants are not used as the expected evidence here.  The
    observation's literal matrices and exported raw/display arrays must come
    from an independent 5.0.6 worker/API or a documented UI export.
    """

    if not isinstance(observation, ConventionObservation):
        raise TypeError("observation must be a ConventionObservation")
    server_version = str(server_version)
    server_version_matches = _normalize_server_version(server_version) == TARGET_CRYOSPARC_VERSION
    observation_version_matches = (
        _normalize_server_version(observation.server_version)
        == TARGET_CRYOSPARC_VERSION
    )
    camera_checks = {}
    display_checks = {}
    failures = []
    if not server_version_matches:
        failures.append(
            f"server version {server_version!r} is not {TARGET_CRYOSPARC_VERSION}"
        )
    if not observation_version_matches:
        failures.append(
            f"observation server version {observation.server_version!r} is not "
            f"{TARGET_CRYOSPARC_VERSION}"
        )
    if observation.display_orientation != "cryosparc_vertical_flip":
        failures.append("observation display orientation is not cryosparc_vertical_flip")
    observation_provenance_declared = (
        observation.independent
        and observation.source.startswith("cryosparc_5.0.6_")
    )
    if not observation_provenance_declared:
        failures.append(
            "observation must declare independent CryoSPARC 5.0.6 provenance"
        )
    provenance_matches = (
        observation.project_uid == str(project_uid)
        and observation.source_job_uid == str(source_job_uid)
        and observation.source_output_name == str(source_output_name)
        and observation.matching_map_sha256 == str(matching_map_sha256)
        and observation.capture_method in SUPPORTED_CAPTURE_METHODS
        and len(observation.matching_map_sha256) == 64
        and all(character in "0123456789abcdefABCDEF" for character in observation.matching_map_sha256)
        and observation_provenance_declared
    )
    if not provenance_matches:
        failures.append(
            "observation provenance does not match current project/source/map "
            "or has an unsupported capture method"
        )
    expected_families = {family.name for family in axis_family_records("I")}
    seen_families = set()
    for reference in references:
        family_name = reference.family.name
        seen_families.add(family_name)
        expected_camera = observation.camera_matrices.get(family_name)
        expected_raw = observation.raw_projections.get(family_name)
        expected_display = observation.display_projections.get(family_name)
        camera = np.asarray(reference.rotation_matrix, dtype=float)
        camera_checks[family_name] = {
            "orthogonal": bool(np.allclose(camera @ camera.T, np.eye(3), atol=1e-8)),
            "determinant_plus_one": bool(np.isclose(np.linalg.det(camera), 1.0, atol=1e-8)),
            "third_row_matches_direction": bool(
                np.allclose(camera[2], reference.family.representative_view_direction, atol=1e-8)
            ),
            "exact_camera_matches_observation": bool(
                expected_camera is not None
                and np.asarray(expected_camera).shape == (3, 3)
                and np.allclose(camera, expected_camera, atol=1e-8)
            ),
        }
        display_checks[family_name] = {
            "raw_matches_observation": bool(
                expected_raw is not None
                and np.asarray(expected_raw).shape == reference.projection.shape
                and np.allclose(reference.projection, expected_raw, atol=1e-5)
            ),
            "display_matches_observation": bool(
                expected_display is not None
                and np.asarray(expected_display).shape == reference.display_projection.shape
                and np.allclose(reference.display_projection, expected_display, atol=1e-5)
            ),
            "vertical_flip_matches_cryosparc": bool(
                np.array_equal(reference.display_projection, np.flipud(reference.projection))
            ),
            "shape": list(reference.projection.shape),
        }
        for check_name, passed in camera_checks[family_name].items():
            if not passed:
                failures.append(f"{family_name} camera check failed: {check_name}")
        for check_name, passed in display_checks[family_name].items():
            if check_name != "shape" and not passed:
                failures.append(f"{family_name} display check failed: {check_name}")
    missing_families = sorted(expected_families - seen_families)
    if missing_families:
        failures.append("missing axis-family references: " + ", ".join(missing_families))
    for missing in sorted(expected_families - set(observation.camera_matrices)):
        failures.append(f"observation missing camera matrix: {missing}")
    for missing in sorted(expected_families - set(observation.raw_projections)):
        failures.append(f"observation missing raw projection: {missing}")
    for missing in sorted(expected_families - set(observation.display_projections)):
        failures.append(f"observation missing display projection: {missing}")
    return LiveConventionValidation(
        server_version=server_version,
        server_version_matches=server_version_matches and observation_version_matches,
        provenance_matches=provenance_matches,
        camera_checks=camera_checks,
        display_checks=display_checks,
        failures=tuple(failures),
    )


def record_fixture_result(
    result_path,
    *,
    status,
    config,
    reason=None,
    executed=None,
    server_version=None,
    observation_path=None,
    job_uid=None,
    output_names=(),
    camera_checks=None,
    display_checks=None,
    provenance=None,
    error=None,
):
    """Write one honest fixture result record and return its value object."""

    status = LiveFixtureStatus(status).value
    result_path = Path(result_path)
    result = LiveFixtureResult(
        status=status,
        reason=reason,
        executed=(status == LiveFixtureStatus.PASSED.value if executed is None else bool(executed)),
        server_version=server_version,
        result_path=result_path,
        observation_path=(
            None if observation_path is None else Path(observation_path)
        ),
        job_uid=job_uid,
        output_names=tuple(output_names),
        camera_checks=camera_checks,
        display_checks=display_checks,
        provenance=(
            {
                "project_uid": config.project_uid,
                "source_job_uid": config.volume_job_uid,
                "source_output_name": config.volume_output,
                **(provenance or {}),
            }
        ),
        error=error,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    return result


def run_live_convention_fixture(config, *, client_factory=None):
    """Run the real 5.0.6 worker-facing fixture when configured.

    Missing settings, missing saved authentication, and unreachable servers
    produce a ``skipped`` record.  A reachable server with any version other
    than exactly v5.0.6 produces ``failed``.  A successful run is recorded only
    after the External Job has registered both output stacks and all checks.
    """

    result_path = config.result_path
    if result_path is None:
        result_path = Path("cryosparc_5_0_6_convention_result.json")
    if not config.is_configured:
        missing = ", ".join(config.missing_fields)
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.SKIPPED,
            config=config,
            reason=f"missing {missing}",
            executed=False,
            observation_path=config.observation_path,
        )

    if client_factory is None:
        from cryosparc.tools import CryoSPARC

        client_factory = CryoSPARC
    try:
        client = client_factory(config.base_url, timeout=config.timeout)
        if not client.test_connection():
            raise ConnectionError(f"CryoSPARC API health check failed at {config.base_url}")
        server_version = _read_server_version(client)
    except Exception as error:
        message = str(error)
        reason = (
            f"CryoSPARC authentication unavailable: {message}"
            if "auth" in message.lower() or "session" in message.lower()
            else f"CryoSPARC live connection unavailable: {message}"
        )
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.SKIPPED,
            config=config,
            reason=reason,
            executed=False,
            error=message,
            observation_path=config.observation_path,
        )

    if _normalize_server_version(server_version) != TARGET_CRYOSPARC_VERSION:
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.FAILED,
            config=config,
            reason=(
                f"fixture requires CryoSPARC {TARGET_CRYOSPARC_VERSION}; "
                f"server reported {server_version!r}"
            ),
            executed=False,
            server_version=server_version,
            observation_path=config.observation_path,
        )

    try:
        observation = load_convention_observation(config.observation_path)
    except FileNotFoundError as error:
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.SKIPPED,
            config=config,
            reason=(
                "independent CryoSPARC 5.0.6 observation is required; "
                f"file was not found: {config.observation_path}"
            ),
            executed=False,
            server_version=server_version,
            observation_path=config.observation_path,
            error=str(error),
        )
    except Exception as error:
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.FAILED,
            config=config,
            reason="independent convention observation could not be loaded",
            executed=False,
            server_version=server_version,
            observation_path=config.observation_path,
            error=str(error),
        )

    try:
        project = client.find_project(config.project_uid)
        source_job = project.find_job(config.volume_job_uid)
        if config.volume_output not in source_job.outputs:
            raise ValueError(
                f"source job {config.volume_job_uid} has no output {config.volume_output!r}"
            )
        job = project.create_external_job(
            config.workspace_uid,
            title=f"Axis Convention Fixture (CryoSPARC {TARGET_CRYOSPARC_VERSION})",
        )
        job.add_input(
            type="volume",
            name="matching_volume",
            min=1,
            max=1,
            slots=["map"],
            title="Unsharpened Matching Map fixture input",
        )
        job.connect("matching_volume", config.volume_job_uid, config.volume_output)
        job.add_output(
            type="template",
            name="axis_reference_raw",
            slots=["blob"],
            title="Raw exact axis reference projections",
        )
        job.add_output(
            type="template",
            name="axis_reference_display",
            slots=["blob"],
            title="CryoSPARC display-oriented axis references",
        )

        camera_checks = {}
        display_checks = {}
        with job.run():
            volume_input = job.load_input("matching_volume")
            volume_path = _resolve_project_path_local(project, volume_input["map/path"][0])
            _, volume = mrc.read(volume_path)
            pixel_size_A = float(volume_input["map/psize_A"][0])
            matching_map_sha256 = hashlib.sha256(
                np.ascontiguousarray(volume).tobytes()
            ).hexdigest()
            references = [
                project_axis_reference(
                    volume,
                    family,
                    pixel_size_A=pixel_size_A,
                )
                for family in axis_family_records("I")
            ]
            raw_stack = np.asarray([reference.projection for reference in references], dtype=np.float32)
            display_stack = np.asarray(
                [reference.display_projection for reference in references],
                dtype=np.float32,
            )
            validation = validate_live_convention(
                server_version,
                references,
                observation,
                project_uid=config.project_uid,
                source_job_uid=config.volume_job_uid,
                source_output_name=config.volume_output,
                matching_map_sha256=matching_map_sha256,
            )
            camera_checks = validation.camera_checks
            display_checks = validation.display_checks
            if not validation.passed:
                raise ValueError("; ".join(validation.failures))
            job_directory = _resource_directory(job)
            raw_filename = "axis_reference_raw.mrcs"
            display_filename = "axis_reference_display.mrcs"
            mrc.write(job_directory / raw_filename, raw_stack, pixel_size_A)
            mrc.write(job_directory / display_filename, display_stack, pixel_size_A)
            metadata = {
                "schema": FIXTURE_SCHEMA,
                "server_version": server_version,
                "observation": (
                    None if observation.path is None else str(observation.path)
                ),
                "observation_source": observation.source,
                "observation_capture_method": observation.capture_method,
                "matching_map": "unsharpened map slot",
                "matching_map_sha256": matching_map_sha256,
                "families": [family.name for family in axis_family_records("I")],
                "camera_matrices": {
                    reference.family.name: reference.rotation_matrix.tolist()
                    for reference in references
                },
                "camera_checks": camera_checks,
                "display_checks": display_checks,
                "raw_output": raw_filename,
                "display_output": display_filename,
            }
            (job_directory / "axis_convention_fixture.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            for name, filename, stack in (
                ("axis_reference_raw", raw_filename, raw_stack),
                ("axis_reference_display", display_filename, display_stack),
            ):
                output = job.alloc_output(name, len(stack))
                output["blob/path"][:] = f">{job.uid}/{filename}"
                output["blob/idx"][:] = np.arange(len(stack))
                output["blob/shape"][:] = stack.shape[1:]
                output["blob/psize_A"][:] = pixel_size_A
                job.save_output(name, output)
            job.log(
                "Verified CryoSPARC 5.0.6 I axis cameras and vertical display "
                "orientation using an unsharpened Matching Map."
            )
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.PASSED,
            config=config,
            reason="live External Job completed and outputs were registered",
            executed=True,
            server_version=server_version,
            job_uid=job.uid,
            output_names=("axis_reference_raw", "axis_reference_display"),
            camera_checks=camera_checks,
            display_checks=display_checks,
            provenance={
                "matching_map_sha256": matching_map_sha256,
                "capture_method": observation.capture_method,
            },
            observation_path=config.observation_path,
        )
    except Exception as error:
        return record_fixture_result(
            result_path,
            status=LiveFixtureStatus.FAILED,
            config=config,
            reason="live External Job did not complete",
            executed=True,
            server_version=server_version,
            observation_path=config.observation_path,
            error=str(error),
        )


def _read_server_version(client):
    version = client.api.config.get_version()
    return str(version)


def _normalize_server_version(value):
    return str(value).strip().removeprefix("v")


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resource_directory(resource):
    directory = getattr(resource, "dir")
    directory = directory() if callable(directory) else directory
    return Path(directory)


def _resolve_project_path_local(project, path):
    if isinstance(path, bytes):
        path = path.decode()
    path = Path(str(path).removeprefix(">"))
    if path.is_absolute():
        return path
    return _resource_directory(project) / path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the optional CryoSPARC 5.0.6 I convention fixture."
    )
    parser.add_argument("--url", help="CryoSPARC URL (or CRYOSPARC_FIXTURE_URL)")
    parser.add_argument("--project", help="Project UID")
    parser.add_argument("--workspace", help="Workspace UID")
    parser.add_argument("--volume-job", help="Job UID exposing the volume output")
    parser.add_argument("--volume-output", default=None)
    parser.add_argument(
        "--observation",
        type=Path,
        default=None,
        help="Independent CryoSPARC 5.0.6 observation JSON",
    )
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    environment_config = LiveFixtureConfig.from_environment()
    config = LiveFixtureConfig(
        base_url=args.url or environment_config.base_url,
        project_uid=args.project or environment_config.project_uid,
        workspace_uid=args.workspace or environment_config.workspace_uid,
        volume_job_uid=args.volume_job or environment_config.volume_job_uid,
        volume_output=args.volume_output or environment_config.volume_output,
        observation_path=args.observation or environment_config.observation_path,
        result_path=args.result or environment_config.result_path,
        timeout=args.timeout,
        required=args.required or environment_config.required,
    )
    result = run_live_convention_fixture(config)
    print(json.dumps(result.as_dict(), indent=2))
    if result.status == LiveFixtureStatus.FAILED.value:
        return 1
    if result.status == LiveFixtureStatus.SKIPPED.value and config.required:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
