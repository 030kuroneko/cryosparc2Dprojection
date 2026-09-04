import json

import numpy as np
import pytest
from cryosparc import mrc
from cryosparc.dataset import Dataset

from cryosparc_2d_projection.axis_cli import build_parser, main
from cryosparc_2d_projection.axis_external_job import (
    AxisSourceOutput,
    run_axis_search_job,
)
from cryosparc_2d_projection.axis_projection import project_axis_reference
from cryosparc_2d_projection.axis_search import AxisClassScoreError, AxisSearchConfig
from cryosparc_2d_projection.surface_render import ClassRenderOptions

from tests.external_job_backend import InMemoryExternalJobBackend


def _axis_job(
    directory,
    *,
    class_size=32,
    rendering_pixel_size_A=1.0,
    log_error=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    volume = np.zeros((class_size, class_size, class_size), dtype=np.float32)
    volume[
        int(0.22 * class_size) : int(0.38 * class_size),
        int(0.28 * class_size) : int(0.47 * class_size),
        int(0.56 * class_size) : int(0.75 * class_size),
    ] = 1.0
    volume[
        int(0.62 * class_size) : int(0.78 * class_size),
        int(0.53 * class_size) : int(0.72 * class_size),
        int(0.19 * class_size) : int(0.31 * class_size),
    ] = 0.6
    reference = project_axis_reference(volume, "2fold").projection
    mrc.write(directory / "volume.mrc", volume, 1.0)
    mrc.write(directory / "volume_sharp.mrc", volume[::-1].copy(), 1.0)
    mrc.write(directory / "templates.mrcs", reference[None], 1.0)
    return InMemoryExternalJobBackend(
        directory,
        {
            "templates": np.array(
                [("templates.mrcs", 0, 1.0)],
                dtype=[
                    ("blob/path", "U128"),
                    ("blob/idx", "i4"),
                    ("blob/psize_A", "f4"),
                ],
            ),
            "volume": np.array(
                [
                    (
                        "volume.mrc",
                        1.0,
                        "volume_sharp.mrc",
                        rendering_pixel_size_A,
                    )
                ],
                dtype=[
                    ("map/path", "U128"),
                    ("map/psize_A", "f4"),
                    ("map_sharp/path", "U128"),
                    ("map_sharp/psize_A", "f4"),
                ],
            ),
        },
        log_error=log_error,
    )


class AxisClient:
    def __init__(self, project):
        self.project = project

    def test_connection(self):
        return True

    def find_project(self, project_uid):
        return self.project


class AdvancingClock:
    def __init__(self, step):
        self.value = 0.0
        self.step = float(step)

    def __call__(self):
        self.value += self.step
        return self.value


def test_axis_cli_runs_from_templates_and_volume_only(tmp_path, capsys):
    job = _axis_job(tmp_path)
    client = AxisClient(job)

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--select-output", "templates_selected",
            "--volume-job", "J20",
            "--volume-output", "volume",
            "--axis-family", "2fold",
            "--top-n", "1",
            "--render-size", "64",
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0
    assert {item["name"] for item in job.inputs} == {"templates", "volume"}
    output_names = {item["name"] for item in job.outputs}
    assert output_names == {
        "axis_candidates_raw",
        "axis_candidates_aligned",
        "axis_exact_references",
        "axis_exact_search_projections",
        "axis_exact_matched_projections",
        "axis_search_preview",
    }
    assert set(job.saved) == output_names
    metadata = json.loads((tmp_path / "axis_search_results.json").read_text())
    assert len(job.plots) == 1
    assert job.output_images["axis_search_preview"]
    assert len(job.tile_images) == 1
    preview = job.saved["axis_search_preview"]
    aligned = job.saved["axis_candidates_aligned"]
    assert np.array_equal(preview["blob/idx"], aligned["blob/idx"])
    assert metadata["timings"]["exact-ranking"]["elapsed_seconds"] >= 0
    assert any(
        message.startswith("Axis Search stage: stage=exact-ranking status=started")
        for message in job.logs
    )
    assert any(message.startswith("Surface Sampling Grid:") for message in job.logs)
    assert "Axis Search stage: stage=exact-ranking status=started" in capsys.readouterr().out
    assert any(
        "family=2fold rank=1 class=1" in message for message in job.logs
    )
    assert any(
        message.startswith("Axis Search row JSON:")
        and "exact_axis_rotation_matrix" in message
        for message in job.logs
    )
    rendering_started = next(
        index
        for index, message in enumerate(job.logs)
        if message.startswith(
            "Axis Search stage: stage=result-rendering status=started"
        )
    )
    sampling = next(
        index
        for index, message in enumerate(job.logs)
        if message.startswith("Surface Sampling Grid:")
    )
    candidate_completed = next(
        index
        for index, message in enumerate(job.logs)
        if message.startswith("Result Rendering progress:")
    )
    output_started = next(
        index
        for index, message in enumerate(job.logs)
        if message.startswith("Axis Search stage: stage=output-writing status=started")
    )
    output_completed = next(
        index
        for index, message in enumerate(job.logs)
        if message.startswith(
            "Axis Search stage: stage=output-writing status=completed"
        )
    )
    rendering_completed = next(
        index
        for index, message in enumerate(job.logs)
        if message.startswith(
            "Axis Search stage: stage=result-rendering status=completed"
        )
    )
    assert (
        rendering_started
        < sampling
        < candidate_completed
        < output_started
        < output_completed
        < rendering_completed
    )


def test_axis_search_preserves_source_class_number_from_blob_index(tmp_path):
    job = _axis_job(tmp_path)
    _, source = mrc.read(tmp_path / "templates.mrcs")
    mrc.write(tmp_path / "templates.mrcs", np.repeat(source, 8, axis=0), 1.0)
    job.datasets["templates"]["blob/idx"][0] = 7
    client = AxisClient(job)

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1", "--workspace", "W1",
            "--select-job", "J1", "--select-output", "templates_selected",
            "--volume-job", "J2", "--volume-output", "volume",
            "--axis-family", "2fold", "--top-n", "1", "--render-size", "64",
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0
    metadata = json.loads((tmp_path / "axis_search_results.json").read_text())
    assert metadata["rows"][0]["class_number"] == 8


def test_dashboard_preview_upload_failure_warns_without_failing_job(tmp_path):
    job = _axis_job(tmp_path)
    job.set_output_image = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("preview upload unavailable")
    )
    job.set_tile_image = job.set_output_image
    client = AxisClient(job)

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1", "--workspace", "W1",
            "--select-job", "J1", "--select-output", "templates_selected",
            "--volume-job", "J2", "--volume-output", "volume",
            "--axis-family", "2fold", "--top-n", "1", "--render-size", "64",
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0
    assert "axis_search_preview" in job.saved
    assert any(
        "Could not attach Axis Search Dashboard Preview" in message
        and "preview upload unavailable" in message
        for message in job.logs
    )


def test_axis_cli_loads_templates_from_cryosparc_dataset(tmp_path):
    job = _axis_job(tmp_path)
    job.datasets["templates"] = Dataset(job.datasets["templates"])
    client = AxisClient(job)

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--select-output", "templates_selected",
            "--volume-job", "J20",
            "--volume-output", "volume",
            "--axis-family", "2fold",
            "--top-n", "1",
            "--render-size", "64",
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0


def test_axis_search_completes_when_progress_logging_is_unavailable(tmp_path):
    job = _axis_job(
        tmp_path,
        log_error=RuntimeError("Event Log is temporarily unavailable"),
    )

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--volume-job", "J20",
            "--axis-family", "2fold",
            "--top-n", "1",
            "--render-size", "64",
            "--render-grid-size", "16",
        ],
        client_factory=lambda url: AxisClient(job),
    )

    assert exit_code == 0
    assert (tmp_path / "axis_search_results.json").exists()


def test_axis_search_heartbeat_clock_is_injectable_without_a_cli_option(tmp_path):
    job = _axis_job(tmp_path)
    messages = []

    run_axis_search_job(
        job,
        "W1",
        AxisSourceOutput("J10", "templates_selected"),
        AxisSourceOutput("J20", "volume"),
        families=("2fold",),
        config=AxisSearchConfig(top_n=1),
        render_options=ClassRenderOptions(image_size=64, grid_size=16),
        status_callback=messages.append,
        progress_clock=AdvancingClock(31.0),
    )

    assert any(
        message.startswith("Axis Search progress:")
        and "pass=normal-coarse" in message
        and "angles=2/" in message
        for message in messages
    )


def test_axis_search_reports_recovery_after_a_cooperative_progress_stall(tmp_path):
    job = _axis_job(tmp_path)
    warnings = []

    run_axis_search_job(
        job,
        "W1",
        AxisSourceOutput("J10", "templates_selected"),
        AxisSourceOutput("J20", "volume"),
        families=("2fold",),
        config=AxisSearchConfig(top_n=1),
        render_options=ClassRenderOptions(image_size=64, grid_size=16),
        warning_callback=warnings.append,
        progress_clock=AdvancingClock(301.0),
    )

    assert warnings
    assert all("progress resumed" in warning for warning in warnings)


def test_axis_search_logs_the_active_stage_before_propagating_a_failure(tmp_path):
    job = _axis_job(tmp_path)
    mrc.write(tmp_path / "templates.mrcs", np.ones((1, 32, 32), dtype=np.float32), 1.0)
    mrc.write(tmp_path / "volume.mrc", np.ones((32, 32, 32), dtype=np.float32), 1.0)

    with pytest.raises(AxisClassScoreError):
        main(
            [
                "--url", "https://cryosparc.example.test",
                "--project", "P1",
                "--workspace", "W1",
                "--select-job", "J10",
                "--volume-job", "J20",
                "--axis-family", "2fold",
                "--top-n", "1",
                "--render-size", "64",
                "--render-grid-size", "16",
            ],
            client_factory=lambda url: AxisClient(job),
        )

    assert any(
        message.startswith("Axis Search stage: stage=exact-ranking status=failed")
        for message in job.logs
    )


def test_axis_cli_exposes_approved_one_family_defaults():
    args = build_parser().parse_args(
        [
            "--url", "http://localhost:39000",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--volume-job", "J20",
            "--axis-family", "3fold",
        ]
    )

    assert args.low_resolution_A == 80.0
    assert args.high_resolution_A == 15.0
    assert args.axis_family == ("3fold",)
    assert args.roll_coarse_step == 5.0
    assert args.roll_refine_step == 0.5
    assert args.shift_bound_fraction == 0.10
    assert args.top_n == 5


def test_axis_cli_searches_all_families_by_default_and_accepts_subset():
    common = [
        "--url", "http://localhost:39000",
        "--project", "P1",
        "--workspace", "W1",
        "--select-job", "J10",
        "--volume-job", "J20",
    ]

    defaults = build_parser().parse_args(common)
    subset = build_parser().parse_args([*common, "--axis-family", "2fold,5fold"])

    assert defaults.axis_family is None
    assert subset.axis_family == ("2fold", "5fold")
    assert subset.mirror_warning_margin == 0.05
    assert subset.axis_cone_degrees == 15.0
    assert subset.tilt_coarse_step == 3.0
    assert subset.tilt_refine_step == 0.5
    assert defaults.refine_near_axis is False
    assert build_parser().parse_args(
        [*common, "--refine-near-axis"]
    ).refine_near_axis is True


def test_axis_cli_accepts_repeatable_display_only_axis_roll_and_render_controls():
    args = build_parser().parse_args(
        [
            "--url", "http://localhost:39000",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--volume-job", "J20",
            "--axis-roll", "2fold=10",
            "--axis-roll", "3fold=-5",
            "--comparison-dpi", "150",
            "--preview-page-size", "2",
            "--render-map", "sharpened",
            "--render-size", "256",
            "--render-grid-size", "128",
            "--surface-level", "0.2",
        ]
    )

    assert args.axis_roll == ["2fold=10", "3fold=-5"]
    assert args.comparison_dpi == 150
    assert args.preview_page_size == 2
    assert args.render_map == "sharpened"
    assert args.render_size == 256
    assert args.render_grid_size == 128
    assert args.surface_level == 0.2


def test_axis_cli_exposes_opt_in_auto_crop_2d_flag():
    common = [
        "--url", "http://localhost:39000",
        "--project", "P1",
        "--workspace", "W1",
        "--select-job", "J10",
        "--volume-job", "J20",
    ]

    assert build_parser().parse_args(common).auto_crop_2d is False
    assert build_parser().parse_args([*common, "--auto-crop-2d"]).auto_crop_2d is True


def test_axis_cli_uses_rendering_pixel_size_for_physical_auto_crop(tmp_path):
    fine_directory = tmp_path / "fine"
    coarse_directory = tmp_path / "coarse"
    fine_job = _axis_job(
        fine_directory,
        class_size=64,
        rendering_pixel_size_A=1.0,
    )
    coarse_job = _axis_job(
        coarse_directory,
        class_size=64,
        rendering_pixel_size_A=2.0,
    )
    common = [
        "--url", "https://cryosparc.example.test",
        "--project", "P1",
        "--workspace", "W1",
        "--select-job", "J10",
        "--volume-job", "J20",
        "--axis-family", "2fold",
        "--top-n", "1",
        "--render-map", "sharpened",
        "--render-grid-size", "16",
        "--auto-crop-2d",
    ]

    assert main(common, client_factory=lambda _url: AxisClient(fine_job)) == 0
    assert main(common, client_factory=lambda _url: AxisClient(coarse_job)) == 0

    fine_metadata = json.loads(
        (fine_directory / "axis_search_results.json").read_text()
    )
    coarse_metadata = json.loads(
        (coarse_directory / "axis_search_results.json").read_text()
    )
    fine_viewport = fine_metadata["presentation"]["auto_crop_2d"][
        "camera_viewport_A"
    ]
    coarse_viewport = coarse_metadata["presentation"]["auto_crop_2d"][
        "camera_viewport_A"
    ]
    assert coarse_viewport == pytest.approx(2.0 * fine_viewport)


def test_axis_cli_rejects_removed_axis_families_option():
    common = [
        "--url", "http://localhost:39000",
        "--project", "P1",
        "--workspace", "W1",
        "--select-job", "J10",
        "--volume-job", "J20",
    ]

    with pytest.raises(SystemExit):
        build_parser().parse_args([*common, "--axis-families", "2fold,5fold"])


def test_axis_cli_runs_with_comma_separated_axis_family_selection(tmp_path):
    job = _axis_job(tmp_path)
    client = AxisClient(job)

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--volume-job", "J20",
            "--axis-family", "2fold,5fold",
            "--top-n", "1",
            "--render-size", "64",
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0
    metadata = json.loads((tmp_path / "axis_search_results.json").read_text())
    assert metadata["families"] == ["2fold", "5fold"]


def test_axis_cli_enables_near_axis_refinement_only_when_requested(tmp_path):
    job = _axis_job(tmp_path)

    exit_code = main(
        [
            "--url", "https://cryosparc.example.test",
            "--project", "P1",
            "--workspace", "W1",
            "--select-job", "J10",
            "--volume-job", "J20",
            "--axis-family", "2fold",
            "--top-n", "1",
            "--render-size", "64",
            "--render-grid-size", "16",
            "--axis-cone-degrees", "3",
            "--tilt-coarse-step", "3",
            "--tilt-refine-step", "1",
            "--refine-near-axis",
        ],
        client_factory=lambda url: AxisClient(job),
    )

    assert exit_code == 0
    assert {
        "axis_near_projections",
        "axis_near_search_projections",
        "axis_near_matched_projections",
    } <= set(job.saved)
    metadata = json.loads((tmp_path / "axis_search_results.json").read_text())
    assert metadata["proximity_config"]["enabled"] is True
