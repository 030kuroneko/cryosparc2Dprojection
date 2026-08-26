from contextlib import nullcontext
import json

import numpy as np
import pytest
from cryosparc import mrc
from cryosparc.dataset import Dataset

from cryosparc_2d_projection.axis_cli import build_parser, main
from cryosparc_2d_projection.axis_projection import project_axis_reference


class AxisJob:
    def __init__(self, directory):
        self.uid = "J99"
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        volume = np.zeros((32, 32, 32), dtype=np.float32)
        volume[7:12, 9:15, 18:24] = 1.0
        volume[20:25, 17:23, 6:10] = 0.6
        reference = project_axis_reference(volume, "2fold").projection
        mrc.write(directory / "volume.mrc", volume, 1.0)
        mrc.write(directory / "volume_sharp.mrc", volume[::-1].copy(), 1.0)
        mrc.write(directory / "templates.mrcs", reference[None], 1.0)
        self.datasets = {
            "templates": np.array(
                [("templates.mrcs", 0, 1.0)],
                dtype=[
                    ("blob/path", "U128"),
                    ("blob/idx", "i4"),
                    ("blob/psize_A", "f4"),
                ],
            ),
            "volume": np.array(
                [("volume.mrc", 1.0, "volume_sharp.mrc", 1.0)],
                dtype=[
                    ("map/path", "U128"),
                    ("map/psize_A", "f4"),
                    ("map_sharp/path", "U128"),
                    ("map_sharp/psize_A", "f4"),
                ],
            ),
        }
        self.inputs = []
        self.outputs = []
        self.saved = {}
        self.plots = []
        self.logs = []

    def add_input(self, **spec):
        self.inputs.append(spec)

    def connect(self, *args):
        pass

    def add_output(self, **spec):
        self.outputs.append(spec["name"])

    def alloc_output(self, name, size):
        return {
            "blob/path": np.empty(size, dtype=object),
            "blob/idx": np.zeros(size, dtype=np.int32),
            "blob/shape": np.zeros((size, 2), dtype=np.int32),
            "blob/psize_A": np.zeros(size, dtype=np.float32),
        }

    def save_output(self, name, dataset):
        self.saved[name] = dataset

    def load_input(self, name):
        return self.datasets[name]

    def run(self):
        return nullcontext(self)

    def dir(self):
        return str(self.directory)

    def log(self, message):
        self.logs.append(message)

    def log_plot(self, figure, text, formats, savefig_kw=None):
        self.plots.append((figure, text))


class AxisProject:
    def __init__(self, job):
        self.job = job
        self.dir = job.directory

    def create_external_job(self, workspace_uid, title):
        return self.job


class AxisClient:
    def __init__(self, project):
        self.project = project

    def test_connection(self):
        return True

    def find_project(self, project_uid):
        return self.project


def test_axis_cli_runs_from_templates_and_volume_only(tmp_path):
    job = AxisJob(tmp_path)
    client = AxisClient(AxisProject(job))

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
    assert set(job.outputs) == {
        "axis_candidates_raw",
        "axis_candidates_aligned",
        "axis_near_projections",
        "axis_exact_references",
    }
    assert set(job.saved) == set(job.outputs)
    metadata = json.loads((tmp_path / "axis_search_results.json").read_text())
    assert metadata["symmetry"] == "I"
    assert metadata["families"] == ["2fold"]
    assert metadata["rows"][0]["class_number"] == 1
    assert metadata["rows"][0]["score_provenance"]
    assert metadata["rows"][0]["refined_score"] >= metadata["rows"][0]["axis_class_score"]
    assert metadata["rows"][0]["exact_axis_rotation_matrix"]
    assert metadata["rows"][0]["near_axis_rotation_matrix"]
    assert metadata["presentation"]["columns"] == [
        "Axis-Aligned Class",
        "Best Near-Axis Projection",
        "Exact Axis Projection",
        "Best Near-Axis 3D View",
        "Exact Axis 3D View",
    ]
    assert metadata["presentation"]["static_only"] is True
    assert len(job.plots) == 1
    assert len(job.plots[0][0].axes) == 5
    assert any(
        "family=2fold rank=1 class=1" in message for message in job.logs
    )
    assert any(
        message.startswith("Axis Search row JSON:")
        and "exact_axis_rotation_matrix" in message
        for message in job.logs
    )


def test_axis_cli_loads_templates_from_cryosparc_dataset(tmp_path):
    job = AxisJob(tmp_path)
    job.datasets["templates"] = Dataset(job.datasets["templates"])
    client = AxisClient(AxisProject(job))

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
    job = AxisJob(tmp_path)
    client = AxisClient(AxisProject(job))

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


def test_presentation_overrides_do_not_change_scores_or_orientations(tmp_path):
    common = [
        "--url", "https://cryosparc.example.test",
        "--project", "P1",
        "--workspace", "W1",
        "--select-job", "J10",
        "--volume-job", "J20",
        "--axis-family", "2fold",
        "--top-n", "1",
        "--render-size", "64",
        "--render-grid-size", "16",
    ]
    first_job = AxisJob(tmp_path / "first")
    second_job = AxisJob(tmp_path / "second")

    main(common, client_factory=lambda url: AxisClient(AxisProject(first_job)))
    main(
        [
            *common,
            "--render-map", "sharpened",
            "--comparison-dpi", "150",
            "--preview-page-size", "1",
            "--axis-roll", "2fold=25",
        ],
        client_factory=lambda url: AxisClient(AxisProject(second_job)),
    )

    first = json.loads((first_job.directory / "axis_search_results.json").read_text())
    second = json.loads((second_job.directory / "axis_search_results.json").read_text())
    scientific_keys = (
        "axis_class_score",
        "refined_score",
        "roll_degrees",
        "shift_xy_pixels",
        "near_axis_rotation_matrix",
        "angular_distance_degrees",
    )
    assert {key: first["rows"][0][key] for key in scientific_keys} == {
        key: second["rows"][0][key] for key in scientific_keys
    }
    assert first["presentation"] != second["presentation"]
    _, first_raw = mrc.read(first_job.directory / "axis_candidates_raw.mrcs")
    _, second_raw = mrc.read(second_job.directory / "axis_candidates_raw.mrcs")
    _, first_aligned = mrc.read(
        first_job.directory / "axis_candidates_aligned.mrcs"
    )
    _, second_aligned = mrc.read(
        second_job.directory / "axis_candidates_aligned.mrcs"
    )
    assert np.array_equal(first_raw, second_raw)
    assert not np.array_equal(first_aligned, second_aligned)
