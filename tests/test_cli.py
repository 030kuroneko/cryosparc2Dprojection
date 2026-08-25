from contextlib import nullcontext

import numpy as np
from cryosparc import mrc

import pytest

from cryosparc_2d_projection.cli import build_parser, main, parse_class_numbers


class Job:
    def __init__(self, directory):
        self.uid = "J99"
        self.directory = directory
        volume = np.zeros((3, 3, 3), dtype=np.float32)
        volume[0, 1, 1] = 1.0
        volume[2, 0, 2] = 2.0
        mrc.write(directory / "volume.mrc", volume, 1.5)
        mrc.write(directory / "templates.mrcs", volume.sum(axis=0)[None], 1.5)
        self.datasets = {
            "select_2d_particles": np.array(
                [(101, 0, 0.0)],
                dtype=[
                    ("uid", "u8"),
                    ("alignments2D/class", "i4"),
                    ("alignments2D/pose", "f8"),
                ],
            ),
            "select_2d_templates": np.array(
                [("templates.mrcs", 0, 1.5)],
                dtype=[
                    ("blob/path", "U128"),
                    ("blob/idx", "i4"),
                    ("blob/psize_A", "f4"),
                ],
            ),
            "refinement_particles": np.array(
                [(101, [0.0, 0.0, 0.0])],
                dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
            ),
            "refinement_volume": np.array(
                [("volume.mrc", 1.5)],
                dtype=[("map/path", "U128"), ("map/psize_A", "f4")],
            ),
        }

    def add_input(self, **spec):
        pass

    def add_output(self, **spec):
        return spec["name"]

    def alloc_output(self, name, alloc):
        if not isinstance(alloc, int):
            return alloc.copy()
        if name == "rendering_map" or name.startswith("class_"):
            return {
                "map/path": np.empty(alloc, dtype=object),
                "map/shape": np.zeros((alloc, 3), dtype=np.int32),
                "map/psize_A": np.zeros(alloc, dtype=np.float32),
            }
        return {
            "blob/path": np.empty(alloc, dtype=object),
            "blob/idx": np.zeros(alloc, dtype=np.int32),
            "blob/shape": np.zeros((alloc, 2), dtype=np.int32),
            "blob/psize_A": np.zeros(alloc, dtype=np.float32),
        }

    def save_output(self, name, dataset, image=None):
        pass

    def connect(self, target_input, source_job_uid, source_output):
        pass

    def run(self):
        return nullcontext(self)

    def load_input(self, name):
        return self.datasets[name]

    def dir(self):
        return str(self.directory)

    def log(self, message):
        pass

    def log_plot(self, figure, text, formats, savefig_kw=None):
        pass


class Project:
    def __init__(self, job):
        self.job = job
        self.dir = job.directory

    def create_external_job(self, workspace_uid, title):
        return self.job


class Client:
    def __init__(self, project):
        self.project = project
        self.requested_project = None

    def test_connection(self):
        return True

    def find_project(self, project_uid):
        self.requested_project = project_uid
        return self.project


def test_cli_creates_job_from_cryosparc_job_output_ids(tmp_path, capsys):
    client = Client(Project(Job(tmp_path)))

    exit_code = main(
        [
            "--url",
            "https://cryosparc.example.test",
            "--project",
            "P1",
            "--workspace",
            "W1",
            "--select-job",
            "J10",
            "--select-output",
            "particles_selected",
            "--refinement-job",
            "J20",
            "--refinement-particles-output",
            "particles",
            "--volume-output",
            "volume",
            "--surface-level",
            "0.5",
            "--render-background",
            "light",
            "--render-size",
            "64",
            "--render-grid-size",
            "3",
            "--comparison-dpi",
            "200",
            "--preview-page-size",
            "1",
            "--diagnostic-low-resolution-A",
            "30",
            "--diagnostic-high-resolution-A",
            "8",
            "--diagnostic-mask-radius-fraction",
            "0.4",
            "--diagnostic-mask-edge-fraction",
            "0.08",
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0
    assert client.requested_project == "P1"
    assert (tmp_path / "class_orientations.json").exists()
    results = __import__("json").loads(
        (tmp_path / "class_orientations.json").read_text()
    )
    assert results["rendering"] == {
        "map": "map",
        "surface_level": 0.5,
        "surface_level_was_automatic": False,
        "warning": None,
        "background": "light",
        "image_size": 64,
        "grid_size": 3,
    }
    assert results["presentation"]["comparison_dpi"] == 200
    assert results["presentation"]["preview_page_size"] == 1
    assert results["presentation"]["requested_render_size"] == 64
    assert results["presentation"]["effective_render_size"] == 64
    assert "third comparison column may appear blurred" in capsys.readouterr().err
    diagnostic = results["classes"][0]["camera"][
        "diagnostic_band_limited_score"
    ]
    assert diagnostic["band_low_resolution_A_requested"] == 30.0
    assert diagnostic["band_high_resolution_A_requested"] == 8.0
    assert np.isclose(diagnostic["mask_radius_px"], 1.2)
    assert np.isclose(diagnostic["mask_edge_width_px"], 0.24)


def test_cli_accepts_one_based_class_numbers():
    assert parse_class_numbers("3,8,12") == (3, 8, 12)


def test_cli_accepts_surface_rendering_overrides():
    args = build_parser().parse_args(
        [
            "--url", "http://localhost:39000",
            "--project", "P1",
            "--workspace", "W9",
            "--select-job", "J1025",
            "--refinement-job", "J1083",
            "--surface-level", "0.12",
            "--render-map", "sharpened",
            "--render-background", "light",
            "--render-size", "768",
            "--render-grid-size", "160",
        ]
    )

    assert args.surface_level == 0.12
    assert args.render_map == "sharpened"
    assert args.render_background == "light"
    assert args.render_size == 768
    assert args.render_grid_size == 160


def test_cli_defaults_to_automatic_render_size_and_100_dpi_comparisons():
    args = build_parser().parse_args(
        [
            "--url", "http://localhost:39000",
            "--project", "P1",
            "--workspace", "W9",
            "--select-job", "J1025",
            "--refinement-job", "J1083",
        ]
    )

    assert args.render_size is None
    assert args.comparison_dpi == 100
    assert args.preview_page_size == 10


@pytest.mark.parametrize("value", ["0", "3,3", "class3"])
def test_cli_rejects_invalid_class_numbers(value):
    with pytest.raises(ValueError):
        parse_class_numbers(value)


@pytest.mark.parametrize(
    ("option", "value"),
    [("--render-size", "63"), ("--render-grid-size", "1")],
)
def test_cli_rejects_rendering_sizes_below_supported_minimum(option, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--url", "http://localhost:39000",
                "--project", "P1",
                "--workspace", "W9",
                "--select-job", "J1025",
                "--refinement-job", "J1083",
                option, value,
            ]
        )


def test_cli_rejects_render_grid_larger_than_192_cubed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--url", "http://localhost:39000",
                "--project", "P1",
                "--workspace", "W9",
                "--select-job", "J1025",
                "--refinement-job", "J1083",
                "--render-grid-size", "193",
            ]
        )


def test_cli_rejects_removed_oblique_inspection_option():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--url", "http://localhost:39000",
                "--project", "P1",
                "--workspace", "W9",
                "--select-job", "J1025",
                "--refinement-job", "J1083",
                "--oblique-tilt-degrees", "20",
            ]
        )


@pytest.mark.parametrize("symmetry", ["C2", "D7", "T", "O", "I1", "I2"])
def test_cli_rejects_symmetry_outside_v0_1_support(symmetry, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--url", "http://localhost:39000",
                "--project", "P1",
                "--workspace", "W9",
                "--select-job", "J1025",
                "--refinement-job", "J1083",
                "--symmetry", symmetry,
            ]
        )

    assert "v0.1 only supports C1 and I" in capsys.readouterr().err
