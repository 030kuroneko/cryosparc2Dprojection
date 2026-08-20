from contextlib import nullcontext

import numpy as np
from cryosparc import mrc

import pytest

from cryosparc_2d_projection.cli import main, parse_class_numbers


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
                [("templates.mrcs", 0)],
                dtype=[("blob/path", "U128"), ("blob/idx", "i4")],
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

    def log_plot(self, figure, text, formats):
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


def test_cli_creates_job_from_cryosparc_job_output_ids(tmp_path):
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
        ],
        client_factory=lambda url: client,
    )

    assert exit_code == 0
    assert client.requested_project == "P1"
    assert (tmp_path / "class_orientations.json").exists()


def test_cli_accepts_one_based_class_numbers():
    assert parse_class_numbers("3,8,12") == (3, 8, 12)


@pytest.mark.parametrize("value", ["0", "3,3", "class3"])
def test_cli_rejects_invalid_class_numbers(value):
    with pytest.raises(ValueError):
        parse_class_numbers(value)
