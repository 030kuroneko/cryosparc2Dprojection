from contextlib import nullcontext

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.cli import main


class Job:
    def __init__(self, directory):
        self.directory = directory
        mrc.write(directory / "volume.mrc", np.ones((3, 3, 3), dtype=np.float32), 1.5)
        self.datasets = {
            "select_2d_particles": np.array(
                [(101, 0)],
                dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
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
