from contextlib import nullcontext
import json

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.external_job import (
    SourceOutput,
    run_external_orientation_job,
)


class FakeExternalJob:
    def __init__(self, directory, datasets):
        self.directory = directory
        self.datasets = datasets
        self.inputs = []
        self.connections = []
        self.logs = []
        self.plots = []

    def add_input(self, **spec):
        self.inputs.append(spec)

    def connect(self, target_input, source_job_uid, source_output):
        self.connections.append((target_input, source_job_uid, source_output))

    def run(self):
        return nullcontext(self)

    def load_input(self, name):
        return self.datasets[name]

    def dir(self):
        return str(self.directory)

    def log(self, message):
        self.logs.append(message)

    def log_plot(self, figure, text, formats):
        self.plots.append((figure, text, formats))


class FakeProject:
    def __init__(self, job):
        self.job = job
        self.created = None

    def create_external_job(self, workspace_uid, title):
        self.created = (workspace_uid, title)
        return self.job


def test_external_job_writes_orientation_results_for_cryosparc_5_0_6(tmp_path):
    select_2d = np.array(
        [(101, 0)],
        dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
    )
    refinement = np.array(
        [(101, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )
    volume_data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    mrc.write(tmp_path / "volume.mrc", volume_data, 1.5)
    volume = np.array(
        [("volume.mrc", 1.5)],
        dtype=[("map/path", "U128"), ("map/psize_A", "f4")],
    )
    job = FakeExternalJob(
        tmp_path,
        {
            "select_2d_particles": select_2d,
            "refinement_particles": refinement,
            "refinement_volume": volume,
        },
    )
    project = FakeProject(job)
    project.dir = tmp_path

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="I",
    )

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    assert results == {
        "cryosparc_version": "5.0.6",
        "symmetry": "I",
        "classes": [
            {
                "class_id": 0,
                "class_number": 1,
                "particle_count": 1,
                "view_direction": [0.0, 0.0, 1.0],
                "angular_spread_degrees": 0.0,
            }
        ],
    }
    assert project.created == ("W1", "2D Class Orientation (CryoSPARC 5.0.6)")
    assert ("select_2d_particles", "J10", "particles_selected") in job.connections
    assert ("select_2d_templates", "J10", "templates_selected") in job.connections
    assert ("refinement_particles", "J20", "particles") in job.connections
    assert ("refinement_volume", "J20", "volume") in job.connections
    volume_input = next(spec for spec in job.inputs if spec["name"] == "refinement_volume")
    assert volume_input["slots"] == ["map"]
    projection_header, projections = mrc.read(tmp_path / "class_projections.mrcs")
    assert np.isclose(projection_header.xlen / projection_header.nx, 1.5)
    assert projections.shape == (1, 3, 3)
    assert np.allclose(projections[0], volume_data.sum(axis=0))
    assert len(job.plots) == 1
    assert job.plots[0][1] == "Class projection preview"
    assert job.plots[0][2] == ["png"]
