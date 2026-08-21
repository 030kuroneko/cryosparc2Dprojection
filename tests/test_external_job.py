from contextlib import nullcontext
import json

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.external_job import (
    SourceOutput,
    _load_class_averages,
    run_external_orientation_job,
)
from cryosparc_2d_projection.surface_render import ClassRenderOptions


class FakeExternalJob:
    def __init__(self, directory, datasets):
        self.uid = "J99"
        self.directory = directory
        self.datasets = datasets
        self.inputs = []
        self.connections = []
        self.logs = []
        self.plots = []
        self.saved_outputs = {}
        self.outputs = []

    def add_input(self, **spec):
        self.inputs.append(spec)

    def add_output(self, **spec):
        if not spec.get("slots"):
            raise ValueError(
                "Must must provide slots=[...] argument with at least one slot"
            )
        self.outputs.append(spec)
        return spec["name"]

    def alloc_output(self, name, count):
        if not isinstance(count, int):
            return count.copy()
        if name == "rendering_map" or name.startswith("class_"):
            return {
                "map/path": np.empty(count, dtype=object),
                "map/shape": np.zeros((count, 3), dtype=np.int32),
                "map/psize_A": np.zeros(count, dtype=np.float32),
            }
        return {
            "blob/path": np.empty(count, dtype=object),
            "blob/idx": np.zeros(count, dtype=np.int32),
            "blob/shape": np.zeros((count, 2), dtype=np.int32),
            "blob/psize_A": np.zeros(count, dtype=np.float32),
        }

    def save_output(self, name, dataset, image=None):
        self.saved_outputs[name] = (dataset, image)

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


def test_selected_template_blob_indices_remain_original_class_ids(tmp_path):
    stack = np.zeros((50, 3, 3), dtype=np.float32)
    stack[10] = 10
    stack[12] = 12
    mrc.write(tmp_path / "templates.mrcs", stack, 1.5)
    templates = np.array(
        [("templates.mrcs", 10, 1.5), ("templates.mrcs", 12, 1.5)],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )
    project = type("ProjectDirectory", (), {"dir": tmp_path})()

    loaded = _load_class_averages(project, templates)

    assert sorted(loaded) == [10, 12]
    assert np.all(loaded[10].image == 10)
    assert np.all(loaded[12].image == 12)


def test_external_job_writes_orientation_results_for_cryosparc_5_0_6(tmp_path):
    select_2d = np.array(
        [(101, 0, np.pi / 2)],
        dtype=[
            ("uid", "u8"),
            ("alignments2D/class", "i4"),
            ("alignments2D/pose", "f8"),
        ],
    )
    refinement = np.array(
        [(101, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )
    volume_data = np.zeros((7, 7, 7), dtype=np.float32)
    volume_data[1, 2, 3] = 1.0
    volume_data[4, 1, 5] = 2.0
    volume_data[5, 5, 1] = 4.0
    mrc.write(tmp_path / "volume.mrc", volume_data, 1.5)
    sharpened_volume_data = volume_data * 2
    mrc.write(tmp_path / "volume_sharp.mrc", sharpened_volume_data, 1.5)
    class_average = np.rot90(volume_data.sum(axis=0))[None, ...]
    mrc.write(tmp_path / "templates.mrcs", class_average, 1.5)
    templates = np.array(
        [("templates.mrcs", 0, 1.5)],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )
    volume = np.array(
        [("volume.mrc", 1.5, "volume_sharp.mrc", 1.5)],
        dtype=[
            ("map/path", "U128"),
            ("map/psize_A", "f4"),
            ("map_sharp/path", "U128"),
            ("map_sharp/psize_A", "f4"),
        ],
    )
    job = FakeExternalJob(
        tmp_path,
        {
            "select_2d_particles": select_2d,
            "select_2d_templates": templates,
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
        interactive_class_numbers=(1,),
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=128,
            grid_size=32,
        ),
    )

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    assert results["cryosparc_version"] == "5.0.6"
    assert results["symmetry"] == "I"
    assert results["rendering"]["map"] == "sharpened"
    class_result = results["classes"][0]
    assert class_result["class_id"] == 0
    assert class_result["class_number"] == 1
    assert class_result["particle_count"] == 1
    assert class_result["angular_spread_degrees"] == 0.0
    assert np.allclose(class_result["camera"]["rotation_matrix"], [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(class_result["camera"]["quaternion_xyzw"], [
        0.0, 0.0, -(2**-0.5), 2**-0.5
    ])
    assert np.allclose(class_result["camera"]["projection_shift_pixels"], [0.0, 0.0])
    assert np.isclose(class_result["camera"]["match_score"], 1.0)
    assert class_result["camera"]["second_best_score"] < 1.0
    assert class_result["camera"]["score_margin"] > 0.0
    assert class_result["camera"]["match_confidence"] == "high"
    assert class_result["camera"]["matching_box_size"] == 7
    assert class_result["camera"]["matching_pixel_size_A"] == 1.5
    assert class_result["camera"]["search_evaluation_count"] <= 40
    assert class_result["camera"]["coordinate_convention"] == (
        "right-handed Cartesian active rotation; image rows increase downward"
    )
    assert class_result["symmetry_axis"] == {
        "label": "2-fold",
        "nearest_order": 2,
        "distance_degrees": 0.0,
        "threshold_degrees": 5.0,
    }
    assert project.created == ("W1", "2D Class Orientation (CryoSPARC 5.0.6)")
    assert ("select_2d_particles", "J10", "particles_selected") in job.connections
    assert ("select_2d_templates", "J10", "templates_selected") in job.connections
    assert ("refinement_particles", "J20", "particles") in job.connections
    assert ("refinement_volume", "J20", "volume") in job.connections
    volume_input = next(spec for spec in job.inputs if spec["name"] == "refinement_volume")
    assert volume_input["slots"] == ["map", "map_sharp"]
    projection_header, projections = mrc.read(tmp_path / "class_projections.mrcs")
    assert np.isclose(projection_header.xlen / projection_header.nx, 1.5)
    assert projections.shape == (1, 7, 7)
    assert np.allclose(projections[0], class_average[0], atol=1e-6)
    assert len(job.plots) == 1
    assert job.plots[0][1] == "Class camera preview 1/1"
    assert job.plots[0][2] == ["png"]
    preview = job.plots[0][0]
    assert len(preview.axes) == 3
    assert np.allclose(preview.axes[0].images[0].get_array(), class_average[0])
    assert np.allclose(preview.axes[1].images[0].get_array(), class_average[0])
    assert (tmp_path / "renders" / "class_001_exact.png").exists()
    assert not (tmp_path / "renders" / "class_001_oblique.png").exists()
    assert (tmp_path / "renders" / "class_001_comparison.png").exists()
    projection_output, thumbnail = job.saved_outputs["matched_projections"]
    assert projection_output["blob/path"].tolist() == [
        ">J99/class_projections.mrcs"
    ]
    assert projection_output["blob/idx"].tolist() == [0]
    assert projection_output["blob/shape"].tolist() == [[7, 7]]
    assert np.allclose(projection_output["blob/psize_A"], [1.5])
    assert thumbnail is preview
    rendering_output, rendering_thumbnail = job.saved_outputs["rendering_map"]
    assert rendering_output["map/path"].tolist() == ["volume_sharp.mrc"]
    assert rendering_thumbnail is None
    rendering_spec = next(spec for spec in job.outputs if spec["name"] == "rendering_map")
    assert rendering_spec["type"] == "volume"
    assert rendering_spec["slots"] == ["map"]
    assert "passthrough" not in rendering_spec
    _, interactive_volume = mrc.read(tmp_path / "class_001_volume.mrc")
    assert np.allclose(
        interactive_volume, np.rot90(sharpened_volume_data, axes=(1, 2))
    )
    class_volume_output, _ = job.saved_outputs["class_001_volume"]
    assert class_volume_output["map/path"].tolist() == [
        ">J99/class_001_volume.mrc"
    ]
    assert (tmp_path / "chimerax" / "class_001.cxc").exists()
    assert (tmp_path / "chimerax" / "all_classes.cxc").exists()


def test_class_render_options_validate_the_whole_rendering_policy():
    import pytest

    with pytest.raises(ValueError, match="rendering map"):
        ClassRenderOptions(map_name="unknown")
    with pytest.raises(ValueError, match="background"):
        ClassRenderOptions(background="blue")
    with pytest.raises(ValueError, match="192"):
        ClassRenderOptions(grid_size=193)
    with pytest.raises(ValueError, match="64"):
        ClassRenderOptions(image_size=63)
