from cryosparc_2d_projection.external_job_adapter import (
    AxisSourceOutput,
    CryoSPARCExternalJobAdapter,
    ExternalJobSource,
    ExternalJobPublicationError,
    InMemoryExternalJobBackend,
    LoadedVolume,
    SourceOutput,
    TemplateStack,
)

import numpy as np
import pytest
from cryosparc import mrc


def test_external_job_source_is_the_canonical_source_type():
    source = ExternalJobSource("J10", "templates_selected")

    assert SourceOutput is ExternalJobSource
    assert AxisSourceOutput is ExternalJobSource
    assert source.job_uid == "J10"
    assert source.output_name == "templates_selected"


def test_adapter_reads_template_stack_without_exposing_dataset_slots(tmp_path):
    stack = np.asarray(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.float32,
    )
    mrc.write(tmp_path / "templates.mrcs", stack, 1.5)
    dataset = np.asarray(
        [("templates.mrcs", 1, 1.5)],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )

    class Job:
        def load_input(self, name):
            assert name == "templates"
            return dataset

    class Project:
        dir = tmp_path

    from cryosparc_2d_projection.external_job_adapter import CryoSPARCExternalJobAdapter

    adapter = CryoSPARCExternalJobAdapter(Project(), "W1", job=Job())
    templates = adapter.read_template_stack("templates")

    assert isinstance(templates, TemplateStack)
    assert templates.class_averages[1].image.tolist() == [[5, 6], [7, 8]]
    assert templates.class_averages[1].pixel_size_A == 1.5


def test_template_reader_preserves_last_duplicate_and_empty_input_behavior(tmp_path):
    first = np.ones((1, 2, 2), dtype=np.float32)
    second = np.full((1, 2, 2), 2.0, dtype=np.float32)
    mrc.write(tmp_path / "first.mrcs", first, 1.5)
    mrc.write(tmp_path / "second.mrcs", second, 1.5)
    dtype = [
        ("blob/path", "U128"),
        ("blob/idx", "i4"),
        ("blob/psize_A", "f4"),
    ]
    datasets = {
        "duplicates": np.array(
            [("first.mrcs", 0, 1.5), ("second.mrcs", 0, 1.5)],
            dtype=dtype,
        ),
        "empty": np.array([], dtype=dtype),
    }
    backend = InMemoryExternalJobBackend(tmp_path, datasets)
    adapter = CryoSPARCExternalJobAdapter(backend, "W1", job=backend)

    duplicates = adapter.read_template_stack("duplicates")
    empty = adapter.read_template_stack("empty")

    assert duplicates.class_averages[0].image.tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert empty.class_averages == {}


def test_adapter_reads_matching_and_selected_rendering_volume(tmp_path):
    matching = np.ones((3, 3, 3), dtype=np.float32)
    sharpened = np.full((3, 3, 3), 2, dtype=np.float32)
    mrc.write(tmp_path / "matching.mrc", matching, 1.5)
    mrc.write(tmp_path / "sharpened.mrc", sharpened, 2.0)
    dataset = np.asarray(
        [("matching.mrc", 1.5, "sharpened.mrc", 2.0)],
        dtype=[
            ("map/path", "U128"),
            ("map/psize_A", "f4"),
            ("map_sharp/path", "U128"),
            ("map_sharp/psize_A", "f4"),
        ],
    )

    class Job:
        def load_input(self, name):
            assert name == "volume"
            return dataset

    class Project:
        dir = tmp_path

    from cryosparc_2d_projection.external_job_adapter import CryoSPARCExternalJobAdapter

    adapter = CryoSPARCExternalJobAdapter(Project(), "W1", job=Job())
    volume = adapter.read_volume("volume", rendering_map="sharpened")

    assert isinstance(volume, LoadedVolume)
    assert np.all(volume.matching_map == 1)
    assert np.all(volume.rendering_map == 2)
    assert volume.matching_pixel_size_A == 1.5
    assert volume.rendering_pixel_size_A == 2.0
    assert volume.matching_path == tmp_path / "matching.mrc"
    assert volume.rendering_path == tmp_path / "sharpened.mrc"


def test_adapter_reads_typed_particle_alignments_without_exposing_slots(tmp_path):
    selected = np.array(
        [(101, 3, 0.25)],
        dtype=[
            ("uid", "u8"),
            ("alignments2D/class", "i4"),
            ("alignments2D/pose", "f8"),
        ],
    )
    refined = np.array(
        [(101, [0.1, 0.2, 0.3])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )
    backend = InMemoryExternalJobBackend(
        tmp_path,
        {"selected": selected, "refined": refined},
    )
    adapter = CryoSPARCExternalJobAdapter(backend, "W1", job=backend)

    alignments_2d = adapter.read_2d_particle_alignments("selected")
    alignments_3d = adapter.read_3d_particle_alignments("refined")

    assert alignments_2d.uids.tolist() == [101]
    assert alignments_2d.class_ids.tolist() == [3]
    assert alignments_2d.poses.tolist() == [0.25]
    assert alignments_3d.uids.tolist() == [101]
    assert alignments_3d.poses.tolist() == [[0.1, 0.2, 0.3]]


def test_adapter_stages_template_output_before_publishing(tmp_path):
    class Job:
        uid = "J99"
        dir = tmp_path

        def add_output(self, **spec):
            self.spec = spec

        def alloc_output(self, name, count):
            return {
                "blob/path": np.empty(count, dtype=object),
                "blob/idx": np.empty(count, dtype=np.int32),
                "blob/shape": np.empty((count, 2), dtype=np.int32),
                "blob/psize_A": np.empty(count, dtype=np.float32),
            }

        def save_output(self, name, dataset):
            self.saved = (name, dataset)

    class Project:
        dir = tmp_path

    from cryosparc_2d_projection.external_job_adapter import CryoSPARCExternalJobAdapter

    job = Job()
    adapter = CryoSPARCExternalJobAdapter(Project(), "W1", job=job)
    adapter.add_template_output("projections", title="Projections")
    stack = np.zeros((2, 4, 4), dtype=np.float32)
    adapter.stage_template_stack(
        "projections", "projections.mrcs", stack, pixel_size_A=1.5
    )

    assert not hasattr(job, "saved")
    assert (tmp_path / "projections.mrcs").exists()

    adapter.publish()

    name, output = job.saved
    assert name == "projections"
    assert output["blob/path"].tolist() == [">J99/projections.mrcs"] * 2
    assert output["blob/idx"].tolist() == [0, 1]
    assert output["blob/shape"].tolist() == [[4, 4], [4, 4]]
    assert output["blob/psize_A"].tolist() == [1.5, 1.5]


def test_adapter_publication_error_names_output_without_rollback(tmp_path):
    class Job:
        uid = "J99"
        dir = tmp_path

        def add_output(self, **spec):
            pass

        def alloc_output(self, name, count):
            if name == "second":
                raise OSError("API unavailable")
            return {
                "blob/path": np.empty(count, dtype=object),
                "blob/idx": np.empty(count, dtype=np.int32),
                "blob/shape": np.empty((count, 2), dtype=np.int32),
                "blob/psize_A": np.empty(count, dtype=np.float32),
            }

        def save_output(self, name, dataset):
            self.saved = getattr(self, "saved", []) + [name]

    class Project:
        dir = tmp_path

    from cryosparc_2d_projection.external_job_adapter import CryoSPARCExternalJobAdapter

    job = Job()
    adapter = CryoSPARCExternalJobAdapter(Project(), "W1", job=job)
    adapter.add_template_output("first", title="First")
    adapter.add_template_output("second", title="Second")
    stack = np.zeros((1, 2, 2), dtype=np.float32)
    adapter.stage_template_stack("first", "first.mrcs", stack, pixel_size_A=1.0)
    adapter.stage_template_stack("second", "second.mrcs", stack, pixel_size_A=1.0)

    with pytest.raises(ExternalJobPublicationError, match="'second'"):
        adapter.publish()

    assert job.saved == ["first"]


def test_shared_in_memory_adapter_exposes_one_backend_for_workflow_tests(tmp_path):
    backend = InMemoryExternalJobBackend(tmp_path)
    adapter = CryoSPARCExternalJobAdapter(backend, "W1", job=backend)

    adapter.add_template_output("projections", title="Projections")
    stack = np.zeros((1, 2, 2), dtype=np.float32)
    adapter.stage_template_stack("projections", "projections.mrcs", stack, pixel_size_A=1.0)
    adapter.publish()

    assert backend.saved["projections"]["blob/idx"].tolist() == [0]
    assert backend.saved_outputs["projections"][0]["blob/path"].tolist() == [
        ">J99/projections.mrcs"
    ]
