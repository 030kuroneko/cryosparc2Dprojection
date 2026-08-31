from cryosparc_2d_projection.external_job_adapter import (
    AxisSourceOutput,
    ExternalJobSource,
    ExternalJobPublicationError,
    InMemoryExternalJobAdapter,
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
    assert templates.images[1].tolist() == [[5, 6], [7, 8]]


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
    adapter = InMemoryExternalJobAdapter(tmp_path)

    adapter.add_template_output("projections", title="Projections")
    stack = np.zeros((1, 2, 2), dtype=np.float32)
    adapter.stage_template_stack("projections", "projections.mrcs", stack, pixel_size_A=1.0)
    adapter.publish()

    assert adapter.saved["projections"]["blob/idx"].tolist() == [0]
    assert adapter.saved_outputs["projections"][0]["blob/path"].tolist() == [
        ">J99/projections.mrcs"
    ]
