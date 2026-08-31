"""Small, supported CryoSPARC External Job adapter.

The workflow modules use this boundary instead of depending on CryoSPARC's
Dataset slots and ExternalJobController methods directly.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryosparc import mrc


TARGET_CRYOSPARC_VERSION = "5.0.6"


@dataclass(frozen=True)
class ExternalJobSource:
    """A source output to connect to an External Job input."""

    job_uid: str
    output_name: str


SourceOutput = ExternalJobSource
AxisSourceOutput = ExternalJobSource


@dataclass(frozen=True)
class LoadedClassAverage:
    """One native class average loaded from a CryoSPARC template stack."""

    image: np.ndarray
    pixel_size_A: float
    source_index: int
    source_path: Path

    @property
    def pixel_size(self):
        """Compatibility spelling used by the existing workflow code."""

        return self.pixel_size_A


@dataclass(frozen=True)
class TemplateStack:
    """Typed template-stack data keyed by source blob index."""

    class_averages: dict[int, LoadedClassAverage]

    @property
    def images(self):
        return {
            source_index: template.image
            for source_index, template in self.class_averages.items()
        }

    @property
    def pixel_size_A(self):
        pixel_sizes = {
            template.pixel_size_A
            for template in self.class_averages.values()
        }
        if len(pixel_sizes) != 1:
            raise ValueError("all class averages must share one native pixel size")
        return next(iter(pixel_sizes))

    @property
    def pixel_size(self):
        return self.pixel_size_A


@dataclass(frozen=True)
class LoadedVolume:
    """Matching and presentation volume data loaded from one input."""

    matching_map: np.ndarray
    rendering_map: np.ndarray
    matching_pixel_size_A: float
    rendering_pixel_size_A: float
    matching_path: Path
    rendering_path: Path
    matching_dataset_path: str = ""
    rendering_dataset_path: str = ""

    @property
    def volume(self):
        return self.matching_map

    @property
    def map(self):
        return self.matching_map

    @property
    def pixel_size_A(self):
        return self.matching_pixel_size_A

    @property
    def rendering_volume(self):
        return self.rendering_map


class ExternalJobPublicationError(RuntimeError):
    """A staged output could not be published to CryoSPARC."""

    def __init__(self, output_name, error):
        self.output_name = output_name
        self.cause = error
        super().__init__(
            f"Could not publish External Job output={output_name!r}: {error}"
        )


class CryoSPARCExternalJobAdapter:
    """Adapt the supported CryoSPARC 5.0.6 External Job API.

    Workflow code only sees typed inputs and narrow output publishers. Dataset
    field names, source-path resolution, allocation, and job lifecycle stay in
    this module.
    """

    def __init__(self, project, workspace_uid, title=None, *, job=None):
        self.project = project
        self.workspace_uid = workspace_uid
        self.job = (
            job
            if job is not None
            else project.create_external_job(workspace_uid, title=title or "")
        )
        self._staged_outputs = []
        self._published = False

    @property
    def uid(self):
        return self.job.uid

    @property
    def directory(self):
        directory = getattr(self.job, "dir", None)
        if directory is None:
            directory = getattr(self.job, "directory", None)
        if callable(directory):
            directory = directory()
        return Path(directory)

    @property
    def resource_directory(self):
        return self.directory

    def add_template_input(self, name, source, *, title):
        self._add_and_connect_input(
            name=name,
            type="template",
            slots=["blob"],
            source=source,
            title=title,
        )

    def add_particle_input(self, name, source, *, slots, title):
        self._add_and_connect_input(
            name=name,
            type="particle",
            slots=slots,
            source=source,
            title=title,
        )

    def add_volume_input(self, name, source, *, rendering_map="map", title):
        slots = ["map"]
        if rendering_map == "sharpened":
            slots.append("map_sharp")
        elif rendering_map != "map":
            raise ValueError("rendering map must be 'map' or 'sharpened'")
        self._add_and_connect_input(
            name=name,
            type="volume",
            slots=slots,
            source=source,
            title=title,
        )

    def add_template_output(self, name, *, title):
        self.job.add_output(
            type="template",
            name=name,
            slots=["blob"],
            title=title,
        )

    def add_volume_output(self, name, *, title):
        self.job.add_output(
            type="volume",
            name=name,
            slots=["map"],
            title=title,
        )

    def read_template_stack(self, name):
        dataset = self.job.load_input(name)
        class_averages = {}
        stack_cache = {}
        for path_value, image_index, pixel_size in zip(
            dataset["blob/path"],
            dataset["blob/idx"],
            dataset["blob/psize_A"],
            strict=True,
        ):
            source_path = self.resolve_source_path(path_value)
            if source_path not in stack_cache:
                _, stack_cache[source_path] = mrc.read(source_path)
            stack = stack_cache[source_path]
            source_index = int(image_index)
            if source_index in class_averages:
                raise ValueError(
                    f"source Class Number {source_index + 1} occurs more than once"
                )
            image = stack[source_index] if stack.ndim == 3 else stack
            class_averages[source_index] = LoadedClassAverage(
                image=np.asarray(image),
                pixel_size_A=float(pixel_size),
                source_index=source_index,
                source_path=source_path,
            )
        if not class_averages:
            raise ValueError("Select 2D templates input is empty")
        return TemplateStack(class_averages)

    def read_volume(self, name, *, rendering_map="map"):
        if rendering_map not in {"map", "sharpened"}:
            raise ValueError("rendering map must be 'map' or 'sharpened'")
        dataset = self.job.load_input(name)
        matching_path = self.resolve_source_path(dataset["map/path"][0])
        _, matching_data = mrc.read(matching_path)
        matching_pixel_size_A = float(dataset["map/psize_A"][0])
        rendering_slot = "map_sharp" if rendering_map == "sharpened" else "map"
        rendering_dataset_path = _path_text(dataset[f"{rendering_slot}/path"][0])
        rendering_path = self.resolve_source_path(rendering_dataset_path)
        _, rendering_data = mrc.read(rendering_path)
        rendering_pixel_size_A = float(
            dataset[f"{rendering_slot}/psize_A"][0]
        )
        return LoadedVolume(
            matching_map=np.asarray(matching_data),
            rendering_map=np.asarray(rendering_data),
            matching_pixel_size_A=matching_pixel_size_A,
            rendering_pixel_size_A=rendering_pixel_size_A,
            matching_path=matching_path,
            rendering_path=rendering_path,
            matching_dataset_path=_path_text(dataset["map/path"][0]),
            rendering_dataset_path=rendering_dataset_path,
        )

    def read_particles(self, name):
        return self.job.load_input(name)

    def resolve_source_path(self, path):
        if isinstance(path, bytes):
            path = path.decode()
        path = Path(str(path).removeprefix(">"))
        if path.is_absolute():
            return path
        project_directory = getattr(self.project, "dir", None)
        if project_directory is None:
            project_directory = self.directory
        if callable(project_directory):
            project_directory = project_directory()
        return Path(project_directory) / path

    def path(self, relative_path):
        """Resolve a path for a file generated inside this External Job."""

        return self.resource_directory / Path(relative_path)

    def log(self, message):
        self.job.log(message)

    def safe_log(self, message):
        try:
            self.job.log(message)
        except Exception:
            pass

    def log_plot(self, figure, text, formats, savefig_kw=None):
        return self.job.log_plot(figure, text, formats, savefig_kw=savefig_kw)

    def set_status(self, message, callback=None):
        self.safe_log(message)
        if callback is not None:
            try:
                callback(message)
            except Exception:
                pass

    def set_warning(self, message, callback=None):
        self.set_status(message, callback)

    def stage_template_stack(self, name, filename, stack, *, pixel_size_A):
        filename = Path(filename)
        output_path = self.path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mrc.write(output_path, np.asarray(stack, dtype=np.float32), pixel_size_A)
        self._stage(
            name,
            kind="template",
            filename=filename,
            count=len(stack),
            shape=np.asarray(stack).shape[1:],
            pixel_size_A=pixel_size_A,
        )

    def stage_volume(self, name, filename, volume, *, pixel_size_A):
        filename = Path(filename)
        output_path = self.path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mrc.write(output_path, np.asarray(volume, dtype=np.float32), pixel_size_A)
        self._stage(
            name,
            kind="volume",
            filename=filename,
            count=1,
            shape=np.asarray(volume).shape,
            pixel_size_A=pixel_size_A,
        )

    def stage_volume_source(
        self, name, source_path, *, shape, pixel_size_A, dataset_path=None
    ):
        source_path = Path(source_path)
        try:
            filename = source_path.relative_to(self.resource_directory)
        except ValueError:
            filename = source_path
        self._stage(
            name,
            kind="volume",
            filename=filename,
            count=1,
            shape=tuple(shape),
            pixel_size_A=pixel_size_A,
            source_path=source_path,
            dataset_reference=dataset_path,
        )

    def attach_output_preview(
        self,
        output_name,
        image_path,
        *,
        warning_callback=None,
        warning_label=None,
        warning_formatter=None,
    ):
        try:
            self.job.set_output_image(output_name, image_path)
        except Exception as error:
            message = (
                warning_formatter(error)
                if warning_formatter is not None
                else (
                    "WARNING: Could not attach "
                    f"{warning_label or f'{output_name} preview'}; "
                    "scientific output remains available. "
                    f"{error}"
                )
            )
            self.set_warning(message, warning_callback)

    def attach_tile_preview(
        self,
        image_path,
        *,
        warning_callback=None,
        warning_label=None,
        warning_formatter=None,
    ):
        try:
            self.job.set_tile_image(image_path)
        except Exception as error:
            message = (
                warning_formatter(error)
                if warning_formatter is not None
                else (
                    "WARNING: Could not attach "
                    f"{warning_label or 'job tile preview'}; "
                    f"scientific output remains available. {error}"
                )
            )
            self.set_warning(message, warning_callback)

    def publish(self):
        if self._published:
            return
        for staged in self._staged_outputs:
            try:
                output = self.job.alloc_output(staged.name, staged.count)
                path_value = staged.dataset_path(self.uid)
                if staged.kind == "template":
                    output["blob/path"][:] = path_value
                    output["blob/idx"][:] = np.arange(staged.count)
                    output["blob/shape"][:] = staged.shape
                    output["blob/psize_A"][:] = staged.pixel_size_A
                else:
                    output["map/path"][:] = path_value
                    output["map/shape"][:] = staged.shape
                    output["map/psize_A"][:] = staged.pixel_size_A
                self.job.save_output(staged.name, output)
            except Exception as error:
                raise ExternalJobPublicationError(staged.name, error) from error
        self._published = True

    @contextmanager
    def run(self):
        with self.job.run():
            yield self
            self.publish()

    def _add_and_connect_input(self, *, name, type, slots, source, title):
        self.job.add_input(
            type=type,
            name=name,
            min=1,
            max=1,
            slots=slots,
            title=title,
        )
        self.job.connect(name, source.job_uid, source.output_name)

    def _stage(
        self,
        name,
        *,
        kind,
        filename,
        count,
        shape,
        pixel_size_A,
        source_path=None,
        dataset_reference=None,
    ):
        if any(item.name == name for item in self._staged_outputs):
            raise ValueError(f"External Job output {name!r} is already staged")
        self._staged_outputs.append(
            _StagedOutput(
                name=name,
                kind=kind,
                filename=Path(filename),
                count=int(count),
                shape=tuple(shape),
                pixel_size_A=float(pixel_size_A),
                source_path=source_path,
                dataset_reference=dataset_reference,
            )
        )


def read_template_stack_dataset(project, dataset):
    """Read a template Dataset through the same adapter boundary.

    This compatibility helper supports callers that already loaded a Dataset
    while keeping path and slot interpretation in the adapter module.
    """

    class _LoadedDatasetJob:
        def load_input(self, name):
            return dataset

    adapter = CryoSPARCExternalJobAdapter(
        project,
        workspace_uid="",
        job=_LoadedDatasetJob(),
    )
    return adapter.read_template_stack("templates")


@dataclass(frozen=True)
class _StagedOutput:
    name: str
    kind: str
    filename: Path
    count: int
    shape: tuple[int, ...]
    pixel_size_A: float
    source_path: Path | None = None
    dataset_reference: str | None = None

    def dataset_path(self, job_uid):
        if self.source_path is not None:
            return self.dataset_reference or str(self.source_path)
        return f">{job_uid}/{self.filename}"


def _path_text(path):
    if isinstance(path, bytes):
        return path.decode()
    return str(path)


class InMemoryExternalJobAdapter:
    """A shared in-memory External Job seam for workflow tests.

    It intentionally implements the small project/backend surface consumed by
    :class:`CryoSPARCExternalJobAdapter`, so both workflows can use one test
    double without duplicating CryoSPARC Dataset mechanics.
    """

    def __init__(
        self,
        directory,
        datasets=None,
        *,
        uid="J99",
        output_image_error=None,
        tile_image_error=None,
        fail_output=None,
        log_error=None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.datasets = dict(datasets or {})
        self.uid = uid
        self.output_image_error = output_image_error
        self.tile_image_error = tile_image_error
        self.fail_output = fail_output
        self.log_error = log_error
        self.inputs = []
        self.connections = []
        self.outputs = []
        self.saved = {}
        self.saved_outputs = {}
        self.logs = []
        self.plots = []
        self.output_images = {}
        self.tile_images = []
        self._output_kinds = {}
        self.project = self
        self.created = None
        self._adapter = CryoSPARCExternalJobAdapter(
            self,
            "W1",
            title="In-memory External Job",
            job=self,
        )

    def create_external_job(self, workspace_uid, title):
        self.created = (workspace_uid, title)
        return self

    def add_input(self, **spec):
        self.inputs.append(spec)

    def connect(self, target_input, source_job_uid, source_output):
        self.connections.append((target_input, source_job_uid, source_output))

    def add_output(self, **spec):
        name = spec["name"]
        self.outputs.append(spec)
        self._output_kinds[name] = spec["type"]
        return name

    def alloc_output(self, name, count):
        if self.fail_output == name:
            raise OSError(f"publish unavailable for {name}")
        if self._output_kinds.get(name) == "volume":
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
        self.saved[name] = dataset
        self.saved_outputs[name] = (dataset, image)

    def load_input(self, name):
        return self.datasets[name]

    def run(self):
        return nullcontext(self)

    def dir(self):
        return str(self.directory)

    def log(self, message):
        if self.log_error is not None:
            raise self.log_error
        self.logs.append(message)

    def log_plot(self, figure, text, formats, savefig_kw=None):
        self.plots.append((figure, text, formats, savefig_kw))

    def set_output_image(self, name, image, savefig_kw=None):
        if self.output_image_error is not None:
            raise self.output_image_error
        self.output_images[name] = image

    def set_tile_image(self, image, savefig_kw=None):
        if self.tile_image_error is not None:
            raise self.tile_image_error
        self.tile_images.append(image)

    # Explicit high-level forwarding keeps this test double usable as a
    # project while avoiding a magic __getattr__ bridge between two APIs.
    @property
    def resource_directory(self):
        return self._adapter.resource_directory

    def add_template_input(self, *args, **kwargs):
        return self._adapter.add_template_input(*args, **kwargs)

    def add_particle_input(self, *args, **kwargs):
        return self._adapter.add_particle_input(*args, **kwargs)

    def add_volume_input(self, *args, **kwargs):
        return self._adapter.add_volume_input(*args, **kwargs)

    def add_template_output(self, *args, **kwargs):
        return self._adapter.add_template_output(*args, **kwargs)

    def add_volume_output(self, *args, **kwargs):
        return self._adapter.add_volume_output(*args, **kwargs)

    def read_template_stack(self, *args, **kwargs):
        return self._adapter.read_template_stack(*args, **kwargs)

    def read_volume(self, *args, **kwargs):
        return self._adapter.read_volume(*args, **kwargs)

    def read_particles(self, *args, **kwargs):
        return self._adapter.read_particles(*args, **kwargs)

    def resolve_source_path(self, *args, **kwargs):
        return self._adapter.resolve_source_path(*args, **kwargs)

    def path(self, *args, **kwargs):
        return self._adapter.path(*args, **kwargs)

    def stage_template_stack(self, *args, **kwargs):
        return self._adapter.stage_template_stack(*args, **kwargs)

    def stage_volume(self, *args, **kwargs):
        return self._adapter.stage_volume(*args, **kwargs)

    def stage_volume_source(self, *args, **kwargs):
        return self._adapter.stage_volume_source(*args, **kwargs)

    def attach_output_preview(self, *args, **kwargs):
        return self._adapter.attach_output_preview(*args, **kwargs)

    def attach_tile_preview(self, *args, **kwargs):
        return self._adapter.attach_tile_preview(*args, **kwargs)

    def publish(self):
        return self._adapter.publish()

    def run_adapter(self):
        return self._adapter.run()
