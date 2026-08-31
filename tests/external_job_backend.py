"""Shared in-memory CryoSPARC backend for External Job tests."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np


class InMemoryExternalJobBackend:
    """Implement the CryoSPARC surface used by the production adapter."""

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
        self.created = None

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
