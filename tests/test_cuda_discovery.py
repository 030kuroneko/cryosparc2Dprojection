"""CUDA device discovery behavior without requiring CUDA hardware."""

import sys
import types

import numpy as np
import pytest
from scipy.ndimage import affine_transform

from cryosparc_2d_projection.image_camera import (
    ImageCameraSearchConfig,
    solve_class_cameras_from_images,
)


CUDA_ERROR_INSUFFICIENT_DRIVER = 35
CUDA_ERROR_NO_DEVICE = 100


def _install_fake_cupy(monkeypatch, *, status):
    class FakeCUDARuntimeError(RuntimeError):
        def __init__(self, error_status):
            self.status = error_status
            super().__init__(f"fake CUDA status {error_status}")

    runtime = types.SimpleNamespace(
        CUDARuntimeError=FakeCUDARuntimeError,
        cudaErrorInsufficientDriver=CUDA_ERROR_INSUFFICIENT_DRIVER,
        cudaErrorNoDevice=CUDA_ERROR_NO_DEVICE,
    )
    cp = types.ModuleType("cupy")
    cp.cuda = types.SimpleNamespace(runtime=runtime)
    monkeypatch.setitem(sys.modules, "cupy", cp)

    cupyx = types.ModuleType("cupyx")
    cupyx_scipy = types.ModuleType("cupyx.scipy")
    cupyx_ndimage = types.ModuleType("cupyx.scipy.ndimage")
    cupyx_ndimage.affine_transform = affine_transform
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", cupyx_ndimage)

    def get_device_count():
        raise FakeCUDARuntimeError(status)

    runtime.getDeviceCount = get_device_count
    return FakeCUDARuntimeError


def _inputs():
    volume = np.zeros((9, 9, 9), dtype=float)
    volume[1, 2, 3] = 1.0
    volume[6, 1, 5] = 2.0
    volume[7, 7, 2] = 3.0
    return volume, {0: volume.sum(axis=0)}


@pytest.mark.parametrize(
    "status",
    [
        CUDA_ERROR_INSUFFICIENT_DRIVER,
        CUDA_ERROR_NO_DEVICE,
    ],
)
def test_auto_falls_back_for_known_unavailable_cuda_statuses(monkeypatch, status):
    _install_fake_cupy(monkeypatch, status=status)
    volume, images = _inputs()
    warnings = []

    result = solve_class_cameras_from_images(
        images,
        volume,
        config=ImageCameraSearchConfig(device="auto"),
        warning_callback=warnings.append,
    )[0]

    assert result.search_metadata["device"] == "cpu"
    assert warnings
    assert str(status) in warnings[0]


def test_cuda_raises_for_known_unavailable_status_and_preserves_reason(monkeypatch):
    error_status = CUDA_ERROR_NO_DEVICE
    _install_fake_cupy(monkeypatch, status=error_status)
    volume, images = _inputs()

    with pytest.raises(RuntimeError, match=f"fake CUDA status {error_status}"):
        solve_class_cameras_from_images(
            images,
            volume,
            config=ImageCameraSearchConfig(device="cuda"),
        )


def test_unexpected_cuda_runtime_status_is_propagated(monkeypatch):
    unexpected_status = 999
    error_type = _install_fake_cupy(monkeypatch, status=unexpected_status)
    volume, images = _inputs()

    with pytest.raises(error_type, match=f"fake CUDA status {unexpected_status}"):
        solve_class_cameras_from_images(
            images,
            volume,
            config=ImageCameraSearchConfig(device="auto"),
        )
