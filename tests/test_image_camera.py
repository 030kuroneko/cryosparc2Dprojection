"""Image-only camera behavior through the public search boundary."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.image_camera import (
    ImageCameraSearchConfig,
    solve_class_cameras_from_images,
)


def asymmetric_volume(size=17):
    volume = np.zeros((size, size, size), dtype=np.float32)
    for z, y, x, weight in [(4, 5, 6, 1), (11, 4, 10, 2),
                            (8, 12, 5, 3), (5, 10, 12, 4)]:
        volume[z, y, x] = weight
    return gaussian_filter(volume, 0.8)


def test_image_only_search_recovers_camera_without_particle_poses():
    volume = asymmetric_volume()
    # Independent exact array operations: x viewing axis followed by a quarter
    # turn in the image plane. No production projector builds this target.
    target = np.rot90(volume.sum(axis=2))
    result = solve_class_cameras_from_images(
        {7: target}, volume, config=ImageCameraSearchConfig(device="cpu")
    )[7]

    expected = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
    distance = np.rad2deg(Rotation.from_matrix(result.rotation_matrix @ expected.T).magnitude())
    assert distance < 4
    assert result.match_score > 0.98
    assert result.orientation_method == "image_global_search"
    assert result.search_metadata["device"] == "cpu"
    assert result.search_evaluation_count > 1


def test_auto_device_reports_cpu_fallback_when_cupy_is_unavailable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cupy", None)
    volume = asymmetric_volume()
    warnings = []
    result = solve_class_cameras_from_images(
        {0: volume.sum(axis=0)}, volume,
        config=ImageCameraSearchConfig(device="auto"), warning_callback=warnings.append,
    )[0]

    assert result.search_metadata["device"] == "cpu"
    assert result.search_metadata["requested_device"] == "auto"
    assert warnings and "CPU" in warnings[0]


def test_cuda_camera_agrees_with_cpu_when_hardware_is_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No usable CUDA runtime")
    volume = asymmetric_volume()
    target = np.rot90(volume.sum(axis=2))
    cpu = solve_class_cameras_from_images(
        {0: target}, volume, config=ImageCameraSearchConfig(device="cpu"),
    )[0]
    gpu = solve_class_cameras_from_images(
        {0: target}, volume, config=ImageCameraSearchConfig(device="cuda"),
    )[0]
    assert gpu.search_metadata["device"] == "cuda"
    assert np.allclose(cpu.rotation_matrix, gpu.rotation_matrix, atol=1e-5)
    assert np.allclose(cpu.projection_shift_pixels, gpu.projection_shift_pixels)
    assert abs(cpu.match_score - gpu.match_score) < 1e-5


def test_global_search_returns_the_actual_selection_grid_and_shift_units():
    size = 35
    xyz = np.stack(np.meshgrid(*([np.arange(size) - (size-1)/2]*3), indexing="ij"))
    volume = np.zeros((size, size, size))
    target = np.zeros((size, size))
    matrix = Rotation.from_euler("xyz", [37., -24., 61.], degrees=True).as_matrix()
    expected_shift = np.array([2., -1.])
    # Analytic Gaussian projections are independent of the interpolating
    # production projector. Input volume axes are Z,Y,X; camera coordinates X,Y,Z.
    for center, weight, sigma in [((-7, -5, -4), 1., 1.2), ((6, -7, 5), 2., 1.5),
                                  ((-6, 7, 2), 3., 1.1), ((7, 5, -6), 4., 1.4)]:
        position = np.array(center, dtype=float)
        volume += weight * np.exp(-np.sum((xyz - position[::-1, None, None, None])**2, axis=0)/(2*sigma**2))
        projected = (matrix @ position)[:2] + expected_shift
        yy, xx = np.indices(target.shape) - (size-1)/2
        target += weight*np.sqrt(2*np.pi)*sigma*np.exp(-((xx-projected[0])**2 + (yy-projected[1])**2)/(2*sigma**2))
    result = solve_class_cameras_from_images(
        {0: target}, volume, config=ImageCameraSearchConfig(device="cpu"),
    )[0]
    assert result.search_metadata["selection_box_size"] <= 32
    assert result.matched_projection.shape == (32, 32)
    assert np.allclose(result.projection_shift_pixels,
                       result.search_metadata["selection_shift_pixels"])
    error_degrees = np.rad2deg(Rotation.from_matrix(result.rotation_matrix @ matrix.T).magnitude())
    assert error_degrees < 5
    scale = result.search_metadata["selection_pixel_size_in_input_pixels"]
    assert np.allclose(result.projection_shift_pixels * scale, expected_shift, atol=1)


def _simulate_cuda_dependency(monkeypatch, *, max_batch=None, compute_error=None):
    """CPU stand-in for the external CUDA API, only for failure-path tests."""
    import sys
    import types
    from scipy.ndimage import affine_transform

    class OutOfMemoryError(Exception):
        pass

    cp = types.ModuleType("cupy")
    cp.__getattr__ = lambda name: getattr(np, name)
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDeviceCount=lambda: 1, CUDARuntimeError=RuntimeError),
        memory=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError),
    )
    cp.asnumpy = np.asarray
    cp.get_default_memory_pool = lambda: types.SimpleNamespace(free_all_blocks=lambda: None)

    def stack(arrays, **kwargs):
        if max_batch is not None and len(arrays) > max_batch:
            raise OutOfMemoryError("simulated allocation failure")
        return np.stack(arrays, **kwargs)

    def transform(*args, **kwargs):
        if compute_error:
            raise compute_error
        return affine_transform(*args, **kwargs)

    cp.stack = stack
    monkeypatch.setitem(sys.modules, "cupy", cp)
    for name in ("cupyx", "cupyx.scipy", "cupyx.scipy.ndimage"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["cupyx.scipy.ndimage"].affine_transform = transform


@pytest.mark.parametrize("max_batch, expected_device, expected_batch", [(4, "cuda", 4), (0, "cpu", 16)])
def test_gpu_memory_recovery_is_visible_and_preserves_search_result(monkeypatch, max_batch,
                                                                  expected_device, expected_batch):
    _simulate_cuda_dependency(monkeypatch, max_batch=max_batch)
    volume = asymmetric_volume()
    warnings = []
    result = solve_class_cameras_from_images(
        {0: volume.sum(axis=0)}, volume,
        config=ImageCameraSearchConfig(device="auto"), warning_callback=warnings.append,
    )[0]
    assert result.search_metadata["device"] == expected_device
    assert result.search_metadata["batch_size"] == expected_batch
    assert result.match_score > 0.99
    assert any("batch size 4" in message for message in warnings)
    if expected_device == "cpu":
        assert any("restarting" in message and "CPU" in message for message in warnings)


def test_gpu_computation_errors_are_not_silently_retried_on_cpu(monkeypatch):
    _simulate_cuda_dependency(monkeypatch, compute_error=RuntimeError("kernel failure"))
    volume = asymmetric_volume()
    with pytest.raises(RuntimeError, match="kernel failure"):
        solve_class_cameras_from_images({0: volume.sum(axis=0)}, volume)


def test_symmetry_equivalent_cameras_do_not_compete_as_distinct_answers():
    volume = asymmetric_volume()
    volume += np.rot90(volume, 2, axes=(1, 2))
    result = solve_class_cameras_from_images(
        {0: volume.sum(axis=0)}, volume, symmetry="C2",
        config=ImageCameraSearchConfig(device="cpu"),
    )[0]
    # C2 about Z is exactly diag(-1,-1,1), independent of the production registry.
    operators = [np.eye(3), np.diag([-1., -1., 1.])]
    matrices = [result.rotation_matrix] + [np.array(item["rotation_matrix"])
                                           for item in result.alternative_orientations]
    for index, left in enumerate(matrices):
        for right in matrices[index+1:]:
            distance = min(np.rad2deg(Rotation.from_matrix(left @ op @ right.T).magnitude())
                           for op in operators)
            assert distance > 5 - 1e-6
    assert result.match_score > 0.99


def test_image_search_is_independent_of_positive_density_scale():
    volume = asymmetric_volume()
    result = solve_class_cameras_from_images(
        {0: volume.sum(axis=0) * 1e-10}, volume * 1e-12,
        config=ImageCameraSearchConfig(device="cpu"),
    )[0]
    assert result.match_score > 0.99
    assert np.allclose(result.rotation_matrix, np.eye(3), atol=0.04)


def test_indistinguishable_nonequivalent_views_are_reported_as_uncertain():
    zz, yy, xx = np.indices((9, 9, 9)) - 4
    volume = np.exp(-(xx**2 + yy**2 + zz**2)/3.)
    result = solve_class_cameras_from_images(
        {0: volume.sum(axis=0)}, volume, symmetry="C1",
        config=ImageCameraSearchConfig(device="cpu"),
    )[0]
    assert result.match_score > 0.99
    assert result.match_confidence == "low"
    assert result.score_margin < 0.03
    assert result.alternative_orientations


@pytest.mark.parametrize("quality", ["standard", "fine"])
def test_quality_presets_recover_the_same_simple_camera(quality):
    volume = asymmetric_volume()
    result = solve_class_cameras_from_images(
        {0: volume.sum(axis=0)}, volume,
        config=ImageCameraSearchConfig(quality=quality, device="cpu"),
    )[0]
    assert result.match_score > 0.99
    assert np.allclose(result.rotation_matrix, np.eye(3), atol=0.04)
    assert result.search_metadata["quality"] == quality
