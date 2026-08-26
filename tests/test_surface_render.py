import numpy as np
import pytest
from PIL import Image

from cryosparc_2d_projection.surface_render import (
    build_surface_model,
    write_camera_view_render,
    ClassRenderOptions,
    resolve_surface_sampling_grid,
    SurfaceRenderMemoryError,
    recommend_lower_surface_grid_size,
)


def test_surface_model_is_a_centered_triangle_mesh_at_requested_level():
    coordinates = np.linspace(-1.0, 1.0, 25)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    volume = (1.0 - np.sqrt(x**2 + y**2 + z**2)).astype(np.float32)

    surface = build_surface_model(volume, surface_level=0.2, max_size=25)

    assert surface.faces.ndim == 2
    assert surface.faces.shape[1] == 3
    assert len(surface.faces) > 0
    assert surface.vertices.shape[1] == 3
    assert np.allclose(surface.vertices.mean(axis=0), [0.0, 0.0, 0.0], atol=0.1)
    assert np.allclose(np.linalg.norm(surface.normals, axis=1), 1.0, atol=1e-5)
    assert surface.surface_level == 0.2


def test_surface_model_lowers_an_unusable_automatic_level_and_reports_it():
    volume = np.zeros((12, 12, 12), dtype=np.float32)
    volume[:6] = 1.0

    surface = build_surface_model(volume, surface_level=None, max_size=12)

    assert np.isclose(surface.surface_level, 0.875)
    assert surface.surface_level_was_automatic is True
    assert "lowered" in surface.warning


def test_camera_view_render_is_saved_without_an_oblique_inspection_render(
    tmp_path,
):
    z, y, x = np.mgrid[-1:1:25j, -1:1:25j, -1:1:25j]
    volume = (1.0 - np.sqrt((x / 0.8) ** 2 + (y / 0.5) ** 2 + (z / 0.3) ** 2))
    surface = build_surface_model(volume, surface_level=0.2, max_size=25)

    camera_view_path = write_camera_view_render(
        tmp_path,
        surface=surface,
        rotation_matrix=np.eye(3),
        class_number=3,
        image_size=128,
        background="dark",
    )

    assert camera_view_path == tmp_path / "class_003_exact.png"
    assert not (tmp_path / "class_003_oblique.png").exists()
    exact = np.asarray(Image.open(camera_view_path).convert("RGB"))
    assert exact.shape == (128, 128, 3)
    assert exact[0, 0].max() < 20
    assert exact.max() > 200
    assert exact[:, 0].max() < 20
    assert exact[:, -1].max() < 20
    assert exact[0].max() < 20
    assert exact[35:115, 10:118].max() > 200


def test_camera_view_render_uses_vertical_display_flip_without_horizontal_mirror(
    tmp_path,
):
    z, y, x = np.mgrid[-1:1:41j, -1:1:41j, -1:1:41j]
    main = (
        ((x - 0.35) / 0.3) ** 2
        + ((y - 0.35) / 0.3) ** 2
        + (z / 0.3) ** 2
    ) < 1
    tail = (
        ((x + 0.45) / 0.18) ** 2
        + ((y + 0.45) / 0.18) ** 2
        + (z / 0.18) ** 2
    ) < 1
    surface = build_surface_model((main | tail).astype(np.float32), surface_level=0.5)

    camera_view_path = write_camera_view_render(
        tmp_path,
        surface=surface,
        rotation_matrix=np.eye(3),
        class_number=1,
        image_size=256,
    )

    image = np.asarray(Image.open(camera_view_path).convert("L"))
    rows, columns = np.nonzero(image > 40)
    assert rows.mean() < image.shape[0] / 2
    assert columns.mean() > image.shape[1] / 2


def test_camera_view_render_contains_only_a_centered_surface_without_text(tmp_path):
    z, y, x = np.mgrid[-1:1:41j, -1:1:41j, -1:1:41j]
    volume = (x**2 + y**2 + z**2 < 0.45**2).astype(np.float32)
    surface = build_surface_model(volume, surface_level=0.5)

    camera_view_path = write_camera_view_render(
        tmp_path,
        surface=surface,
        rotation_matrix=np.eye(3),
        class_number=8,
        image_size=256,
    )

    image = np.asarray(Image.open(camera_view_path).convert("L"))
    bright_rows, bright_columns = np.nonzero(image > 40)
    assert image[:16].max() < 20
    assert 110 < bright_rows.mean() < 146
    assert 110 < bright_columns.mean() < 146


def test_automatic_surface_extraction_retries_at_a_lower_level(monkeypatch):
    import cryosparc_2d_projection.surface_render as rendering

    volume = np.linspace(0, 1, 12**3, dtype=np.float32).reshape((12, 12, 12))
    real_extractor = rendering._extract_triangle_mesh
    attempted_levels = []

    def fail_first_attempt(sampled, level):
        attempted_levels.append(level)
        if len(attempted_levels) == 1:
            raise RuntimeError("synthetic extraction failure")
        return real_extractor(sampled, level)

    monkeypatch.setattr(rendering, "_extract_triangle_mesh", fail_first_attempt)

    surface = rendering.build_surface_model(volume, surface_level=None, max_size=12)

    assert len(attempted_levels) == 2
    assert attempted_levels[1] < attempted_levels[0]
    assert surface.surface_level == attempted_levels[1]
    assert "lowered" in surface.warning


def test_surface_model_removes_density_islands_smaller_than_one_percent_of_main_body():
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume[11:21, 11:21, 11:21] = 1.0
    volume[28:30, 28:30, 28:30] = 1.0

    surface = build_surface_model(volume, surface_level=0.5, max_size=32)

    assert np.max(np.abs(surface.vertices)) < 8


def test_render_options_default_to_native_surface_grid_and_accept_large_manual_values():
    assert ClassRenderOptions().grid_size is None
    assert ClassRenderOptions(grid_size=384).grid_size == 384


def test_automatic_surface_sampling_grid_uses_native_non_cubic_shape():
    resolved = resolve_surface_sampling_grid((20, 10, 5))

    assert resolved.original_shape == (20, 10, 5)
    assert resolved.requested_grid_size is None
    assert resolved.effective_grid_size == 20
    assert resolved.sampled_shape == (20, 10, 5)
    assert resolved.grid_size_was_automatic is True
    assert resolved.was_downsampled is False
    assert resolved.warnings == ()


def test_manual_surface_sampling_grid_downsamples_each_axis_proportionally():
    resolved = resolve_surface_sampling_grid((20, 10, 5), requested_grid_size=8)

    assert resolved.original_shape == (20, 10, 5)
    assert resolved.requested_grid_size == 8
    assert resolved.effective_grid_size == 8
    assert resolved.sampled_shape == (8, 4, 2)
    assert resolved.grid_size_was_automatic is False
    assert resolved.was_downsampled is True


def test_manual_surface_sampling_grid_never_upsamples_the_rendering_map():
    resolved = resolve_surface_sampling_grid((20, 10, 5), requested_grid_size=100)

    assert resolved.effective_grid_size == 20
    assert resolved.sampled_shape == (20, 10, 5)
    assert resolved.was_downsampled is False


@pytest.mark.parametrize("requested", [1, 0, -3])
def test_surface_sampling_grid_rejects_invalid_manual_sizes(requested):
    with pytest.raises(ValueError, match="at least 2"):
        resolve_surface_sampling_grid((12, 12, 12), requested_grid_size=requested)


def test_surface_sampling_grid_estimate_is_recorded_and_warns_above_one_gib():
    resolved = resolve_surface_sampling_grid((512, 512, 512))

    assert resolved.estimated_memory_bytes > 1024**3
    assert len(resolved.warnings) == 1
    assert "512 x 512 x 512" in resolved.warnings[0]
    assert "1 GiB" in resolved.warnings[0]


def test_surface_sampling_grid_metadata_is_json_compatible():
    resolved = resolve_surface_sampling_grid((20, 10, 5), requested_grid_size=8)

    assert resolved.as_dict() == {
        "original_shape": [20, 10, 5],
        "requested_grid_size": 8,
        "effective_grid_size": 8,
        "sampled_shape": [8, 4, 2],
        "grid_size_was_automatic": False,
        "mode": "manual",
        "was_downsampled": True,
        "estimated_memory_bytes": 8 * 4 * 2 * 14,
        "estimated_memory_gib": (8 * 4 * 2 * 14) / (1024**3),
        "memory_estimate_includes": [
            "sampled float32 volume",
            "binary occupancy mask",
            "connected-component labels",
            "retained-component mask",
            "density cleanup copy",
        ],
        "memory_estimate_excludes": [
            "marching-cubes mesh",
            "plotting allocations",
        ],
        "warnings": [],
    }


def test_surface_model_exposes_resolved_surface_sampling_metadata():
    volume = np.zeros((12, 8, 4), dtype=np.float32)
    volume[2:10, 2:6, :] = 1.0

    surface = build_surface_model(volume, surface_level=0.5, max_size=6)

    assert surface.sampling_grid.requested_grid_size == 6
    assert surface.sampling_grid.sampled_shape == (6, 4, 2)
    assert surface.sampling_grid.was_downsampled is True


def test_surface_model_uses_the_pre_resolved_sampling_grid_object():
    volume = np.zeros((6, 4, 3), dtype=np.float32)
    volume[1:5, 1:3, :] = 1.0
    resolved = resolve_surface_sampling_grid((6, 4, 3), requested_grid_size=4)

    surface = build_surface_model(
        volume,
        surface_level=0.5,
        sampling_grid=resolved,
    )

    assert surface.sampling_grid is resolved


@pytest.mark.parametrize(
    "failed_grid, recommended_grid",
    [
        (1024, 512),
        (512, 384),
        (384, 256),
        (256, 192),
        (192, 128),
        (128, 64),
        (64, 32),
        (2, None),
    ],
)
def test_memory_failure_recommendation_uses_lower_grid_tiers(
    failed_grid, recommended_grid
):
    assert recommend_lower_surface_grid_size(failed_grid) == recommended_grid


def test_surface_extraction_memory_error_reports_shape_and_does_not_retry(
    monkeypatch,
):
    import cryosparc_2d_projection.surface_render as rendering

    volume = np.zeros((12, 12, 12), dtype=np.float32)
    volume[2:10, 2:10, 2:10] = 1.0
    attempts = []

    def fail_surface_extraction(sampled, level):
        attempts.append((sampled.shape, level))
        raise MemoryError("synthetic surface allocation failure")

    monkeypatch.setattr(
        rendering, "_extract_triangle_mesh", fail_surface_extraction
    )

    with pytest.raises(SurfaceRenderMemoryError) as raised:
        rendering.build_surface_model(volume, surface_level=0.5, max_size=8)

    error = raised.value
    assert attempts == [((8, 8, 8), 0.5)]
    assert error.stage == "surface extraction"
    assert error.sampled_shape == (8, 8, 8)
    assert error.effective_grid_size == 8
    assert error.recommended_grid_size == 4
    assert "8 x 8 x 8" in str(error)
    assert "--render-grid-size 4" in str(error)
    assert "did not retry" in str(error)


def test_png_render_memory_error_reports_shape_and_does_not_retry(
    tmp_path, monkeypatch
):
    import cryosparc_2d_projection.surface_render as rendering

    z, y, x = np.mgrid[-1:1:17j, -1:1:17j, -1:1:17j]
    volume = (x**2 + y**2 + z**2 < 0.6**2).astype(np.float32)
    surface = rendering.build_surface_model(volume, surface_level=0.5, max_size=8)
    attempts = []

    def fail_png_render(*args, **kwargs):
        attempts.append(True)
        raise MemoryError("synthetic PNG allocation failure")

    monkeypatch.setattr(rendering, "_write_surface_image", fail_png_render)

    with pytest.raises(SurfaceRenderMemoryError) as raised:
        rendering.write_camera_view_render(
            tmp_path,
            surface=surface,
            rotation_matrix=np.eye(3),
            class_number=1,
            image_size=128,
        )

    error = raised.value
    assert attempts == [True]
    assert error.stage == "PNG rendering"
    assert error.sampled_shape == (8, 8, 8)
    assert error.effective_grid_size == 8
    assert error.recommended_grid_size == 4
    assert "8 x 8 x 8" in str(error)
    assert "--render-grid-size 4" in str(error)
    assert "did not retry" in str(error)


def test_validation_errors_are_not_relabelled_as_memory_failures():
    with pytest.raises(ValueError, match="at least 2") as raised:
        build_surface_model(
            np.zeros((8, 8, 8), dtype=np.float32),
            surface_level=0.5,
            max_size=1,
        )

    assert not isinstance(raised.value, SurfaceRenderMemoryError)
