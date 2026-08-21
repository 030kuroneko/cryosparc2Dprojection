import numpy as np
from PIL import Image

from cryosparc_2d_projection.surface_render import (
    build_surface_model,
    write_camera_view_render,
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
