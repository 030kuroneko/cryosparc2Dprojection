import numpy as np
from PIL import Image

from cryosparc_2d_projection.surface_render import (
    build_surface_model,
    write_surface_render_pair,
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


def test_exact_and_oblique_surface_renders_are_saved_as_distinct_square_images(
    tmp_path,
):
    z, y, x = np.mgrid[-1:1:25j, -1:1:25j, -1:1:25j]
    volume = (1.0 - np.sqrt((x / 0.8) ** 2 + (y / 0.5) ** 2 + (z / 0.3) ** 2))
    surface = build_surface_model(volume, surface_level=0.2, max_size=25)

    renders = write_surface_render_pair(
        tmp_path,
        surface=surface,
        rotation_matrix=np.eye(3),
        class_number=3,
        match_score=0.982,
        match_confidence="high",
        symmetry_label="5-fold",
        symmetry_distance_degrees=1.2,
        oblique_tilt_degrees=20,
        image_size=128,
        background="dark",
    )

    assert renders.exact_path == tmp_path / "class_003_exact.png"
    assert renders.oblique_path == tmp_path / "class_003_oblique.png"
    exact = np.asarray(Image.open(renders.exact_path).convert("RGB"))
    oblique = np.asarray(Image.open(renders.oblique_path).convert("RGB"))
    assert exact.shape == (128, 128, 3)
    assert oblique.shape == (128, 128, 3)
    assert exact[0, 0].max() < 20
    assert exact.max() > 200
    assert exact[:, 0].max() < 20
    assert exact[:, -1].max() < 20
    assert exact[0].max() < 20
    assert not np.array_equal(exact, oblique)


def test_surface_model_removes_density_islands_smaller_than_one_percent_of_main_body():
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume[11:21, 11:21, 11:21] = 1.0
    volume[28:30, 28:30, 28:30] = 1.0

    surface = build_surface_model(volume, surface_level=0.5, max_size=32)

    assert np.max(np.abs(surface.vertices)) < 8
