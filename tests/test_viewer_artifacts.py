from types import SimpleNamespace

import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import label as label_components

from cryosparc_2d_projection.auto_crop import compute_auto_crop_2d_framing
from cryosparc_2d_projection.surface_render import SurfaceSilhouetteBounds
from cryosparc_2d_projection.viewer import (
    create_class_preview_figure,
    create_class_preview_pages,
    write_chimerax_bundle,
)
from cryosparc_2d_projection.presentation import ComparisonRenderOptions


_VISIBLE_LUMINANCE_FRACTION = 0.075


def test_chimerax_bundle_reproduces_each_class_camera(tmp_path):
    cameras = {
        0: SimpleNamespace(
            rotation_matrix=np.array(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
        )
    }

    written = write_chimerax_bundle(
        tmp_path,
        map_path="/project P1/J20/volume.mrc",
        cameras=cameras,
    )

    assert written == [tmp_path / "class_001.cxc", tmp_path / "all_classes.cxc"]
    class_script = (tmp_path / "class_001.cxc").read_text()
    assert 'open "/project P1/J20/volume.mrc"' in class_script
    assert "camera ortho" in class_script
    assert (
        "view matrix models #1,0,1,0,0,-1,0,0,0,0,0,1,0"
        in class_script
    )
    master_script = (tmp_path / "all_classes.cxc").read_text()
    assert "view name class_001" in master_script
    assert "view class_001" in master_script


def test_class_preview_pages_contain_at_most_ten_three_column_rows(tmp_path):
    class_ids = list(range(11))
    class_averages = {
        class_id: SimpleNamespace(image=np.full((3, 3), class_id))
        for class_id in class_ids
    }
    projections = np.asarray([np.full((3, 3), class_id) for class_id in class_ids])
    cameras = {
        class_id: SimpleNamespace(match_score=0.9) for class_id in class_ids
    }
    orientations = {
        class_id: SimpleNamespace(particle_count=100, angular_spread_degrees=4.0)
        for class_id in class_ids
    }
    render_paths = {}
    for class_id in class_ids:
        exact_path = tmp_path / f"class_{class_id + 1:03d}_exact.png"
        Image.new("RGB", (8, 8), "white").save(exact_path)
        render_paths[class_id] = exact_path

    pages = create_class_preview_pages(
        class_averages,
        projections,
        cameras,
        orientations,
        render_paths,
        diagnostic_scores={
            class_id: SimpleNamespace(
                score=0.75,
                valid=True,
                metadata={
                    "band_low_resolution_A_effective": 80.0,
                    "band_high_resolution_A_effective": 15.0,
                },
            )
            for class_id in class_ids
        },
        comparison_options=ComparisonRenderOptions(dpi=300, page_size=10),
    )

    assert len(pages) == 2
    assert len(pages[0].axes) == 30
    assert len(pages[1].axes) == 3
    assert pages[0].dpi == 300
    assert pages[1].dpi == 300
    assert pages[0].axes[2].get_title() == "Camera View Render"
    assert pages[0].axes[1].get_title() == (
        "Matched | search raw=0.900\nband (80–15 Å)=0.750"
    )


def test_class_result_displays_2d_images_in_cryosparc_display_orientation(tmp_path):
    class_average = np.array([[1, 2, 3], [4, 5, 6]])
    matched_projection = np.array([[7, 8, 9], [10, 11, 12]])
    render_path = tmp_path / "class_001_exact.png"
    Image.new("RGB", (8, 8), "white").save(render_path)

    page = create_class_preview_pages(
        {0: SimpleNamespace(image=class_average)},
        np.asarray([matched_projection]),
        {0: SimpleNamespace(match_score=0.9)},
        {0: SimpleNamespace(particle_count=1, angular_spread_degrees=0.0)},
        {0: render_path},
    )[0]

    displayed_class = np.asarray(page.axes[0].images[0].get_array())
    displayed_projection = np.asarray(page.axes[1].images[0].get_array())
    assert np.array_equal(displayed_class, [[4, 5, 6], [1, 2, 3]])
    assert np.array_equal(displayed_projection, [[10, 11, 12], [7, 8, 9]])
    assert not np.array_equal(displayed_class, np.fliplr(class_average))
    assert not np.array_equal(displayed_projection, np.fliplr(matched_projection))
    assert page.axes[0].images[0].get_interpolation() == "hanning"
    assert page.axes[1].images[0].get_interpolation() == "hanning"


def test_class_result_applies_one_auto_crop_decision_to_both_2d_panels(tmp_path):
    class_average = np.zeros((32, 32), dtype=np.float32)
    class_average[12:20, 13:21] = 1.0
    render_path = tmp_path / "class_001_exact.png"
    Image.new("RGB", (8, 8), "white").save(render_path)
    decision = compute_auto_crop_2d_framing(
        [np.flipud(class_average)],
        [SurfaceSilhouetteBounds(0.25, 0.25, 0.75, 0.75)],
        enabled=True,
    )

    page = create_class_preview_pages(
        {0: type("ClassAverage", (), {"image": class_average})()},
        np.asarray([class_average]),
        {0: type("Camera", (), {"match_score": 0.9})()},
        {0: type("Orientation", (), {"particle_count": 1, "angular_spread_degrees": 0.0})()},
        {0: render_path},
        comparison_options=ComparisonRenderOptions(auto_crop_2d=True),
        auto_crop_decisions={0: decision},
    )[0]

    displayed_class = np.asarray(page.axes[0].images[0].get_array())
    displayed_projection = np.asarray(page.axes[1].images[0].get_array())
    assert displayed_class.shape == displayed_projection.shape == (32, 32)
    assert np.array_equal(displayed_class, np.flipud(class_average))
    assert np.array_equal(displayed_projection, np.flipud(class_average))
    assert np.allclose(page.axes[0].get_xlim(), (8.5, 24.5))
    assert np.allclose(page.axes[0].get_ylim(), (23.5, 7.5))
    assert np.allclose(page.axes[1].get_xlim(), (8.5, 24.5))
    assert np.allclose(page.axes[1].get_ylim(), (23.5, 7.5))
    assert page.axes[2].images[0].get_array().shape == (8, 8, 3)


def test_disabled_auto_crop_keeps_rendered_preview_pixel_identical(tmp_path):
    class_average = np.zeros((32, 32), dtype=np.float32)
    class_average[12:20, 13:21] = 1.0
    render_path = tmp_path / "class_001_exact.png"
    Image.new("RGB", (8, 8), "white").save(render_path)
    inputs = (
        {0: SimpleNamespace(image=class_average)},
        np.asarray([class_average]),
        {0: SimpleNamespace(match_score=0.9)},
        {0: SimpleNamespace(particle_count=1, angular_spread_degrees=0.0)},
        {0: render_path},
    )
    decision = compute_auto_crop_2d_framing(
        [np.flipud(class_average)],
        [SurfaceSilhouetteBounds(0.25, 0.25, 0.75, 0.75)],
        enabled=True,
    )

    baseline = create_class_preview_pages(*inputs)[0]
    disabled = create_class_preview_pages(
        *inputs,
        comparison_options=ComparisonRenderOptions(auto_crop_2d=False),
        auto_crop_decisions={0: decision},
    )[0]
    baseline_canvas = FigureCanvasAgg(baseline)
    disabled_canvas = FigureCanvasAgg(disabled)
    baseline_canvas.draw()
    disabled_canvas.draw()

    assert np.array_equal(
        np.asarray(baseline_canvas.buffer_rgba()),
        np.asarray(disabled_canvas.buffer_rgba()),
    )


def test_auto_crop_aligns_the_rendered_matched_panel_with_camera_silhouette(
    tmp_path,
):
    size = 128
    rows, columns = np.indices((size, size))
    delta_x = columns - (size - 1) / 2.0
    delta_y = rows - (size - 1) / 2.0
    matched_projection = np.zeros((size, size), dtype=np.float32)

    # Keep the core smoothly varying so the fixture exercises raster display,
    # while retaining a connected weak halo just above the old 5% cutoff.
    core = (np.abs(delta_x) < 29.0) & (np.abs(delta_y) < 29.0)
    core_profile = 6.2 + 2.0 * np.exp(
        -((delta_x / 24.0) ** 2 + (delta_y / 22.0) ** 2)
    )
    matched_projection[core] = core_profile[core]
    halo = (
        (columns >= 33)
        & (columns < 94)
        & (rows >= 34)
        & (rows < 94)
        & ~core
    )
    halo_distance = np.maximum(np.abs(delta_x), np.abs(delta_y)) - 28.5
    halo_profile = 0.52 + 0.08 * np.clip(
        (31.0 - halo_distance) / 2.5,
        0.0,
        1.0,
    )
    matched_projection[halo] = halo_profile[halo]
    for row, column, value in (
        (62, 63, 10.0),
        (63, 64, 9.7),
        (64, 63, 9.4),
        (61, 64, 9.2),
        (65, 62, 8.9),
        (63, 61, 8.7),
    ):
        matched_projection[row, column] = value

    camera_silhouette = np.zeros((size, size), dtype=np.uint8)
    camera_silhouette[18:109, 18:109] = 255
    render_path = tmp_path / "class_001_silhouette.png"
    Image.fromarray(camera_silhouette, mode="L").save(render_path)

    class_average = SimpleNamespace(image=matched_projection)
    camera = SimpleNamespace(match_score=0.9)
    orientation = SimpleNamespace(particle_count=1, angular_spread_degrees=0.0)
    silhouette = SurfaceSilhouetteBounds(0.1425, 0.1425, 0.8575, 0.8575)
    decision = compute_auto_crop_2d_framing(
        [matched_projection], [silhouette], enabled=True
    )
    assert decision.crop_shape == (82, 82)

    figure = create_class_preview_figure(
        {0: class_average},
        np.asarray([matched_projection]),
        {0: camera},
        {0: orientation},
        {0: render_path},
        comparison_options=ComparisonRenderOptions(auto_crop_2d=True),
        class_ids=[0],
        auto_crop_decisions={0: decision},
    )
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())

    def axis_raster(axis):
        left, bottom, width, height = axis.get_window_extent().bounds
        # The extent is a display-space boundary.  Keep a one-pixel interior
        # margin so antialiased axes edges cannot become a foreground object.
        x_start = int(np.ceil(left)) + 1
        x_stop = int(np.floor(left + width)) - 1
        y_start = int(np.ceil(bottom)) + 1
        y_stop = int(np.floor(bottom + height)) - 1
        return rgba[
            rgba.shape[0] - y_stop : rgba.shape[0] - y_start,
            x_start:x_stop,
            :3,
        ]

    def visible_occupancy(axis):
        raster = axis_raster(axis)
        luminance = raster.astype(np.float32).mean(axis=2)
        border = np.concatenate(
            [
                luminance[0, :],
                luminance[-1, :],
                luminance[1:-1, 0],
                luminance[1:-1, -1],
            ]
        )
        background = float(np.median(border))
        deviations = np.abs(luminance - background)
        visibility_threshold = _VISIBLE_LUMINANCE_FRACTION * float(
            deviations.max()
        )
        visible = deviations >= visibility_threshold
        components, component_count = label_components(visible)
        assert component_count > 0
        component_sizes = np.bincount(components.ravel())[1:]
        largest = components == (int(np.argmax(component_sizes)) + 1)
        visible_rows, visible_columns = np.nonzero(largest)
        visible_width = visible_columns.max() - visible_columns.min() + 1
        visible_height = visible_rows.max() - visible_rows.min() + 1
        return max(visible_width, visible_height) / raster.shape[0]

    matched_occupancy = visible_occupancy(figure.axes[1])
    camera_occupancy = visible_occupancy(figure.axes[2])

    assert abs(matched_occupancy - camera_occupancy) <= 0.03
