from pathlib import Path

import numpy as np

from cryosparc_2d_projection.presentation import ComparisonRenderOptions


def write_matched_projection_thumbnail(path, projection):
    """Write one native-grid Matched Projection as a UI-only grayscale PNG."""
    from PIL import Image

    displayed = np.flipud(np.asarray(projection, dtype=np.float32))
    finite = np.isfinite(displayed)
    if not finite.any():
        pixels = np.zeros(displayed.shape, dtype=np.uint8)
    else:
        low = float(np.min(displayed[finite]))
        high = float(np.max(displayed[finite]))
        if high == low:
            pixels = np.zeros(displayed.shape, dtype=np.uint8)
        else:
            scaled = (np.nan_to_num(displayed, nan=low) - low) / (high - low)
            pixels = np.round(np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
    path = Path(path)
    Image.fromarray(pixels, mode="L").save(path)
    return path


def write_chimerax_bundle(output_directory, *, map_path, cameras):
    """Write per-class and master ChimeraX camera scripts."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    common = [f'open "{map_path}"', "camera ortho", "view orient"]
    written = []

    for class_id, camera in sorted(cameras.items()):
        class_number = class_id + 1
        script = [
            *common,
            _model_matrix_command(camera.rotation_matrix),
            "view #1",
            f"view name class_{class_number:03d}",
        ]
        path = output_directory / f"class_{class_number:03d}.cxc"
        path.write_text("\n".join(script) + "\n")
        written.append(path)

    master = list(common)
    for class_id, camera in sorted(cameras.items()):
        class_number = class_id + 1
        master.extend(
            [
                _model_matrix_command(camera.rotation_matrix),
                "view #1",
                f"view name class_{class_number:03d}",
            ]
        )
    if cameras:
        master.append(f"view class_{min(cameras) + 1:03d}")
    master_path = output_directory / "all_classes.cxc"
    master_path.write_text("\n".join(master) + "\n")
    written.append(master_path)
    return written


def create_class_preview_pages(
    class_averages,
    projections,
    cameras,
    orientations,
    render_paths,
    *,
    diagnostic_scores=None,
    comparison_options=None,
    page_size=None,
):
    """Create three-column Class Result pages with bounded row counts."""
    comparison_options = comparison_options or ComparisonRenderOptions()
    effective_page_size = (
        comparison_options.page_size if page_size is None else page_size
    )
    if type(effective_page_size) is not int or effective_page_size <= 0:
        raise ValueError("preview page size must be a positive integer")
    class_ids = sorted(orientations)
    return [
        create_class_preview_figure(
            class_averages,
            projections,
            cameras,
            orientations,
            render_paths,
            diagnostic_scores=diagnostic_scores,
            comparison_options=comparison_options,
            class_ids=class_ids[start : start + effective_page_size],
        )
        for start in range(0, len(class_ids), effective_page_size)
    ]


def create_class_preview_figure(
    class_averages,
    projections,
    cameras,
    orientations,
    render_paths,
    *,
    diagnostic_scores=None,
    comparison_options=None,
    class_ids,
):
    """Create one three-column Class Result figure for the requested classes."""
    from matplotlib.figure import Figure
    from PIL import Image

    projection_rows = {
        class_id: row for row, class_id in enumerate(sorted(orientations))
    }
    comparison_options = comparison_options or ComparisonRenderOptions()
    figure = Figure(
        figsize=(9, 3 * len(class_ids)),
        dpi=comparison_options.dpi,
        constrained_layout=True,
    )

    for row, class_id in enumerate(class_ids):
        orientation = orientations[class_id]
        class_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 1)
        class_axis.imshow(
            np.flipud(class_averages[class_id].image),
            cmap="gray",
            interpolation="hanning",
        )
        class_axis.set_title(
            f"Class {class_id + 1} | n={orientation.particle_count}\n"
            f"spread={orientation.angular_spread_degrees:.1f}°"
        )
        class_axis.axis("off")

        projection_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 2)
        projection_axis.imshow(
            np.flipud(projections[projection_rows[class_id]]),
            cmap="gray",
            interpolation="hanning",
        )
        projection_title = (
            f"Matched | search raw={cameras[class_id].match_score:.3f}"
        )
        if diagnostic_scores is not None:
            diagnostic = diagnostic_scores[class_id]
            if diagnostic.valid:
                low_resolution = diagnostic.metadata[
                    "band_low_resolution_A_effective"
                ]
                high_resolution = diagnostic.metadata[
                    "band_high_resolution_A_effective"
                ]
                projection_title += (
                    f"\nband ({low_resolution:g}–{high_resolution:g} Å)="
                    f"{diagnostic.score:.3f}"
                )
            else:
                projection_title += "\nband=invalid"
        projection_axis.set_title(projection_title)
        projection_axis.axis("off")

        camera_view_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 3)
        with Image.open(render_paths[class_id]) as camera_view_image:
            camera_view_axis.imshow(camera_view_image.convert("RGB"))
        camera_view_axis.set_title("Camera View Render")
        camera_view_axis.axis("off")

    return figure


def _model_matrix_command(rotation_matrix):
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    values = []
    for row in rotation_matrix:
        values.extend([*row, 0.0])
    encoded = ",".join(f"{value:.12g}" for value in values)
    return f"view matrix models #1,{encoded}"
