from pathlib import Path

import numpy as np


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
    page_size=10,
):
    """Create three-column Class Result pages with bounded row counts."""
    class_ids = sorted(orientations)
    return [
        create_class_preview_figure(
            class_averages,
            projections,
            cameras,
            orientations,
            render_paths,
            class_ids=class_ids[start : start + page_size],
        )
        for start in range(0, len(class_ids), page_size)
    ]


def create_class_preview_figure(
    class_averages,
    projections,
    cameras,
    orientations,
    render_paths,
    *,
    class_ids,
):
    """Create one three-column Class Result figure for the requested classes."""
    from matplotlib.figure import Figure
    from PIL import Image

    projection_rows = {
        class_id: row for row, class_id in enumerate(sorted(orientations))
    }
    figure = Figure(figsize=(9, 3 * len(class_ids)), constrained_layout=True)

    for row, class_id in enumerate(class_ids):
        orientation = orientations[class_id]
        class_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 1)
        class_axis.imshow(class_averages[class_id].image, cmap="gray")
        class_axis.set_title(
            f"Class {class_id + 1} | n={orientation.particle_count}\n"
            f"spread={orientation.angular_spread_degrees:.1f}°"
        )
        class_axis.axis("off")

        projection_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 2)
        projection_axis.imshow(projections[projection_rows[class_id]], cmap="gray")
        projection_axis.set_title(
            f"Matched | score={cameras[class_id].match_score:.3f}"
        )
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
