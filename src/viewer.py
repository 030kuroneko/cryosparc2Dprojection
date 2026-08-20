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


def _model_matrix_command(rotation_matrix):
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    values = []
    for row in rotation_matrix:
        values.extend([*row, 0.0])
    encoded = ",".join(f"{value:.12g}" for value in values)
    return f"view matrix models #1,{encoded}"
