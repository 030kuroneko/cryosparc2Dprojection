from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.class_poses import analyze_class_orientations
from cryosparc_2d_projection.projection import project_volume


TARGET_CRYOSPARC_VERSION = "5.0.6"


@dataclass(frozen=True)
class SourceOutput:
    job_uid: str
    output_name: str


def run_external_orientation_job(
    project,
    workspace_uid,
    select_2d_source,
    refinement_source,
    volume_source,
    symmetry="C1",
):
    """Create and run the CryoSPARC External Job for class orientation analysis."""
    job = project.create_external_job(
        workspace_uid,
        title=f"2D Class Orientation (CryoSPARC {TARGET_CRYOSPARC_VERSION})",
    )
    _add_and_connect_input(
        job,
        name="select_2d_particles",
        type="particle",
        slots=["alignments2D"],
        source=select_2d_source,
        title="Select 2D particles",
    )
    _add_and_connect_input(
        job,
        name="refinement_particles",
        type="particle",
        slots=["alignments3D"],
        source=refinement_source,
        title="NU or Local Refinement particles",
    )
    _add_and_connect_input(
        job,
        name="refinement_volume",
        type="volume",
        slots=["blob"],
        source=volume_source,
        title="NU or Local Refinement volume",
    )

    with job.run():
        orientations = analyze_class_orientations(
            job.load_input("select_2d_particles"),
            job.load_input("refinement_particles"),
            symmetry=symmetry,
        )
        artifact = {
            "cryosparc_version": TARGET_CRYOSPARC_VERSION,
            "symmetry": symmetry,
            "classes": [
                {
                    "class_id": class_id,
                    "class_number": class_id + 1,
                    "particle_count": orientation.particle_count,
                    "view_direction": orientation.view_direction.tolist(),
                    "angular_spread_degrees": orientation.angular_spread_degrees,
                }
                for class_id, orientation in sorted(orientations.items())
            ],
        }
        job_directory = Path(_directory_of(job))
        output_path = job_directory / "class_orientations.json"
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")

        volume_input = job.load_input("refinement_volume")
        volume_path = _resolve_project_path(project, volume_input["blob/path"][0])
        _, volume_data = mrc.read(volume_path)
        pixel_size = float(volume_input["blob/psize_A"][0])
        projections = np.asarray(
            [
                project_volume(volume_data, orientations[class_id].view_direction)
                for class_id in sorted(orientations)
            ],
            dtype=np.float32,
        )
        mrc.write(job_directory / "class_projections.mrcs", projections, pixel_size)
        preview = _projection_preview(projections, orientations)
        job.log_plot(preview, "Class projection preview", formats=["png"])
        job.log(
            f"Analyzed {len(orientations)} 2D classes using overlapping particle UIDs."
        )

    return job


def _add_and_connect_input(job, *, name, type, slots, source, title):
    job.add_input(
        type=type,
        name=name,
        min=1,
        max=1,
        slots=slots,
        title=title,
    )
    job.connect(name, source.job_uid, source.output_name)


def _directory_of(resource):
    directory = resource.dir
    return directory() if callable(directory) else directory


def _resolve_project_path(project, path):
    if isinstance(path, bytes):
        path = path.decode()
    path = Path(str(path).removeprefix(">"))
    if path.is_absolute():
        return path
    return Path(_directory_of(project)) / path


def _projection_preview(projections, orientations, *, limit=25):
    from matplotlib.figure import Figure

    class_ids = sorted(orientations)[:limit]
    columns = min(5, len(class_ids))
    rows = (len(class_ids) + columns - 1) // columns
    figure = Figure(figsize=(3 * columns, 3 * rows), constrained_layout=True)

    for plot_index, class_id in enumerate(class_ids, start=1):
        axis = figure.add_subplot(rows, columns, plot_index)
        axis.imshow(projections[plot_index - 1], cmap="gray")
        orientation = orientations[class_id]
        axis.set_title(
            f"Class {class_id + 1} | n={orientation.particle_count}\n"
            f"spread={orientation.angular_spread_degrees:.1f}°"
        )
        axis.axis("off")

    return figure
