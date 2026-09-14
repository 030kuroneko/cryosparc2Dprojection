"""Durable, credential-free artifacts for completed web Class Orientation jobs."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def particle_membership_digest(uids, class_ids):
    """Identify original particle membership independent of dataset row order."""
    pairs = sorted((int(uid), int(class_id)) for uid, class_id in zip(uids, class_ids, strict=True))
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()


def write_selection_artifacts(directory, *, project_uid, workspace_uid, source_job_uid,
                              particles_source, templates_source, original_particles,
                              cameras, comparison_paths):
    """Publish the manifest last, once every class comparison has been copied."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    counts = {}
    for class_id in original_particles.class_ids:
        counts[int(class_id)] = counts.get(int(class_id), 0) + 1
    rows = []
    for class_id, camera in sorted(cameras.items()):
        number = int(class_id) + 1
        filename = f'class_{number}.png'
        shutil.copyfile(comparison_paths[class_id], directory / filename)
        score = float(camera.match_score)
        rows.append(dict(class_number=number, particle_count=counts.get(class_id, 0),
                         score=score if math.isfinite(score) else None,
                         orientation_method=camera.orientation_method,
                         confidence=camera.match_confidence, image=filename))
    manifest = dict(schema_version=1, project_uid=project_uid,
                    workspace_uid=workspace_uid, source_job_uid=source_job_uid,
                    particles_source=dict(job_uid=particles_source.job_uid, output=particles_source.output_name),
                    templates_source=dict(job_uid=templates_source.job_uid, output=templates_source.output_name),
                    particles_membership_digest=particle_membership_digest(original_particles.uids, original_particles.class_ids),
                    classes=rows)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix='.manifest-', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(manifest, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / 'manifest.json')
    finally:
        Path(temporary).unlink(missing_ok=True)
