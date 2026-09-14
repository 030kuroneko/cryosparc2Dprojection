"""Selection artifacts observed at the orientation workflow output boundary."""
import json

import numpy as np

from cryosparc_2d_projection.external_job import SourceOutput, run_external_orientation_job
from cryosparc_2d_projection.surface_render import ClassRenderOptions
from tests.test_external_job import _native_grid_external_job


def test_web_results_keep_original_particle_count_for_invalid_pose_fallback(tmp_path, monkeypatch):
    project, job = _native_grid_external_job(tmp_path, class_size=9)
    project.uid = 'P1'
    job.datasets['select_2d_particles']['alignments2D/pose'][:] = np.nan
    directory = tmp_path / 'selection'
    monkeypatch.setenv('CRYOSPARC2D_SELECTION_DIR', str(directory))
    run_external_orientation_job(
        project, 'W1', SourceOutput('J1', 'particles'), SourceOutput('J1', 'templates'),
        SourceOutput('J2', 'particles'), SourceOutput('J2', 'volume'),
        render_options=ClassRenderOptions(image_size=64, grid_size=8),
    )
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['schema_version'] == 1
    assert manifest['project_uid'] == 'P1'
    assert manifest['workspace_uid'] == 'W1'
    assert manifest['particles_source'] == {'job_uid': 'J1', 'output': 'particles'}
    assert manifest['templates_source'] == {'job_uid': 'J1', 'output': 'templates'}
    row, = manifest['classes']
    assert row['class_number'] == 1
    assert row['particle_count'] == 1
    assert row['orientation_method'] == 'image_global_search'
    assert row['confidence'] in ('low', 'high')
    assert isinstance(row['score'], float)
    assert row['image'] == 'class_1.png'
    assert (directory / row['image']).read_bytes().startswith(b'\x89PNG')
    assert str(tmp_path) not in json.dumps(manifest)
