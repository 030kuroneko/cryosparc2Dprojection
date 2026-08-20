from types import SimpleNamespace

import numpy as np

from cryosparc_2d_projection.viewer import write_chimerax_bundle


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
