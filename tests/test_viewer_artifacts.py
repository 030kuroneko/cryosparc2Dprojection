from types import SimpleNamespace

import numpy as np
from PIL import Image

from cryosparc_2d_projection.viewer import (
    create_class_preview_pages,
    write_chimerax_bundle,
)


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


def test_class_preview_pages_contain_at_most_ten_four_column_rows(tmp_path):
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
        oblique_path = tmp_path / f"class_{class_id + 1:03d}_oblique.png"
        Image.new("RGB", (8, 8), "white").save(exact_path)
        Image.new("RGB", (8, 8), "gray").save(oblique_path)
        render_paths[class_id] = SimpleNamespace(
            camera_view_path=exact_path,
            oblique_inspection_path=oblique_path,
        )

    pages = create_class_preview_pages(
        class_averages,
        projections,
        cameras,
        orientations,
        render_paths,
        page_size=10,
    )

    assert len(pages) == 2
    assert len(pages[0].axes) == 40
    assert len(pages[1].axes) == 4
    assert pages[0].axes[2].get_title() == "Camera View Render"
    assert pages[0].axes[3].get_title() == "Oblique Inspection Render"
