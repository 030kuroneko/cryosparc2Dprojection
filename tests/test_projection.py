import numpy as np

from cryosparc_2d_projection.projection import project_volume


def test_identity_view_projects_along_mrc_z_axis():
    volume = np.arange(27, dtype=float).reshape(3, 3, 3)

    projection = project_volume(volume, [0.0, 0.0, 1.0])

    assert np.allclose(projection, volume.sum(axis=0))


def test_x_axis_view_integrates_density_along_x():
    volume = np.broadcast_to(np.arange(3, dtype=float), (3, 3, 3))

    projection = project_volume(volume, [1.0, 0.0, 0.0])

    assert np.allclose(projection, np.full((3, 3), 3.0))
