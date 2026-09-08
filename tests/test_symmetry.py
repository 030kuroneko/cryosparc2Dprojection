"""Convention contracts using independently specified axes and vertex sets.

Sources and live-validation limits: docs/research/cryosparc-point-groups.md.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.symmetry import symmetry_operators


@pytest.mark.parametrize("name,count", [
    ("C1", 1), ("C2", 2), ("C3", 3), ("C11", 11),
    ("D1", 2), ("D2", 4), ("D3", 6), ("D7", 14),
    ("T", 12), ("O", 24), ("I", 60),
])
def test_point_groups_have_distinct_proper_rotations_and_are_closed(name, count):
    operators = symmetry_operators(name)
    assert operators.shape == (count, 3, 3)
    np.testing.assert_allclose(np.linalg.det(operators), 1., atol=1e-12)
    np.testing.assert_allclose(
        operators @ operators.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), operators.shape), atol=1e-12,
    )
    distances = np.linalg.norm(operators[:, None] - operators[None, :], axis=(2, 3))
    assert np.count_nonzero(distances < 1e-10) == count
    for operator in operators:
        products = operator @ operators
        distances = np.linalg.norm(products[:, None] - operators[None, :], axis=(2, 3))
        assert np.all(distances.min(axis=1) < 1e-10)


@pytest.mark.parametrize("name,vertices", [
    ("T", [[0, 0, 1], [0, 2*np.sqrt(2)/3, -1/3],
           [np.sqrt(2/3), -np.sqrt(2)/3, -1/3],
           [-np.sqrt(2/3), -np.sqrt(2)/3, -1/3]]),
    ("O", [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]),
])
def test_polyhedral_operators_preserve_declared_vertices(name, vertices):
    vertices = np.array(vertices)
    for operator in symmetry_operators(name):
        rotated = vertices @ operator.T
        distances = np.linalg.norm(rotated[:, None] - vertices[None, :], axis=2)
        assert np.all(distances.min(axis=1) < 1e-12)


def test_odd_dihedral_group_does_not_use_relion_x_dyad():
    operators = symmetry_operators("D7")
    x_dyad = np.diag([1., -1., -1.])
    assert np.min(np.linalg.norm(operators - x_dyad, axis=(1, 2))) > .1


@pytest.mark.parametrize("name,n", [("C3", 3), ("C11", 11), ("D3", 3), ("D7", 7)])
def test_cyclic_and_dihedral_principal_axis_is_z(name, n):
    generator = Rotation.from_euler("z", 360/n, degrees=True).as_matrix()
    assert np.min(np.linalg.norm(symmetry_operators(name) - generator, axis=(1, 2))) < 1e-12
