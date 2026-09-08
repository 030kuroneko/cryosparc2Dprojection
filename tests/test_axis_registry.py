import numpy as np
import pytest

from cryosparc_2d_projection.axis_registry import (
    AxisFamilyRegistry,
    axis_family_records,
    get_axis_family,
)
from cryosparc_2d_projection.projection import project_volume_at_rotation


def test_icosahedral_registry_exposes_canonical_axis_records():
    records = axis_family_records("I")

    assert [record.name for record in records] == ["2fold", "3fold", "5fold"]
    assert [record.undirected_axis_count for record in records] == [15, 10, 6]
    assert [record.directed_axis_count for record in records] == [30, 20, 12]
    assert [record.roll_period_degrees for record in records] == [180.0, 120.0, 72.0]
    assert np.allclose(records[0].representative_view_direction, [0.0, 1.0, 0.0])
    assert np.allclose(
        records[1].representative_view_direction,
        [-0.934172358962716, 0.356822089773090, 0.0],
    )
    assert np.array_equal(
        records[1].representative_view_direction,
        np.array([-0.934172358962716, 0.356822089773090, 0.0]),
    )
    assert np.allclose(
        records[2].canonical_camera_matrix,
        [
            [0.525731112119134, 0.0, 0.850650808352040],
            [0.0, 1.0, 0.0],
            [-0.850650808352040, 0.0, 0.525731112119134],
        ],
    )


def test_registry_lookup_is_case_insensitive_and_rejects_unverified_families():
    record = get_axis_family("i", " 3FOLD ")

    assert record.name == "3fold"
    assert AxisFamilyRegistry.for_symmetry("I").lookup("5fold").name == "5fold"
    with pytest.raises(ValueError, match="2fold, 3fold, 5fold"):
        get_axis_family("I", "principal")
    for alias in ("2", "2-fold", "2_fold", "3", "5-fold"):
        with pytest.raises(ValueError, match="2fold, 3fold, 5fold"):
            get_axis_family("I", alias)
    with pytest.raises(ValueError, match="C1 has no nontrivial symmetry axes"):
        axis_family_records("C1")
    with pytest.raises(ValueError, match="Supported symmetry:"):
        axis_family_records("I1")
    with pytest.raises(ValueError, match="Supported symmetry:"):
        axis_family_records("I2")


def test_record_canonical_presentation_is_stable_and_read_only():
    for record in axis_family_records("I"):
        camera = record.canonical_camera_matrix
        assert np.allclose(camera @ camera.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(camera) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(camera[2], record.representative_view_direction)

        presentation = record.canonical_presentation()

        assert presentation.rule == "nearest_cross_family_axis_horizontal"
        assert presentation.roll_degrees == 0.0
        assert presentation.reference_family_name != record.name
        assert np.dot(presentation.projected_axis, camera[1]) == pytest.approx(
            0.0, abs=1e-8
        )
        assert np.allclose(presentation.camera_matrix, camera)
        with pytest.raises(ValueError):
            record.canonical_camera_matrix[0, 0] = 1.0


def test_each_family_roll_period_is_observable_in_reference_projection():
    from scipy.spatial.transform import Rotation

    from cryosparc_2d_projection.projection import rotate_volume_at_rotation

    size = 33
    center = size // 2
    marker = np.zeros((size, size, size), dtype=np.float64)
    marker[center - 2, center + 1, center + 3] = 1.0
    marker[center, center - 3, center - 2] = 0.7
    group = Rotation.create_group("I").as_matrix()
    # Symmetrize an asymmetric marker independently of the registry records.
    # The interpolation tolerance reflects the existing projection boundary's
    # linear voxel resampling for non-grid-aligned I axes.
    volume = np.mean(
        [rotate_volume_at_rotation(marker, operator) for operator in group],
        axis=0,
    )

    for record in axis_family_records("I"):
        base = project_volume_at_rotation(volume, record.camera_for_roll(0.0))
        periodic = project_volume_at_rotation(
            volume,
            record.camera_for_roll(record.roll_period_degrees),
        )
        assert not np.allclose(
            record.camera_for_roll(0.0),
            record.camera_for_roll(record.roll_period_degrees),
            atol=1e-12,
        )
        assert np.allclose(periodic, base, atol=2e-2)


@pytest.mark.parametrize('symmetry,names,counts,periods', [
    ('C3', ['3fold', '3fold-2'], [1, 1], [120, 120]),
    ('D1', ['2fold', '2fold-2'], [1, 1], [180, 180]),
    ('D2', ['2fold', '2fold-2', '2fold-3'], [2, 2, 2], [180, 180, 180]),
    ('D7', ['2fold', '2fold-2', '7fold'], [7, 7, 2], [180, 180, 360/7]),
    ('T', ['2fold', '3fold', '3fold-2'], [6, 4, 4], [180, 120, 120]),
    ('O', ['2fold', '3fold', '4fold'], [12, 8, 6], [180, 120, 90]),
])
def test_registry_covers_distinct_directed_axis_orbits(symmetry, names, counts, periods):
    records = axis_family_records(symmetry)
    assert [r.name for r in records] == names
    assert [r.directed_axis_count for r in records] == counts
    assert [r.roll_period_degrees for r in records] == pytest.approx(periods)
    for record in records:
        assert record.symmetry == symmetry
        camera = record.canonical_camera_matrix
        np.testing.assert_allclose(camera @ camera.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(camera) == pytest.approx(1)
        np.testing.assert_allclose(camera[2], record.representative_view_direction)
        assert record.canonical_presentation().rule == 'cartesian_reference_axis_horizontal'


@pytest.mark.parametrize('symmetry', ['C2', 'C11', 'D1', 'D2', 'D3', 'D4', 'D7', 'T', 'O'])
def test_axis_families_cover_all_rotation_axes_without_merging_distinct_orbits(symmetry):
    from scipy.spatial.transform import Rotation
    from cryosparc_2d_projection.symmetry import symmetry_operators
    group = symmetry_operators(symmetry)
    records = axis_family_records(symmetry)
    orbits = [r.representative_view_direction @ group for r in records]
    for index, record in enumerate(records):
        direction = record.representative_view_direction
        for other in orbits[index+1:]:
            assert np.min(np.linalg.norm(other - direction, axis=1)) > 1e-6
        # An in-plane period is an actual map symmetry, not merely an axis label.
        camera = record.canonical_camera_matrix
        roll = Rotation.from_euler('z', record.roll_period_degrees, degrees=True).as_matrix()
        operator = camera.T @ roll @ camera
        assert np.min(np.linalg.norm(group - operator, axis=(1, 2))) < 1e-8
    covered = np.concatenate(orbits)
    for vector in Rotation.from_matrix(group).as_rotvec():
        if np.linalg.norm(vector) < 1e-8:
            continue
        direction = vector / np.linalg.norm(vector)
        for pole in (direction, -direction):
            assert np.min(np.linalg.norm(covered - pole, axis=1)) < 1e-8
