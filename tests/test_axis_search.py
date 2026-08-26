import numpy as np
import pytest
from scipy.ndimage import rotate, shift
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.axis_projection import project_axis_reference
from cryosparc_2d_projection.axis_registry import get_axis_family
from cryosparc_2d_projection.axis_search import (
    AxisClassScoreError,
    AxisProximityConfig,
    AxisSearchConfig,
    rank_axis_family,
    rank_axis_families,
    refine_axis_candidates,
)
from cryosparc_2d_projection.projection import project_volume_at_rotation


def _asymmetric_volume(size=32):
    coordinates = np.linspace(-1.0, 1.0, size)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    volume = np.exp(
        -(
            (x + 0.28) ** 2 / 0.08
            + (y - 0.13) ** 2 / 0.04
            + (z + 0.21) ** 2 / 0.12
        )
    )
    volume += 0.6 * np.exp(
        -(
            (x - 0.34) ** 2 / 0.03
            + (y + 0.26) ** 2 / 0.06
            + (z - 0.18) ** 2 / 0.05
        )
    )
    return volume.astype(np.float32)


def test_one_family_ranking_uses_axis_score_and_breaks_ties_by_class_number():
    volume = _asymmetric_volume()
    reference = project_axis_reference(volume, "2fold").projection

    result = rank_axis_family(
        {7: reference.copy(), 3: reference.copy()},
        volume,
        family="2fold",
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=2),
    )

    assert result.family.name == "2fold"
    assert [candidate.class_number for candidate in result.candidates] == [3, 7]
    assert result.candidates[0].exact_score > 0.999
    assert result.candidates[0].score_metadata["score_role"] == "axis_class_ranking"
    assert result.candidates[0].raw_correlation > 0.999
    assert np.array_equal(result.candidates[0].raw_class, reference)
    assert result.candidates[0].aligned_class.shape == reference.shape


def test_axis_search_defaults_match_the_approved_contract():
    config = AxisSearchConfig()

    assert config.low_resolution_A == 80.0
    assert config.high_resolution_A == 15.0
    assert config.mask_radius_fraction == 0.45
    assert config.mask_edge_fraction == 0.05
    assert config.roll_coarse_step_degrees == 5.0
    assert config.roll_refine_step_degrees == 0.5
    assert config.shift_bound_fraction == 0.10
    assert config.top_n == 5


def test_roll_search_respects_family_period_and_shift_bound():
    volume = _asymmetric_volume()
    reference = project_axis_reference(volume, "5fold").projection
    transformed = shift(
        rotate(reference, 17.0, reshape=False, order=1),
        shift=(0.0, 2.0),
        order=1,
    )

    result = rank_axis_family(
        {1: transformed},
        volume,
        family="5fold",
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(
            roll_coarse_step_degrees=5.0,
            roll_refine_step_degrees=0.5,
            shift_bound_fraction=0.10,
            top_n=1,
        ),
    )

    candidate = result.candidates[0]
    assert 0.0 <= candidate.roll_degrees < 72.0
    assert candidate.roll_degrees == 17.0
    assert abs(candidate.shift_xy_pixels[0]) <= 3.0
    assert abs(candidate.shift_xy_pixels[1]) <= 3.0


def test_invalid_axis_class_score_fails_without_raw_correlation_fallback():
    volume = np.ones((16, 16, 16), dtype=np.float32)

    with pytest.raises(AxisClassScoreError, match="did not fall back"):
        rank_axis_family(
            {1: np.ones((16, 16), dtype=np.float32)},
            volume,
            family="2fold",
            class_pixel_size_A=1.0,
            map_pixel_size_A=1.0,
        )


def test_all_i_families_are_ordered_and_requested_subset_is_honored():
    volume = _asymmetric_volume()
    classes = {
        1: project_axis_reference(volume, "2fold").projection,
        2: project_axis_reference(volume, "3fold").projection,
    }

    complete = rank_axis_families(
        classes,
        volume,
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )
    subset = rank_axis_families(
        classes,
        volume,
        families=("5fold", "2fold"),
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )

    assert list(complete.families) == ["2fold", "3fold", "5fold"]
    assert list(subset.families) == ["2fold", "5fold"]
    assert [row.family_name for row in subset.rows] == ["2fold", "5fold"]


def test_cross_family_repeat_is_retained_and_marked_duplicate():
    volume = _asymmetric_volume()
    one_class = {4: project_axis_reference(volume, "2fold").projection}

    result = rank_axis_families(
        one_class,
        volume,
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )

    assert [row.class_number for row in result.rows] == [4, 4, 4]
    assert all(row.duplicate for row in result.rows)


def test_family_reports_first_second_score_and_margin_without_confidence_label():
    volume = _asymmetric_volume()
    reference = project_axis_reference(volume, "3fold").projection
    weaker = reference + 0.15 * np.sin(np.arange(reference.size)).reshape(reference.shape)

    result = rank_axis_families(
        {1: reference, 2: weaker.astype(np.float32)},
        volume,
        families=("3fold",),
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )

    family = result.families["3fold"]
    assert family.first_score == family.candidates[0].exact_score
    assert len(family.candidates) == 1
    assert family.second_score is not None
    assert family.score_margin == family.first_score - family.second_score
    assert not hasattr(family, "confidence")


def test_mirror_is_diagnostic_only_and_never_flips_source_class():
    volume = _asymmetric_volume()
    reference = project_axis_reference(volume, "2fold").projection
    mirrored = np.fliplr(reference)

    result = rank_axis_families(
        {1: mirrored},
        volume,
        families=("2fold",),
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1, mirror_warning_margin=0.05),
    )

    candidate = result.rows[0]
    assert candidate.mirrored_score > candidate.exact_score + 0.05
    assert "mirrored" in " ".join(candidate.warnings).lower()
    assert np.array_equal(candidate.raw_class, mirrored)


def test_proximity_refines_only_selected_rows_without_mutating_exact_ranking():
    volume = _asymmetric_volume()
    family = get_axis_family("I", "2fold")
    near_camera = (
        Rotation.from_rotvec(np.deg2rad([0.0, 6.0, 0.0])).as_matrix()
        @ family.canonical_camera_matrix
    )
    near_class = project_volume_at_rotation(volume, near_camera)
    exact = rank_axis_families(
        {1: near_class},
        volume,
        families=("2fold",),
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )
    exact_bytes = exact.rows[0].exact_reference.tobytes()

    refined = refine_axis_candidates(
        exact,
        volume,
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisProximityConfig(
            cone_degrees=9.0,
            coarse_step_degrees=3.0,
            refine_step_degrees=1.0,
        ),
    )

    assert refined.exact_result is exact
    assert exact.rows[0].exact_reference.tobytes() == exact_bytes
    assert len(refined.rows) == 1
    row = refined.rows[0]
    assert row.refined_score >= row.exact_candidate.exact_score
    assert row.angular_distance_degrees == pytest.approx(6.0, abs=1.5)
    assert row.near_axis_projection.shape == near_class.shape
    assert np.allclose(row.near_axis_projection_display, np.flipud(row.near_axis_projection))


def test_axis_proximity_defaults_match_the_approved_contract():
    config = AxisProximityConfig()

    assert config.cone_degrees == 15.0
    assert config.coarse_step_degrees == 3.0
    assert config.refine_step_degrees == 0.5


def test_cone_boundary_result_is_retained_with_warning_without_expansion():
    volume = _asymmetric_volume()
    family = get_axis_family("I", "5fold")
    outside_camera = (
        Rotation.from_rotvec(np.deg2rad([0.0, 9.0, 0.0])).as_matrix()
        @ family.canonical_camera_matrix
    )
    exact = rank_axis_families(
        {1: project_volume_at_rotation(volume, outside_camera)},
        volume,
        families=("5fold",),
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )

    row = refine_axis_candidates(
        exact,
        volume,
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisProximityConfig(
            cone_degrees=6.0,
            coarse_step_degrees=3.0,
            refine_step_degrees=1.0,
        ),
    ).rows[0]

    assert row.cone_boundary is True
    assert row.angular_distance_degrees <= 6.0
    assert "not expanded" in " ".join(row.warnings)


def test_aligned_class_and_near_projection_share_canonical_presentation_coordinates():
    volume = _asymmetric_volume()
    family = get_axis_family("I", "3fold")
    near_camera = (
        Rotation.from_rotvec(np.deg2rad([0.0, 6.0, 0.0])).as_matrix()
        @ family.canonical_camera_matrix
    )
    canonical_near = project_volume_at_rotation(volume, near_camera)
    source_class = shift(
        rotate(canonical_near, 12.0, reshape=False, order=1),
        shift=(1.0, -1.0),
        order=1,
    )
    exact = rank_axis_families(
        {1: source_class},
        volume,
        families=("3fold",),
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=1),
    )

    refined = refine_axis_candidates(
        exact,
        volume,
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        config=AxisProximityConfig(
            cone_degrees=9.0,
            coarse_step_degrees=3.0,
            refine_step_degrees=1.0,
        ),
    ).rows[0]

    correlation = np.corrcoef(
        exact.rows[0].aligned_class.ravel(),
        refined.near_axis_projection_display.ravel(),
    )[0, 1]
    assert correlation > 0.95
