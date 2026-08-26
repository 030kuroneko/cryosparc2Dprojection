"""Image-only Symmetry-Axis Class Search domain operations."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage import shift as shift_image
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.axis_projection import project_axis_reference
from cryosparc_2d_projection.axis_registry import (
    AxisFamilyRecord,
    axis_family_records,
    get_axis_family,
)
from cryosparc_2d_projection.matching_grid import prepare_native_matching_grid
from cryosparc_2d_projection.projection import project_volume_at_rotation
from cryosparc_2d_projection.scoring import (
    BandLimitedScoreConfig,
    compute_diagnostic_band_limited_score,
)


class AxisClassScoreError(RuntimeError):
    """An Axis Class Score could not be calculated safely."""


@dataclass(frozen=True)
class AxisSearchConfig:
    low_resolution_A: float = 80.0
    high_resolution_A: float = 15.0
    mask_radius_fraction: float = 0.45
    mask_edge_fraction: float = 0.05
    roll_coarse_step_degrees: float = 5.0
    roll_refine_step_degrees: float = 0.5
    shift_bound_fraction: float = 0.10
    top_n: int = 5
    mirror_warning_margin: float = 0.05

    def __post_init__(self):
        BandLimitedScoreConfig(
            low_resolution_A=self.low_resolution_A,
            high_resolution_A=self.high_resolution_A,
            mask_radius_fraction=self.mask_radius_fraction,
            mask_edge_fraction=self.mask_edge_fraction,
        )
        for name, value in (
            ("roll coarse step", self.roll_coarse_step_degrees),
            ("roll refinement step", self.roll_refine_step_degrees),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.shift_bound_fraction)
            or not 0 <= self.shift_bound_fraction <= 0.5
        ):
            raise ValueError("shift bound fraction must be between zero and 0.5")
        if type(self.top_n) is not int or self.top_n <= 0:
            raise ValueError("top N must be a positive integer")
        if not np.isfinite(self.mirror_warning_margin) or self.mirror_warning_margin < 0:
            raise ValueError("mirror warning margin must be finite and non-negative")

    def score_config(self):
        return BandLimitedScoreConfig(
            low_resolution_A=self.low_resolution_A,
            high_resolution_A=self.high_resolution_A,
            mask_radius_fraction=self.mask_radius_fraction,
            mask_edge_fraction=self.mask_edge_fraction,
        )


@dataclass(frozen=True)
class AxisCandidate:
    family_name: str
    class_number: int
    raw_class: np.ndarray
    aligned_class: np.ndarray
    exact_reference: np.ndarray
    exact_reference_display: np.ndarray
    exact_rotation_matrix: np.ndarray
    exact_score: float
    raw_correlation: float
    roll_degrees: float
    shift_xy_pixels: tuple[float, float]
    score_metadata: dict[str, object] = field(default_factory=dict)
    mirrored_score: float | None = None
    warnings: tuple[str, ...] = ()
    duplicate: bool = False


@dataclass(frozen=True)
class AxisFamilyRanking:
    family: AxisFamilyRecord
    candidates: tuple[AxisCandidate, ...]
    first_score: float | None = None
    second_score: float | None = None
    score_margin: float | None = None


@dataclass(frozen=True)
class AxisSearchResult:
    families: dict[str, AxisFamilyRanking]
    rows: tuple[AxisCandidate, ...]


@dataclass(frozen=True)
class AxisProximityConfig:
    cone_degrees: float = 15.0
    coarse_step_degrees: float = 3.0
    refine_step_degrees: float = 0.5

    def __post_init__(self):
        for name, value in (
            ("axis cone", self.cone_degrees),
            ("coarse tilt step", self.coarse_step_degrees),
            ("refined tilt step", self.refine_step_degrees),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cone_degrees > 90:
            raise ValueError("axis cone must not exceed 90 degrees")


@dataclass(frozen=True)
class AxisRefinedCandidate:
    exact_candidate: AxisCandidate
    refined_score: float
    angular_distance_degrees: float
    near_axis_rotation_matrix: np.ndarray
    near_axis_projection: np.ndarray
    near_axis_projection_display: np.ndarray
    cone_boundary: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AxisRefinementResult:
    exact_result: AxisSearchResult
    rows: tuple[AxisRefinedCandidate, ...]


def refine_axis_candidates(
    exact_result,
    matching_map,
    *,
    class_pixel_size_A,
    map_pixel_size_A,
    config=None,
):
    """Refine only the selected Exact-Axis rows inside a bounded cone."""

    if not isinstance(exact_result, AxisSearchResult):
        raise TypeError("exact_result must be an AxisSearchResult")
    config = config or AxisProximityConfig()
    if not isinstance(config, AxisProximityConfig):
        raise TypeError("config must be an AxisProximityConfig")
    if not exact_result.rows:
        return AxisRefinementResult(exact_result=exact_result, rows=())
    grid = prepare_native_matching_grid(
        exact_result.rows[0].raw_class,
        matching_map,
        class_pixel_size=class_pixel_size_A,
        volume_pixel_size=map_pixel_size_A,
    )
    rows = tuple(
        _refine_candidate(candidate, grid.volume, grid.pixel_size, config)
        for candidate in exact_result.rows
    )
    return AxisRefinementResult(exact_result=exact_result, rows=rows)


def rank_axis_families(
    class_averages,
    matching_map,
    *,
    families=None,
    class_pixel_size_A,
    map_pixel_size_A,
    config=None,
):
    """Rank the complete supported I registry or a requested family subset."""

    config = config or AxisSearchConfig()
    requested = (
        {record.name for record in axis_family_records("I")}
        if families is None
        else {get_axis_family("I", family).name for family in families}
    )
    ordered_records = tuple(
        record for record in axis_family_records("I") if record.name in requested
    )
    if not ordered_records:
        raise ValueError("at least one Axis Family is required")
    rankings = {
        record.name: rank_axis_family(
            class_averages,
            matching_map,
            family=record,
            class_pixel_size_A=class_pixel_size_A,
            map_pixel_size_A=map_pixel_size_A,
            config=config,
        )
        for record in ordered_records
    }
    selected_counts = {}
    for ranking in rankings.values():
        for candidate in ranking.candidates:
            selected_counts[candidate.class_number] = (
                selected_counts.get(candidate.class_number, 0) + 1
            )
    updated_rankings = {}
    rows = []
    for name, ranking in rankings.items():
        candidates = tuple(
            replace(
                candidate,
                duplicate=selected_counts[candidate.class_number] > 1,
            )
            for candidate in ranking.candidates
        )
        updated_rankings[name] = replace(ranking, candidates=candidates)
        rows.extend(candidates)
    return AxisSearchResult(families=updated_rankings, rows=tuple(rows))


def rank_axis_family(
    class_averages,
    matching_map,
    *,
    family,
    class_pixel_size_A,
    map_pixel_size_A,
    config=None,
):
    """Rank one Axis Family from class-average images and a Matching Map."""

    config = config or AxisSearchConfig()
    if not isinstance(config, AxisSearchConfig):
        raise TypeError("config must be an AxisSearchConfig")
    family_record = (
        family if isinstance(family, AxisFamilyRecord) else get_axis_family("I", family)
    )
    classes = _validate_class_averages(class_averages)
    first_class = next(iter(classes.values()))
    grid = prepare_native_matching_grid(
        first_class,
        matching_map,
        class_pixel_size=class_pixel_size_A,
        volume_pixel_size=map_pixel_size_A,
    )
    reference = project_axis_reference(
        grid.volume,
        family_record,
        pixel_size_A=grid.pixel_size,
    )
    candidates = [
        _match_class(
            class_number,
            image,
            reference,
            family_record,
            pixel_size_A=grid.pixel_size,
            config=config,
        )
        for class_number, image in classes.items()
    ]
    candidates.sort(key=lambda candidate: (-candidate.exact_score, candidate.class_number))
    first_score = candidates[0].exact_score if candidates else None
    second_score = candidates[1].exact_score if len(candidates) > 1 else None
    selected = tuple(candidates[: config.top_n])
    return AxisFamilyRanking(
        family=family_record,
        candidates=selected,
        first_score=first_score,
        second_score=second_score,
        score_margin=(
            None if second_score is None else float(first_score - second_score)
        ),
    )


def _validate_class_averages(class_averages):
    if not class_averages:
        raise ValueError("at least one class average is required")
    classes = {}
    expected_shape = None
    for class_number, image in class_averages.items():
        if type(class_number) is not int or class_number <= 0:
            raise ValueError("Class Numbers must be positive one-based integers")
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError("class averages must be square 2D images")
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError("all class averages must share one native box size")
        classes[class_number] = image.copy()
    return dict(sorted(classes.items()))


def _match_class(class_number, raw_class, reference, family, *, pixel_size_A, config):
    score_config = config.score_config()
    period = family.roll_period_degrees
    coarse_angles = np.arange(0.0, period, config.roll_coarse_step_degrees)
    best = _best_transform(
        raw_class,
        reference.projection,
        coarse_angles,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
        shift_bound_fraction=config.shift_bound_fraction,
    )
    refined_angles = np.arange(
        best[1] - config.roll_coarse_step_degrees,
        best[1] + config.roll_coarse_step_degrees + 0.5 * config.roll_refine_step_degrees,
        config.roll_refine_step_degrees,
    )
    refined_angles = np.unique(np.mod(refined_angles, period))
    best = _best_transform(
        raw_class,
        reference.projection,
        refined_angles,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
        shift_bound_fraction=config.shift_bound_fraction,
        initial=best,
    )
    score, angle, shift_xy, matched, raw_correlation, metadata = best
    try:
        mirrored_best = _best_transform(
            np.fliplr(raw_class),
            reference.projection,
            coarse_angles,
            pixel_size_A=pixel_size_A,
            score_config=score_config,
            shift_bound_fraction=config.shift_bound_fraction,
        )
        mirrored_refined_angles = np.arange(
            mirrored_best[1] - config.roll_coarse_step_degrees,
            mirrored_best[1]
            + config.roll_coarse_step_degrees
            + 0.5 * config.roll_refine_step_degrees,
            config.roll_refine_step_degrees,
        )
        mirrored_best = _best_transform(
            np.fliplr(raw_class),
            reference.projection,
            np.unique(np.mod(mirrored_refined_angles, period)),
            pixel_size_A=pixel_size_A,
            score_config=score_config,
            shift_bound_fraction=config.shift_bound_fraction,
            initial=mirrored_best,
        )
        mirrored_score = mirrored_best[0]
    except AxisClassScoreError:
        mirrored_score = None
    warnings = ()
    if (
        mirrored_score is not None
        and mirrored_score >= score + config.mirror_warning_margin
    ):
        warnings = (
            "Mirrored diagnostic score exceeds the normal Axis Class Score; "
            "the source class was not flipped.",
        )
    unshifted_class = shift_image(
        raw_class,
        shift=(-shift_xy[1], -shift_xy[0]),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    aligned_class = rotate_image(
        unshifted_class,
        -angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return AxisCandidate(
        family_name=family.name,
        class_number=class_number,
        raw_class=raw_class.copy(),
        aligned_class=np.flipud(aligned_class).astype(np.float32, copy=False),
        exact_reference=reference.projection.copy(),
        exact_reference_display=reference.display_projection.copy(),
        exact_rotation_matrix=family.canonical_camera_matrix.copy(),
        exact_score=score,
        raw_correlation=raw_correlation,
        roll_degrees=angle,
        shift_xy_pixels=(float(shift_xy[0]), float(shift_xy[1])),
        score_metadata=metadata,
        mirrored_score=mirrored_score,
        warnings=warnings,
    )


def _best_transform(
    target,
    reference,
    angles,
    *,
    pixel_size_A,
    score_config,
    shift_bound_fraction,
    initial=None,
):
    best = initial
    bound = int(np.floor(target.shape[0] * shift_bound_fraction))
    for angle in angles:
        rotated = rotate_image(
            reference,
            float(angle),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        for shift_y in range(-bound, bound + 1):
            for shift_x in range(-bound, bound + 1):
                shift_xy = np.array([shift_x, shift_y], dtype=float)
                matched = shift_image(
                    rotated,
                    shift=(shift_y, shift_x),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                score_result = compute_diagnostic_band_limited_score(
                    target,
                    matched,
                    pixel_size_A=pixel_size_A,
                    settings=score_config,
                )
                if not score_result.valid:
                    continue
                metadata = dict(score_result.metadata)
                metadata.update(
                    {
                        "score_role": "axis_class_ranking",
                        "score_provenance": (
                            "soft_masked_physical_band_limited_wzncc"
                        ),
                    }
                )
                raw_correlation = _correlation(target, matched)
                candidate = (
                    float(score_result.score),
                    float(angle),
                    shift_xy,
                    matched,
                    raw_correlation,
                    metadata,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        raise AxisClassScoreError(
            "invalid Axis Class Score; ranking did not fall back to raw correlation"
        )
    return best


def _refine_candidate(candidate, matching_map, pixel_size_A, config):
    family = get_axis_family("I", candidate.family_name)
    score_config = BandLimitedScoreConfig(
        low_resolution_A=candidate.score_metadata[
            "band_low_resolution_A_requested"
        ],
        high_resolution_A=candidate.score_metadata[
            "band_high_resolution_A_requested"
        ],
        mask_radius_fraction=(
            candidate.score_metadata["mask_radius_px"]
            / candidate.raw_class.shape[0]
        ),
        mask_edge_fraction=(
            candidate.score_metadata["mask_edge_width_px"]
            / candidate.raw_class.shape[0]
        ),
    )
    coarse_values = np.arange(
        -config.cone_degrees,
        config.cone_degrees + 0.5 * config.coarse_step_degrees,
        config.coarse_step_degrees,
    )
    coarse_offsets = [
        (tilt_x, tilt_y)
        for tilt_x in coarse_values
        for tilt_y in coarse_values
        if np.hypot(tilt_x, tilt_y) <= config.cone_degrees + 1e-9
    ]
    best = _best_near_axis_offset(
        candidate,
        family,
        matching_map,
        coarse_offsets,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
    )
    center_x, center_y = best[1]
    refine_x = np.arange(
        center_x - config.coarse_step_degrees,
        center_x + config.coarse_step_degrees + 0.5 * config.refine_step_degrees,
        config.refine_step_degrees,
    )
    refine_y = np.arange(
        center_y - config.coarse_step_degrees,
        center_y + config.coarse_step_degrees + 0.5 * config.refine_step_degrees,
        config.refine_step_degrees,
    )
    refined_offsets = [
        (tilt_x, tilt_y)
        for tilt_x in refine_x
        for tilt_y in refine_y
        if np.hypot(tilt_x, tilt_y) <= config.cone_degrees + 1e-9
    ]
    best = _best_near_axis_offset(
        candidate,
        family,
        matching_map,
        refined_offsets,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
        initial=best,
    )
    score, (tilt_x, tilt_y), camera, projection = best
    angular_distance = float(np.hypot(tilt_x, tilt_y))
    on_boundary = bool(
        angular_distance >= config.cone_degrees - 0.5 * config.refine_step_degrees
    )
    warnings = (
        (
            "Best near-axis orientation touches the configured cone boundary; "
            "the cone was not expanded.",
        )
        if on_boundary
        else ()
    )
    return AxisRefinedCandidate(
        exact_candidate=candidate,
        refined_score=score,
        angular_distance_degrees=angular_distance,
        near_axis_rotation_matrix=camera,
        near_axis_projection=projection,
        near_axis_projection_display=np.flipud(projection).copy(),
        cone_boundary=on_boundary,
        warnings=warnings,
    )


def _best_near_axis_offset(
    candidate,
    family,
    matching_map,
    offsets,
    *,
    pixel_size_A,
    score_config,
    initial=None,
):
    best = initial
    for tilt_x, tilt_y in offsets:
        tilt_rotation = Rotation.from_rotvec(
            np.deg2rad([float(tilt_y), float(tilt_x), 0.0])
        ).as_matrix()
        camera = tilt_rotation @ family.canonical_camera_matrix
        canonical_projection = project_volume_at_rotation(matching_map, camera)
        matched_projection = rotate_image(
            canonical_projection,
            candidate.roll_degrees,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        matched_projection = shift_image(
            matched_projection,
            shift=(candidate.shift_xy_pixels[1], candidate.shift_xy_pixels[0]),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32, copy=False)
        score_result = compute_diagnostic_band_limited_score(
            candidate.raw_class,
            matched_projection,
            pixel_size_A=pixel_size_A,
            settings=score_config,
        )
        if not score_result.valid:
            continue
        proposed = (
            float(score_result.score),
            (float(tilt_x), float(tilt_y)),
            camera,
            canonical_projection.astype(np.float32, copy=False),
        )
        if best is None or proposed[0] > best[0]:
            best = proposed
    if best is None:
        raise AxisClassScoreError("invalid refined Axis Class Score")
    return best


def _correlation(left, right):
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)
