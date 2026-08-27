"""Image-only Symmetry-Axis Class Search domain operations."""

from dataclasses import dataclass, field, replace
from time import monotonic

import numpy as np
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage import shift as shift_image
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.axis_projection import project_axis_reference
from cryosparc_2d_projection.axis_registry import (
    AxisFamilyRecord,
    axis_family_records,
    get_axis_family,
)
from cryosparc_2d_projection.matching_grid import (
    prepare_matching_grid,
    prepare_native_matching_grid,
)
from cryosparc_2d_projection.projection import project_volume_at_rotation
from cryosparc_2d_projection.scoring import (
    BandLimitedScoreConfig,
    _apply_frequency_response,
    _butterworth_bandpass,
    _soft_circular_weights,
    compute_diagnostic_band_limited_score,
)


class AxisClassScoreError(RuntimeError):
    """An Axis Class Score could not be calculated safely."""


@dataclass(frozen=True)
class AxisSearchProgress:
    stage: str
    family_name: str
    class_number: int
    pass_name: str
    completed: int
    total: int
    evaluation_count: int
    elapsed_seconds: float
    eta_seconds: float | None


@dataclass(frozen=True)
class AxisSearchConfig:
    search_max_size: int = 128
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
        if type(self.search_max_size) is not int or self.search_max_size <= 0:
            raise ValueError("search max size must be a positive integer")
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
    search_projection: np.ndarray
    canonical_axis_rotation_matrix: np.ndarray
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
    canonical_near_axis_rotation_matrix: np.ndarray
    near_axis_rotation_matrix: np.ndarray
    near_axis_projection: np.ndarray
    near_axis_projection_display: np.ndarray
    matched_search_projection: np.ndarray
    roll_degrees: float
    shift_xy_pixels: tuple[float, float]
    cone_boundary: bool
    score_metadata: dict[str, object] = field(default_factory=dict)
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
    progress_callback=None,
):
    """Refine only the selected Exact-Axis rows inside a bounded cone."""

    if not isinstance(exact_result, AxisSearchResult):
        raise TypeError("exact_result must be an AxisSearchResult")
    config = config or AxisProximityConfig()
    if not isinstance(config, AxisProximityConfig):
        raise TypeError("config must be an AxisProximityConfig")
    if not exact_result.rows:
        return AxisRefinementResult(exact_result=exact_result, rows=())
    search_max_size = int(
        exact_result.rows[0].score_metadata.get("search_box_size", 128)
    )
    grid = prepare_matching_grid(
        exact_result.rows[0].raw_class,
        matching_map,
        class_pixel_size=class_pixel_size_A,
        volume_pixel_size=map_pixel_size_A,
        max_size=search_max_size,
    )
    rows = []
    for candidate in exact_result.rows:
        search_class = _prepare_class_on_search_grid(
            candidate.raw_class,
            source_pixel_size_A=class_pixel_size_A,
            target_size=grid.class_average.shape[0],
            target_pixel_size_A=grid.pixel_size,
        )
        rows.append(
            _refine_candidate(
                candidate,
                search_class,
                grid.volume,
                grid.pixel_size,
                config,
                progress_callback=progress_callback,
            )
        )
    return AxisRefinementResult(exact_result=exact_result, rows=tuple(rows))


def rank_axis_families(
    class_averages,
    matching_map,
    *,
    families=None,
    class_pixel_size_A,
    map_pixel_size_A,
    config=None,
    progress_callback=None,
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
            progress_callback=progress_callback,
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
    progress_callback=None,
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
    grid = prepare_matching_grid(
        first_class,
        matching_map,
        class_pixel_size=class_pixel_size_A,
        volume_pixel_size=map_pixel_size_A,
        max_size=config.search_max_size,
    )
    reference = project_axis_reference(
        grid.volume,
        family_record,
        pixel_size_A=grid.pixel_size,
    )
    candidates = []
    for class_number, image in classes.items():
        search_class = _prepare_class_on_search_grid(
            image,
            source_pixel_size_A=class_pixel_size_A,
            target_size=grid.class_average.shape[0],
            target_pixel_size_A=grid.pixel_size,
        )
        candidates.append(
            _match_class(
                class_number,
                image,
                search_class,
                reference,
                family_record,
                native_pixel_size_A=class_pixel_size_A,
                pixel_size_A=grid.pixel_size,
                config=config,
                progress_callback=progress_callback,
            )
        )
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


def _prepare_class_on_search_grid(
    image,
    *,
    source_pixel_size_A,
    target_size,
    target_pixel_size_A,
):
    image = np.asarray(image, dtype=np.float32)
    factor = float(source_pixel_size_A) / float(target_pixel_size_A)
    prepared = (
        image.copy()
        if np.isclose(factor, 1.0)
        else zoom(
            image,
            zoom=factor,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    )
    result = np.zeros((target_size, target_size), dtype=np.float32)
    copied_size = min(prepared.shape[0], target_size)
    source_start = (prepared.shape[0] - copied_size) // 2
    target_start = (target_size - copied_size) // 2
    result[
        target_start : target_start + copied_size,
        target_start : target_start + copied_size,
    ] = prepared[
        source_start : source_start + copied_size,
        source_start : source_start + copied_size,
    ]
    return result


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


def _match_class(
    class_number,
    raw_class,
    search_class,
    reference,
    family,
    *,
    native_pixel_size_A,
    pixel_size_A,
    config,
    progress_callback,
):
    score_config = config.score_config()
    period = family.roll_period_degrees
    coarse_angles = np.arange(0.0, period, config.roll_coarse_step_degrees)
    best = _best_transform(
        search_class,
        reference.projection,
        coarse_angles,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
        shift_bound_fraction=config.shift_bound_fraction,
        progress_callback=progress_callback,
        family_name=family.name,
        class_number=class_number,
        pass_name="normal-coarse",
    )
    refined_angles = np.arange(
        best[1] - config.roll_coarse_step_degrees,
        best[1] + config.roll_coarse_step_degrees + 0.5 * config.roll_refine_step_degrees,
        config.roll_refine_step_degrees,
    )
    refined_angles = np.unique(np.mod(refined_angles, period))
    best = _best_transform(
        search_class,
        reference.projection,
        refined_angles,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
        shift_bound_fraction=config.shift_bound_fraction,
        initial=best,
        progress_callback=progress_callback,
        family_name=family.name,
        class_number=class_number,
        pass_name="normal-refine",
    )
    score, angle, shift_xy, matched, raw_correlation, metadata = best
    metadata.update(
        {
            "roll_coarse_step_degrees": float(config.roll_coarse_step_degrees),
            "roll_refine_step_degrees": float(config.roll_refine_step_degrees),
            "shift_bound_fraction": float(config.shift_bound_fraction),
            "translation_strategy": "fft_normalized_cross_correlation",
        }
    )
    mirror_evaluation_count = 0
    try:
        mirrored_best = _best_transform(
            np.fliplr(search_class),
            reference.projection,
            coarse_angles,
            pixel_size_A=pixel_size_A,
            score_config=score_config,
            shift_bound_fraction=config.shift_bound_fraction,
            progress_callback=progress_callback,
            family_name=family.name,
            class_number=class_number,
            pass_name="mirror-coarse",
        )
        mirrored_refined_angles = np.arange(
            mirrored_best[1] - config.roll_coarse_step_degrees,
            mirrored_best[1]
            + config.roll_coarse_step_degrees
            + 0.5 * config.roll_refine_step_degrees,
            config.roll_refine_step_degrees,
        )
        mirrored_best = _best_transform(
            np.fliplr(search_class),
            reference.projection,
            np.unique(np.mod(mirrored_refined_angles, period)),
            pixel_size_A=pixel_size_A,
            score_config=score_config,
            shift_bound_fraction=config.shift_bound_fraction,
            initial=mirrored_best,
            progress_callback=progress_callback,
            family_name=family.name,
            class_number=class_number,
            pass_name="mirror-refine",
        )
        mirrored_score = mirrored_best[0]
        mirror_evaluation_count = int(
            mirrored_best[5].get("search_evaluation_count", 0)
        )
    except AxisClassScoreError:
        mirrored_score = None
    normal_evaluation_count = int(metadata["search_evaluation_count"])
    metadata.update(
        {
            "normal_search_evaluation_count": normal_evaluation_count,
            "mirror_search_evaluation_count": mirror_evaluation_count,
            "search_evaluation_count": (
                normal_evaluation_count + mirror_evaluation_count
            ),
        }
    )
    warnings = ()
    if (
        mirrored_score is not None
        and mirrored_score >= score + config.mirror_warning_margin
    ):
        warnings = (
            "Mirrored diagnostic score exceeds the normal Axis Class Score; "
            "the source class was not flipped.",
        )
    native_shift_xy = shift_xy * (pixel_size_A / float(native_pixel_size_A))
    unshifted_class = shift_image(
        raw_class,
        shift=(-native_shift_xy[1], -native_shift_xy[0]),
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
        search_projection=matched.astype(np.float32, copy=True),
        canonical_axis_rotation_matrix=family.canonical_camera_matrix.copy(),
        exact_rotation_matrix=(
            Rotation.from_euler("z", -angle, degrees=True).as_matrix()
            @ family.canonical_camera_matrix
        ),
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
    progress_callback=None,
    family_name,
    class_number,
    pass_name,
):
    best = initial
    bound = int(np.floor(target.shape[0] * shift_bound_fraction))
    angles = tuple(float(angle) for angle in angles)
    started_at = monotonic()
    frequency_response, mask = _axis_score_filter(
        target.shape[0], pixel_size_A, score_config
    )
    filtered_target = _apply_frequency_response(target, frequency_response)
    for completed, angle in enumerate(angles, start=1):
        rotated = rotate_image(
            reference,
            angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        filtered_reference = _apply_frequency_response(rotated, frequency_response)
        score_map = _fft_weighted_normalized_correlation(
            filtered_target, filtered_reference, mask
        )
        center = target.shape[0] - 1
        bounded = score_map[
            center - bound : center + bound + 1,
            center - bound : center + bound + 1,
        ]
        if np.isfinite(bounded).any():
            peak_y, peak_x = np.unravel_index(np.nanargmax(bounded), bounded.shape)
            shift_xy = np.array([peak_x - bound, peak_y - bound], dtype=float)
            matched = shift_image(
                rotated,
                shift=(shift_xy[1], shift_xy[0]),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            candidate = (
                float(bounded[peak_y, peak_x]),
                angle,
                shift_xy,
                matched,
                _correlation(target, matched),
                {},
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        elapsed = monotonic() - started_at
        eta = (
            None
            if completed == 0
            else elapsed * (len(angles) - completed) / completed
        )
        _emit_axis_progress(
            progress_callback,
            AxisSearchProgress(
                stage="exact-ranking",
                family_name=family_name,
                class_number=class_number,
                pass_name=pass_name,
                completed=completed,
                total=len(angles),
                evaluation_count=completed,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            ),
        )
    if best is None:
        raise AxisClassScoreError(
            "invalid Axis Class Score; ranking did not fall back to raw correlation"
        )
    score, angle, shift_xy, matched, raw_correlation, old_metadata = best
    score_result = compute_diagnostic_band_limited_score(
        target,
        matched,
        pixel_size_A=pixel_size_A,
        settings=score_config,
    )
    if not score_result.valid:
        raise AxisClassScoreError(
            "invalid Axis Class Score; ranking did not fall back to raw correlation"
        )
    metadata = dict(score_result.metadata)
    metadata.update(old_metadata)
    metadata.update(
        {
            "score_role": "axis_class_ranking",
            "score_definition": (
                "maximum_band_limited_soft_masked_fft_normalized_cross_correlation"
            ),
            "score_provenance": (
                "band_limited_soft_masked_fft_normalized_cross_correlation"
            ),
            "search_box_size": int(target.shape[0]),
            "search_pixel_size_A": float(pixel_size_A),
            "search_evaluation_count": int(
                old_metadata.get("search_evaluation_count", 0) + len(angles)
            ),
        }
    )
    return score, angle, shift_xy, matched, raw_correlation, metadata


def _axis_score_filter(size, pixel_size_A, settings):
    frequencies = np.fft.fftfreq(size, d=pixel_size_A)
    frequency_y, frequency_x = np.meshgrid(frequencies, frequencies, indexing="ij")
    radial_frequency = np.hypot(frequency_x, frequency_y)
    low_resolution = min(float(settings.low_resolution_A), size * pixel_size_A)
    high_resolution = max(float(settings.high_resolution_A), 2.0 * pixel_size_A)
    response = _butterworth_bandpass(
        radial_frequency,
        low_frequency=1.0 / low_resolution,
        high_frequency=1.0 / high_resolution,
        order=settings.filter_order,
    )
    mask = _soft_circular_weights(
        size,
        inner_radius=size * settings.mask_radius_fraction,
        edge_width=size * settings.mask_edge_fraction,
    )
    return response, mask


def _fft_weighted_normalized_correlation(target, reference, weights):
    weight_sum = float(np.sum(weights))
    target_sum = float(np.sum(weights * target))
    target_variance = float(
        np.sum(weights * target**2) - target_sum**2 / weight_sum
    )
    if target_variance <= 0 or not np.isfinite(target_variance):
        return np.full(
            (2 * target.shape[0] - 1, 2 * target.shape[1] - 1), np.nan
        )
    reversed_reference = reference[::-1, ::-1]
    cross = fftconvolve(weights * target, reversed_reference, mode="full")
    reference_sum = fftconvolve(weights, reversed_reference, mode="full")
    reference_square_sum = fftconvolve(
        weights, (reference**2)[::-1, ::-1], mode="full"
    )
    numerator = cross - target_sum * reference_sum / weight_sum
    reference_variance = reference_square_sum - reference_sum**2 / weight_sum
    denominator = np.sqrt(np.maximum(0.0, target_variance * reference_variance))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[~np.isfinite(result)] = np.nan
    return np.clip(result, -1.0, 1.0)


def _emit_axis_progress(callback, event):
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        pass


def _refine_candidate(
    candidate,
    search_class,
    matching_map,
    pixel_size_A,
    config,
    *,
    progress_callback,
):
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
            / search_class.shape[0]
        ),
        mask_edge_fraction=(
            candidate.score_metadata["mask_edge_width_px"]
            / search_class.shape[0]
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
        search_class,
        family,
        matching_map,
        coarse_offsets,
        pixel_size_A=pixel_size_A,
        score_config=score_config,
        progress_callback=progress_callback,
        pass_name="coarse",
    )
    evaluation_count = len(coarse_offsets)
    intermediate_step = max(
        config.refine_step_degrees, config.coarse_step_degrees / 3.0
    )
    for pass_name, step in (
        ("local-coarse", intermediate_step),
        ("local-refine", config.refine_step_degrees),
    ):
        center_x, center_y = best[1]
        local_offsets = [
            (center_x + dx * step, center_y + dy * step)
            for dx in range(-2, 3)
            for dy in range(-2, 3)
            if np.hypot(center_x + dx * step, center_y + dy * step)
            <= config.cone_degrees + 1e-9
        ]
        best = _best_near_axis_offset(
            candidate,
            search_class,
            family,
            matching_map,
            local_offsets,
            pixel_size_A=pixel_size_A,
            score_config=score_config,
            initial=best,
            progress_callback=progress_callback,
            pass_name=pass_name,
        )
        evaluation_count += len(local_offsets)
    (
        score,
        (tilt_x, tilt_y),
        camera,
        projection,
        roll_degrees,
        shift_xy,
        matched_projection,
        match_metadata,
    ) = best
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
        canonical_near_axis_rotation_matrix=camera,
        near_axis_rotation_matrix=(
            Rotation.from_euler("z", -roll_degrees, degrees=True).as_matrix()
            @ camera
        ),
        near_axis_projection=projection,
        near_axis_projection_display=np.flipud(projection).copy(),
        matched_search_projection=matched_projection,
        roll_degrees=float(roll_degrees),
        shift_xy_pixels=(float(shift_xy[0]), float(shift_xy[1])),
        cone_boundary=on_boundary,
        score_metadata={
            "score_role": "near_axis_refinement",
            "score_provenance": (
                "bounded_hierarchical_projection_search"
            ),
            "search_box_size": int(search_class.shape[0]),
            "search_pixel_size_A": float(pixel_size_A),
            "projection_evaluation_count": int(evaluation_count),
            "translation_strategy": "fft_normalized_cross_correlation",
            "roll_evaluation_count": int(
                evaluation_count
                * match_metadata.get("search_evaluation_count", 0)
            ),
        },
        warnings=warnings,
    )


def _best_near_axis_offset(
    candidate,
    search_class,
    family,
    matching_map,
    offsets,
    *,
    pixel_size_A,
    score_config,
    initial=None,
    progress_callback=None,
    pass_name,
):
    best = initial
    offsets = tuple(offsets)
    started_at = monotonic()
    for completed, (tilt_x, tilt_y) in enumerate(offsets, start=1):
        tilt_rotation = Rotation.from_rotvec(
            np.deg2rad([float(tilt_y), float(tilt_x), 0.0])
        ).as_matrix()
        camera = tilt_rotation @ family.canonical_camera_matrix
        canonical_projection = project_volume_at_rotation(matching_map, camera)
        roll_range = float(
            candidate.score_metadata.get("roll_coarse_step_degrees", 5.0)
        )
        roll_step = float(
            candidate.score_metadata.get("roll_refine_step_degrees", 0.5)
        )
        roll_angles = np.unique(
            np.mod(
                np.arange(
                    candidate.roll_degrees - roll_range,
                    candidate.roll_degrees + roll_range + 0.5 * roll_step,
                    roll_step,
                ),
                family.roll_period_degrees,
            )
        )
        (
            score,
            roll_degrees,
            shift_xy,
            matched_projection,
            _,
            match_metadata,
        ) = _best_transform(
            search_class,
            canonical_projection,
            roll_angles,
            pixel_size_A=pixel_size_A,
            score_config=score_config,
            shift_bound_fraction=float(
                candidate.score_metadata.get("shift_bound_fraction", 0.10)
            ),
            family_name=candidate.family_name,
            class_number=candidate.class_number,
            pass_name="near-roll-shift",
        )
        proposed = (
            float(score),
            (float(tilt_x), float(tilt_y)),
            camera,
            canonical_projection.astype(np.float32, copy=False),
            float(roll_degrees),
            shift_xy,
            matched_projection.astype(np.float32, copy=False),
            match_metadata,
        )
        if best is None or proposed[0] > best[0]:
            best = proposed
        elapsed = monotonic() - started_at
        _emit_axis_progress(
            progress_callback,
            AxisSearchProgress(
                stage="near-axis-refinement",
                family_name=candidate.family_name,
                class_number=candidate.class_number,
                pass_name=pass_name,
                completed=completed,
                total=len(offsets),
                evaluation_count=completed,
                elapsed_seconds=elapsed,
                eta_seconds=(
                    elapsed * (len(offsets) - completed) / completed
                ),
            ),
        )
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
