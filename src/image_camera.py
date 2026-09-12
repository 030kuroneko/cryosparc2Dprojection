"""Image-only global Class Camera search, independent of particle poses."""

from dataclasses import dataclass, replace
from itertools import product
from time import monotonic

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, shift
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.camera import ClassCameraResult
from cryosparc_2d_projection.camera_compute import CameraCompute, GPUOutOfMemory
from cryosparc_2d_projection.projection import find_projection_shift, project_volume_at_rotation
from cryosparc_2d_projection.symmetry import SupportedSymmetry, symmetry_operators


@dataclass(frozen=True)
class ImageCameraSearchConfig:
    quality: str = "standard"
    device: str = "auto"
    batch_size: int = 16

    def __post_init__(self):
        if self.quality not in ("standard", "fine"):
            raise ValueError("search quality must be standard or fine")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("search device must be auto, cpu, or cuda")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch size must be a positive integer")

    @property
    def max_size(self):
        return 32 if self.quality == "standard" else 48


def _coarse_cameras(step):
    angles = np.unique(np.r_[np.arange(0., 360., step), [0., 90., 180., 270.]])
    tilts = np.unique(np.r_[np.arange(0., 180., step), [90., 180.]])
    for tilt in tilts:
        # At either pole, azimuth is already represented by in-plane rotation.
        for azimuth in ([0.] if tilt in (0., 180.) else angles):
            for roll in angles:
                yield Rotation.from_euler("zyz", [azimuth, tilt, roll], degrees=True).as_matrix()


def _distance(left, right, operators):
    traces = np.einsum("sij,ij->s", left @ operators, right)
    return float(np.rad2deg(np.arccos(np.clip((traces.max()-1)/2, -1, 1))))


def _distinct(candidates, operators, separation, limit):
    selected = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(_distance(candidate[1], other[1], operators) > separation for other in selected):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def solve_class_cameras_from_images(class_averages, volume, *, symmetry="C1", config=None,
                                   progress_callback=None, warning_callback=None):
    """Find complete cameras on an already prepared common physical grid.

    ``class_averages`` maps original zero-based class IDs to images. Coarse
    projections are reused across classes. Pixel shifts use this input grid.
    """
    config = config or ImageCameraSearchConfig()
    symmetry = SupportedSymmetry.parse(symmetry).value
    operators = symmetry_operators(symmetry)
    volume = np.asarray(volume, dtype=float)
    images = {key: np.asarray(value, dtype=float) for key, value in class_averages.items()}
    if volume.ndim != 3 or len(set(volume.shape)) != 1 or not np.isfinite(volume).all():
        raise ValueError("search volume must be a finite cubic array")
    size = volume.shape[0]
    if not images or any(image.shape != (size, size) or not np.isfinite(image).all()
                         for image in images.values()):
        raise ValueError("class averages must be finite images on the volume grid")
    input_volume, input_images = volume, images
    input_size = size
    size = min(size, config.max_size)
    scale = input_size / size
    if size < input_size:
        def downsample(array):
            smoothed = gaussian_filter(array, 0.5*np.sqrt(scale**2-1))
            offset = (input_size-1)/2 - scale*(size-1)/2
            return affine_transform(smoothed, matrix=np.eye(array.ndim)*scale,
                                    offset=np.full(array.ndim, offset),
                                    output_shape=(size,)*array.ndim,
                                    order=1, mode="constant", prefilter=False)
        volume = downsample(volume)
        images = {key: downsample(image) for key, image in images.items()}
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - (size-1)/2, yy - (size-1)/2) / size
    weights = 0.5 + 0.5 * np.cos(np.pi * np.clip((radius - 0.4) / 0.1, 0, 1))
    volume_scale = np.max(np.abs(volume))
    if volume_scale == 0:
        raise ValueError("search volume has no usable density")
    volume = volume / volume_scale
    images = {key: image / max(np.max(np.abs(image)), np.finfo(float).tiny)
              for key, image in images.items()}
    if any(np.sum(weights * (image - np.sum(weights*image)/weights.sum())**2) <= 1e-12
           for image in images.values()):
        raise ValueError("class average has no usable contrast")
    bound = int(size * 0.1)
    step = 20. if config.quality == "standard" else 12.
    seed_count = 8 if config.quality == "standard" else 12
    started = monotonic()
    warnings = []

    def warn(message):
        warnings.append(message)
        if warning_callback is not None:
            warning_callback(message)

    backend = None
    try:
        backend = CameraCompute(volume, images, weights, device=config.device,
                                batch_size=config.batch_size, warning_callback=warn)
        results = _search(images, volume, operators, backend, bound, step, seed_count,
                          config, progress_callback)
    except GPUOutOfMemory as error:
        warn(f"{error}; restarting image-only search on CPU.")
        # Discard partial CUDA search before recomputing every class on CPU.
        backend = None
        backend = CameraCompute(volume, images, weights, device="cpu",
                                batch_size=config.batch_size, warning_callback=warn)
        results = _search(images, volume, operators, backend, bound, step, seed_count,
                          config, progress_callback)
    for key, result in results.items():
        result.search_metadata.update({"device": backend.device, "requested_device": config.device,
                                       "batch_size": backend.batch_size, "warnings": warnings,
                                       "elapsed_seconds": monotonic()-started, "symmetry": symmetry,
                                       "selection_box_size": size, "input_box_size": input_size,
                                       "selection_pixel_size_in_input_pixels": scale,
                                       "selection_shift_pixels": result.projection_shift_pixels.tolist(),
                                       "confidence_policy": "heuristic; score >= 0.5 and distinct-group margin >= 0.03"})
        if size != input_size:
            projection = project_volume_at_rotation(input_volume, result.rotation_matrix)
            xy = find_projection_shift(input_images[key], projection)
            results[key] = replace(
                result, projection_shift_pixels=xy,
                matched_projection=shift(projection, (xy[1], xy[0]), order=1,
                                         mode="constant", prefilter=False),
                alternative_orientations=tuple({**candidate,
                    "projection_shift_pixels": (np.asarray(candidate["projection_shift_pixels"])*scale).tolist()
                } for candidate in result.alternative_orientations),
            )
        else:
            results[key] = replace(result, matched_projection=result.matched_projection*volume_scale)
    return results


def _search(images, volume, operators, backend, bound, step, seed_count, config, callback):
    candidates = {key: [] for key in images}
    evaluations = {key: 0 for key in images}
    matrices = list(_coarse_cameras(step))
    last_report = 0.
    completed = 0
    for batch, scores in backend.batches(matrices, images, bound):
        for key in images:
            candidates[key].extend((float(row[0]), matrix, row[1:].copy())
                                   for matrix, row in zip(batch, scores[key], strict=True)
                                   if np.isfinite(row[0]))
            evaluations[key] += len(batch)
        completed += len(batch)
        if callback and (monotonic()-last_report >= 1. or completed == len(matrices)):
            callback(f"Image-only coarse search: {completed}/{len(matrices)} rotations ({backend.device})")
            last_report = monotonic()
    results = {}
    for key, target in images.items():
        if callback:
            callback(f"Image-only local refinement: Class {key + 1} ({backend.device})")
        seeds = _distinct(candidates[key], operators, step, seed_count)
        if not seeds:
            raise ValueError(f"Class {key + 1} has no valid projection match")
        refined = []
        for seed in seeds:
            best = seed
            for increment in ([10., 5., 2., 1.] if config.quality == "standard" else [6., 3., 1., 0.5]):
                for _ in range(2):
                    origin = best[1]
                    matrices = [Rotation.from_euler("xyz", delta, degrees=True).as_matrix() @ origin
                                for delta in product((-increment, 0., increment), repeat=3)]
                    for batch, scores in backend.batches(matrices, [key], bound):
                        evaluations[key] += len(batch)
                        for matrix, row in zip(batch, scores[key], strict=True):
                            if row[0] > best[0] + 1e-10:
                                best = (float(row[0]), matrix, row[1:].copy())
            refined.append(best)
        ranked = _distinct(refined, operators, 5., seed_count)
        score, matrix, xy = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else None
        margin = score - second if second is not None else None
        raw_projection = project_volume_at_rotation(volume, matrix)
        results[key] = ClassCameraResult(
            rotation_matrix=matrix,
            quaternion_xyzw=Rotation.from_matrix(matrix).as_quat(),
            view_direction=matrix[2].copy(),
            in_plane_rotation_degrees=float(Rotation.from_matrix(matrix).as_euler("zyx", degrees=True)[0]),
            matched_projection=shift(raw_projection, (xy[1], xy[0]), order=1, mode="constant", prefilter=False),
            projection_shift_pixels=xy,
            match_score=score,
            second_best_score=second,
            score_margin=margin,
            match_confidence="high" if score >= 0.5 and margin is not None and margin >= 0.03 else "low",
            search_evaluation_count=evaluations[key],
            orientation_method="image_global_search",
            search_metadata={"quality": config.quality,
                             "score": "soft_masked_normalized_cross_correlation",
                             "coarse_step_degrees": step, "shift_bound_pixels": bound,
                             "group_tolerance_degrees": 5.},
            alternative_orientations=tuple({"rotation_matrix": other[1].tolist(),
                                            "score": other[0],
                                            "projection_shift_pixels": other[2].tolist()}
                                           for other in ranked[1:]),
        )
    return results
