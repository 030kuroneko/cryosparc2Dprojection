import json
from dataclasses import replace

import numpy as np
from cryosparc import mrc
import pytest
import cryosparc_2d_projection.external_job as external_job_module

from cryosparc_2d_projection.external_job import (
    NativeReprojectionError,
    SourceOutput,
    run_external_orientation_job,
)
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    SurfaceRenderMemoryError,
    resolve_surface_sampling_grid,
)
from cryosparc_2d_projection.scoring import BandLimitedScoreConfig
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from PIL import Image

from tests.external_job_backend import InMemoryExternalJobBackend


def _native_grid_external_job(
    tmp_path,
    *,
    class_size=130,
    rendering_shape=(12, 8, 4),
):
    """Build one fast high-resolution class fixture at the External Job seam."""
    select_2d = np.array(
        [(101, 0, 0.0)],
        dtype=[
            ("uid", "u8"),
            ("alignments2D/class", "i4"),
            ("alignments2D/pose", "f8"),
        ],
    )
    refinement = np.array(
        [(101, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )

    matching_volume = np.zeros((9, 9, 9), dtype=np.float32)
    matching_volume[2:7, 2:7, 2:7] = 1.0
    matching_volume[1, 7, 3] = 3.0
    mrc.write(tmp_path / "matching_volume.mrc", matching_volume, 1.5)

    # Deliberately exceed the bounded search grid by default so native and
    # search projection outputs cannot be confused by having the same
    # dimensions. Small callers can use a native-sized class for fast
    # Rendering Map policy tests.
    class_average = np.zeros((class_size, class_size), dtype=np.float32)
    if class_size > 128:
        class_average[52:78, 52:78] = 1.0
        class_average[28:43, 88:108] = 2.0
        class_average[92:109, 30:47] = 3.0
    else:
        center = class_size // 2
        radius = max(1, class_size // 3)
        yy, xx = np.indices(class_average.shape)
        class_average[(xx - center) ** 2 + (yy - center) ** 2 <= radius**2] = 1.0
        class_average[0, -1] = 2.0
        class_average[-1, 0] = 3.0
    mrc.write(tmp_path / "templates.mrcs", class_average[None, ...], 1.5)
    templates = np.array(
        [("templates.mrcs", 0, 1.5)],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )

    # A non-cubic Rendering Map makes the automatic native-grid metadata
    # observable without affecting the matching map used for scoring.
    rendering_volume = np.zeros(rendering_shape, dtype=np.float32)
    z_size, y_size, x_size = rendering_shape
    z_start, z_stop = max(1, z_size // 5), max(2, z_size - z_size // 5)
    y_start, y_stop = max(1, y_size // 5), max(2, y_size - y_size // 5)
    rendering_volume[z_start:z_stop, y_start:y_stop, :] = 1.0
    z_center = z_size // 2
    y_center = y_size // 2
    z_radius = max(1, z_size // 6)
    y_radius = max(1, y_size // 6)
    rendering_volume[
        max(0, z_center - z_radius) : min(z_size, z_center + z_radius + 1),
        max(0, y_center - y_radius) : min(y_size, y_center + y_radius + 1),
        :,
    ] = 2.0
    mrc.write(tmp_path / "rendering_volume.mrc", rendering_volume, 1.5)
    rendering_volume_sharp = rendering_volume * 2.0
    mrc.write(
        tmp_path / "rendering_volume_sharp.mrc", rendering_volume_sharp, 1.5
    )
    volume = np.array(
        [
            (
                "matching_volume.mrc",
                1.5,
                "rendering_volume_sharp.mrc",
                1.5,
            )
        ],
        dtype=[
            ("map/path", "U128"),
            ("map/psize_A", "f4"),
            ("map_sharp/path", "U128"),
            ("map_sharp/psize_A", "f4"),
        ],
    )

    job = InMemoryExternalJobBackend(
        tmp_path,
        {
            "select_2d_particles": select_2d,
            "select_2d_templates": templates,
            "refinement_particles": refinement,
            "refinement_volume": volume,
        },
    )
    return job, job


def _inconsistent_native_box_external_job(tmp_path):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )

    second_class = np.zeros((11, 11), dtype=np.float32)
    second_class[2:9, 2:9] = 1.0
    second_stack = np.stack([np.zeros_like(second_class), second_class])
    mrc.write(tmp_path / "templates_class_002.mrcs", second_stack, 1.5)

    job.datasets["select_2d_particles"] = np.array(
        [(101, 0, 0.0), (102, 1, 0.0)],
        dtype=[
            ("uid", "u8"),
            ("alignments2D/class", "i4"),
            ("alignments2D/pose", "f8"),
        ],
    )
    job.datasets["refinement_particles"] = np.array(
        [(101, [0.0, 0.0, 0.0]), (102, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )
    job.datasets["select_2d_templates"] = np.array(
        [
            ("templates.mrcs", 0, 1.5),
            ("templates_class_002.mrcs", 1, 1.5),
        ],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )
    return project, job


def _two_class_contrast_external_job(tmp_path):
    project, job = _native_grid_external_job(tmp_path, class_size=64)
    _, base_stack = mrc.read(tmp_path / "templates.mrcs")
    base = np.asarray(base_stack[0], dtype=np.float32)
    rows, columns = np.indices(base.shape)
    structured_noise = 0.01 * np.sin(rows * 0.73 + columns * 0.41)
    contrast = (0.15 * base + structured_noise).astype(np.float32)
    mrc.write(
        tmp_path / "templates_contrast.mrcs",
        np.stack([np.zeros_like(base), contrast]),
        1.5,
    )
    job.datasets["select_2d_particles"] = np.array(
        [(101, 0, 0.0), (102, 1, 0.0)],
        dtype=[
            ("uid", "u8"),
            ("alignments2D/class", "i4"),
            ("alignments2D/pose", "f8"),
        ],
    )
    job.datasets["refinement_particles"] = np.array(
        [(101, [0.0, 0.0, 0.0]), (102, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )
    job.datasets["select_2d_templates"] = np.array(
        [
            ("templates.mrcs", 0, 1.5),
            ("templates_contrast.mrcs", 1, 1.5),
        ],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )
    return project, job


@pytest.mark.parametrize("symmetry", ["C2", "D7", "T", "O", "I1", "I2"])
def test_external_job_rejects_symmetry_outside_v0_1_support_before_creating_job(
    tmp_path, symmetry
):
    project = InMemoryExternalJobBackend(tmp_path, {})

    with pytest.raises(ValueError, match="v0.1 only supports C1 and I"):
        run_external_orientation_job(
            project,
            workspace_uid="W1",
            select_2d_source=SourceOutput("J10", "particles_selected"),
            select_templates_source=SourceOutput("J10", "templates_selected"),
            refinement_source=SourceOutput("J20", "particles"),
            volume_source=SourceOutput("J20", "volume"),
            symmetry=symmetry,
        )

    assert project.created is None


def test_external_job_writes_orientation_results_for_cryosparc_5_0_6(tmp_path):
    select_2d = np.array(
        [(101, 0, np.pi / 2)],
        dtype=[
            ("uid", "u8"),
            ("alignments2D/class", "i4"),
            ("alignments2D/pose", "f8"),
        ],
    )
    refinement = np.array(
        [(101, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )
    volume_data = np.zeros((7, 7, 7), dtype=np.float32)
    volume_data[1, 2, 3] = 1.0
    volume_data[4, 1, 5] = 2.0
    volume_data[5, 5, 1] = 4.0
    mrc.write(tmp_path / "volume.mrc", volume_data, 1.5)
    sharpened_volume_data = volume_data * 2
    mrc.write(tmp_path / "volume_sharp.mrc", sharpened_volume_data, 1.5)
    class_average = np.rot90(volume_data.sum(axis=0))[None, ...]
    mrc.write(tmp_path / "templates.mrcs", class_average, 1.5)
    templates = np.array(
        [("templates.mrcs", 0, 1.5)],
        dtype=[
            ("blob/path", "U128"),
            ("blob/idx", "i4"),
            ("blob/psize_A", "f4"),
        ],
    )
    volume = np.array(
        [("volume.mrc", 1.5, "volume_sharp.mrc", 1.5)],
        dtype=[
            ("map/path", "U128"),
            ("map/psize_A", "f4"),
            ("map_sharp/path", "U128"),
            ("map_sharp/psize_A", "f4"),
        ],
    )
    job = InMemoryExternalJobBackend(
        tmp_path,
        {
            "select_2d_particles": select_2d,
            "select_2d_templates": templates,
            "refinement_particles": refinement,
            "refinement_volume": volume,
        },
    )
    project = job

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="I",
        interactive_class_numbers=(1,),
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=128,
            grid_size=32,
        ),
        diagnostic_score_config=BandLimitedScoreConfig(
            low_resolution_A=6.0,
            high_resolution_A=3.0,
        ),
        comparison_options=ComparisonRenderOptions(dpi=200, page_size=1),
    )

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    assert results["cryosparc_version"] == "5.0.6"
    assert results["symmetry"] == "I"
    assert results["rendering"]["map"] == "sharpened"
    assert results["presentation"] == {
        "comparison_dpi": 200,
        "preview_page_size": 1,
        "requested_render_size": 128,
        "effective_render_size": 128,
        "render_size_was_automatic": False,
        "estimated_page_width_px": 1800,
        "estimated_page_height_px": 600,
        "estimated_page_rgba_memory_bytes": 4320000,
        "warnings": [
            "Requested Camera View Render size 128 px is below the 600 px "
            "recommended for 200 DPI; the third comparison column may appear blurred."
        ],
    }
    class_result = results["classes"][0]
    assert class_result["class_id"] == 0
    assert class_result["class_number"] == 1
    assert class_result["particle_count"] == 1
    assert class_result["angular_spread_degrees"] == 0.0
    assert np.allclose(class_result["camera"]["rotation_matrix"], [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(class_result["camera"]["quaternion_xyzw"], [
        0.0, 0.0, -(2**-0.5), 2**-0.5
    ])
    assert np.allclose(class_result["camera"]["projection_shift_pixels"], [0.0, 0.0])
    assert np.isclose(class_result["camera"]["match_score"], 1.0)
    diagnostic_score = class_result["camera"]["diagnostic_band_limited_score"]
    assert np.isclose(diagnostic_score["score"], 1.0)
    assert diagnostic_score["valid"] is True
    assert diagnostic_score["invalid_reason"] is None
    assert diagnostic_score["score_role"] == "diagnostic_only"
    assert diagnostic_score["band_low_resolution_A_requested"] == 6.0
    assert diagnostic_score["band_high_resolution_A_requested"] == 3.0
    assert diagnostic_score["matching_pixel_size_A"] == 1.5
    assert class_result["camera"]["second_best_score"] < 1.0
    assert class_result["camera"]["score_margin"] > 0.0
    assert class_result["camera"]["match_confidence"] == "high"
    assert class_result["camera"]["matching_box_size"] == 7
    assert class_result["camera"]["matching_pixel_size_A"] == 1.5
    assert class_result["camera"]["search_evaluation_count"] <= 40
    assert class_result["camera"]["coordinate_convention"] == (
        "right-handed Cartesian active rotation; image rows increase downward"
    )
    assert "symmetry_axis" not in class_result
    assert project.created == ("W1", "2D Class Orientation (CryoSPARC 5.0.6)")
    assert ("select_2d_particles", "J10", "particles_selected") in job.connections
    assert ("select_2d_templates", "J10", "templates_selected") in job.connections
    assert ("refinement_particles", "J20", "particles") in job.connections
    assert ("refinement_volume", "J20", "volume") in job.connections
    volume_input = next(spec for spec in job.inputs if spec["name"] == "refinement_volume")
    assert volume_input["slots"] == ["map", "map_sharp"]
    projection_header, projections = mrc.read(tmp_path / "class_projections.mrcs")
    assert np.isclose(projection_header.xlen / projection_header.nx, 1.5)
    assert projections.shape == (1, 7, 7)
    assert np.allclose(projections[0], class_average[0], atol=1e-6)
    assert len(job.plots) == 1
    assert job.plots[0][1] == "Class camera preview 1/1"
    assert job.plots[0][2] == ["png"]
    assert job.plots[0][3] == {
        "dpi": 200,
        "bbox_inches": "tight",
        "pad_inches": 0,
    }
    preview = job.plots[0][0]
    assert len(preview.axes) == 3
    assert np.allclose(
        preview.axes[0].images[0].get_array(), np.flipud(class_average[0])
    )
    assert np.allclose(
        preview.axes[1].images[0].get_array(), np.flipud(class_average[0])
    )
    assert (tmp_path / "renders" / "class_001_exact.png").exists()
    assert not (tmp_path / "renders" / "class_001_oblique.png").exists()
    assert (tmp_path / "renders" / "class_001_comparison.png").exists()
    with Image.open(tmp_path / "renders" / "class_001_comparison.png") as comparison:
        assert comparison.size == (1800, 600)
    projection_output, thumbnail = job.saved_outputs["matched_projections"]
    assert projection_output["blob/path"].tolist() == [
        ">J99/class_projections.mrcs"
    ]
    assert projection_output["blob/idx"].tolist() == [0]
    assert projection_output["blob/shape"].tolist() == [[7, 7]]
    assert np.allclose(projection_output["blob/psize_A"], [1.5])
    assert thumbnail is None
    thumbnail_path = job.output_images["matched_projections"]
    assert thumbnail_path == tmp_path / "renders" / "matched_projections_thumbnail.png"
    assert job.tile_images == [thumbnail_path]
    with Image.open(thumbnail_path) as output_thumbnail:
        assert output_thumbnail.size == (7, 7)
        assert output_thumbnail.mode == "L"
        thumbnail_pixels = np.asarray(output_thumbnail)
        assert thumbnail_pixels[3, 2] == 64
        assert thumbnail_pixels[5, 1] == 128
        assert thumbnail_pixels[1, 5] == 255
    rendering_output, rendering_thumbnail = job.saved_outputs["rendering_map"]
    assert rendering_output["map/path"].tolist() == ["volume_sharp.mrc"]
    assert rendering_thumbnail is None
    rendering_spec = next(spec for spec in job.outputs if spec["name"] == "rendering_map")
    assert rendering_spec["type"] == "volume"
    assert rendering_spec["slots"] == ["map"]
    assert "passthrough" not in rendering_spec
    _, interactive_volume = mrc.read(tmp_path / "class_001_volume.mrc")
    assert np.allclose(
        interactive_volume, np.rot90(sharpened_volume_data, axes=(1, 2))
    )
    class_volume_output, _ = job.saved_outputs["class_001_volume"]
    assert class_volume_output["map/path"].tolist() == [
        ">J99/class_001_volume.mrc"
    ]
    assert (tmp_path / "chimerax" / "class_001.cxc").exists()
    assert (tmp_path / "chimerax" / "all_classes.cxc").exists()
    assert any(message.startswith("Surface Level:") for message in job.logs)
    assert any("third comparison column may appear blurred" in message for message in job.logs)


def test_external_job_separates_native_matched_and_bounded_search_projections(
    tmp_path,
):
    project, job = _native_grid_external_job(tmp_path)

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=64,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(dpi=600, page_size=1),
    )

    matched_output, _ = job.saved_outputs["matched_projections"]
    search_output, _ = job.saved_outputs["search_projections"]
    assert matched_output["blob/shape"].tolist() == [[130, 130]]
    assert search_output["blob/shape"].tolist() == [[128, 128]]
    assert np.allclose(matched_output["blob/psize_A"], [1.5])
    assert np.allclose(search_output["blob/psize_A"], [1.5234375])

    matched_path = str(matched_output["blob/path"][0]).split("/", 1)[1]
    search_path = str(search_output["blob/path"][0]).split("/", 1)[1]
    assert matched_path != search_path
    _, matched = mrc.read(tmp_path / matched_path)
    _, search = mrc.read(tmp_path / search_path)
    assert matched.shape == (1, 130, 130)
    assert search.shape == (1, 128, 128)

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    camera = results["classes"][0]["camera"]
    diagnostic = camera["diagnostic_band_limited_score"]
    assert camera["search_box_size"] == 128
    assert camera["search_pixel_size_A"] == pytest.approx(1.5234375)
    assert camera["matching_box_size"] == 130
    assert camera["matching_pixel_size_A"] == pytest.approx(1.5)
    assert diagnostic["matching_box_size"] == 130
    assert diagnostic["matching_pixel_size_A"] == pytest.approx(1.5)
    assert diagnostic["score_role"] == "diagnostic_only"
    assert diagnostic["candidate_set_scope"] == "raw_search_winner"
    assert camera["search_score_provenance"] == {
        "source": "bounded_search_projection",
        "role": "camera_selection_and_ranking",
        "reported_fields": [
            "match_score",
            "second_best_score",
            "score_margin",
            "match_confidence",
        ],
    }
    assert results["presentation"]["comparison_dpi"] == 600

    with Image.open(tmp_path / "renders" / "class_001_comparison.png") as comparison:
        assert comparison.size == (5400, 1800)

    preview = job.plots[0][0]
    assert preview.axes[1].get_title().startswith("Matched | search raw=")


def test_external_job_auto_crops_only_comparison_2d_panels(tmp_path):
    project, job = _native_grid_external_job(tmp_path, class_size=64)

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=8,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(
            dpi=100,
            page_size=1,
            auto_crop_2d=True,
        ),
    )

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    framing = results["classes"][0]["presentation"]["auto_crop_2d"]
    assert results["presentation"]["auto_crop_2d"]["enabled"] is True
    assert results["presentation"]["auto_crop_2d"]["mode"] == (
        "physical_camera_fov"
    )
    assert framing["camera_view"]["camera_viewport_A"] == pytest.approx(
        results["presentation"]["auto_crop_2d"]["camera_viewport_A"]
    )
    assert framing["camera_view"]["projection_shift_pixels"] == pytest.approx(
        results["classes"][0]["camera"]["projection_shift_pixels"]
    )
    assert framing["foreground_bounds"] is None
    assert framing["silhouette_bounds"] is None
    assert framing["fallback"] is False
    assert framing["zoom"] > 1.0
    assert framing["crop_shape"][0] == framing["crop_shape"][1]
    preview = job.plots[0][0]
    assert preview.axes[0].images[0].get_array().shape == (64, 64)
    assert preview.axes[1].images[0].get_array().shape == (64, 64)
    crop_bounds = framing["crop_bounds"]
    expected_xlim = (
        crop_bounds["left"] - 0.5,
        crop_bounds["right"] - 0.5,
    )
    expected_ylim = (
        crop_bounds["bottom"] - 0.5,
        crop_bounds["top"] - 0.5,
    )
    assert np.allclose(preview.axes[0].get_xlim(), expected_xlim)
    assert np.allclose(preview.axes[0].get_ylim(), expected_ylim)
    assert np.allclose(preview.axes[1].get_xlim(), expected_xlim)
    assert np.allclose(preview.axes[1].get_ylim(), expected_ylim)
    assert preview.axes[2].images[0].get_array().shape == (64, 64, 3)
    _, matched = mrc.read(tmp_path / "class_projections.mrcs")
    assert matched.shape == (1, 64, 64)
    with Image.open(tmp_path / "renders" / "class_001_exact.png") as camera_view:
        assert camera_view.size == (64, 64)


def test_external_job_uses_one_physical_fov_for_contrast_and_noise_variants(
    tmp_path, monkeypatch
):
    project, job = _two_class_contrast_external_job(tmp_path)
    real_project = external_job_module.project_native_matched_projection
    real_viewport = external_job_module.get_surface_camera_viewport_A
    native_results = []
    viewport_calls = []

    def varied_native_projection(*args, **kwargs):
        result = real_project(*args, **kwargs)
        index = len(native_results)
        matched = np.asarray(result.matched_projection, dtype=np.float32)
        if index == 0:
            varied = matched * 0.1
        else:
            rows, columns = np.indices(matched.shape)
            varied = matched * 3.0 + 0.2 * np.sin(rows + columns)
        varied_result = replace(
            result,
            matched_projection=varied.astype(np.float32),
            projection_shift_pixels=np.zeros(2, dtype=float),
        )
        native_results.append(varied_result)
        return varied_result

    def count_viewport(*args, **kwargs):
        viewport = real_viewport(*args, **kwargs)
        viewport_calls.append(viewport)
        return viewport

    monkeypatch.setattr(
        external_job_module,
        "project_native_matched_projection",
        varied_native_projection,
    )
    monkeypatch.setattr(
        external_job_module,
        "get_surface_camera_viewport_A",
        count_viewport,
    )

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=8,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(
            dpi=100,
            page_size=2,
            auto_crop_2d=True,
        ),
    )

    assert len(native_results) == 2
    assert not np.array_equal(
        native_results[0].matched_projection,
        native_results[1].matched_projection,
    )
    _, saved_projections = mrc.read(tmp_path / "class_projections.mrcs")
    assert np.array_equal(
        saved_projections,
        np.stack([result.matched_projection for result in native_results]),
    )
    assert len(viewport_calls) == 1
    results = json.loads((tmp_path / "class_orientations.json").read_text())
    framings = [
        result["presentation"]["auto_crop_2d"]
        for result in results["classes"]
    ]
    assert framings[0]["crop_bounds"] == framings[1]["crop_bounds"]
    assert framings[0]["crop_shape"] == framings[1]["crop_shape"]
    assert framings[0]["zoom"] == pytest.approx(framings[1]["zoom"])
    assert framings[0]["camera_view"] == framings[1]["camera_view"]


def test_external_job_auto_crop_invalid_camera_viewport_reports_one_warning(
    tmp_path, monkeypatch
):
    project, job = _native_grid_external_job(tmp_path, class_size=64)

    def fail_camera_viewport(*args, **kwargs):
        raise ValueError("invalid surface units")

    monkeypatch.setattr(
        external_job_module,
        "get_surface_camera_viewport_A",
        fail_camera_viewport,
    )
    callback_messages = []

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=8,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(
            dpi=100,
            page_size=1,
            auto_crop_2d=True,
        ),
        warning_callback=callback_messages.append,
    )

    warning_text = "Auto-Cropped 2D Framing fell back"
    logged = [message for message in job.logs if warning_text in message]
    callbacks = [message for message in callback_messages if warning_text in message]
    assert len(logged) == 1
    assert len(callbacks) == 1
    assert logged[0] == callbacks[0]
    assert "invalid surface units" in logged[0]
    results = json.loads((tmp_path / "class_orientations.json").read_text())
    framing = results["classes"][0]["presentation"]["auto_crop_2d"]
    assert framing["fallback"] is True
    assert framing["crop_bounds"] == {
        "left": 0,
        "top": 0,
        "right": 64,
        "bottom": 64,
    }
    assert framing["camera_view"] is None


def test_external_job_auto_crop_preserves_scientific_outputs(tmp_path):
    (tmp_path / "disabled").mkdir()
    (tmp_path / "enabled").mkdir()
    disabled_project, disabled_job = _native_grid_external_job(
        tmp_path / "disabled", class_size=64
    )
    enabled_project, enabled_job = _native_grid_external_job(
        tmp_path / "enabled", class_size=64
    )
    common = dict(
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=8,
            surface_level=0.5,
        ),
    )
    run_external_orientation_job(
        disabled_project,
        comparison_options=ComparisonRenderOptions(
            dpi=100, page_size=1, auto_crop_2d=False
        ),
        **common,
    )
    run_external_orientation_job(
        enabled_project,
        comparison_options=ComparisonRenderOptions(
            dpi=100, page_size=1, auto_crop_2d=True
        ),
        **common,
    )

    for filename in ("class_projections.mrcs", "search_projections.mrcs"):
        disabled_header, disabled_values = mrc.read(
            tmp_path / "disabled" / filename
        )
        enabled_header, enabled_values = mrc.read(tmp_path / "enabled" / filename)
        assert disabled_values.shape == enabled_values.shape
        assert np.array_equal(disabled_values, enabled_values)
        assert enabled_header.nx == disabled_header.nx
        assert enabled_header.ny == disabled_header.ny
        assert enabled_header.nz == disabled_header.nz
        assert enabled_header.xlen / enabled_header.nx == pytest.approx(
            disabled_header.xlen / disabled_header.nx
        )

    disabled = json.loads(
        (tmp_path / "disabled" / "class_orientations.json").read_text()
    )
    enabled = json.loads(
        (tmp_path / "enabled" / "class_orientations.json").read_text()
    )
    disabled_class = disabled["classes"][0]
    enabled_class = enabled["classes"][0]
    for key in ("class_id", "class_number", "particle_count", "angular_spread_degrees"):
        assert enabled_class[key] == disabled_class[key]
    for key in ("view_direction",):
        assert np.allclose(enabled_class[key], disabled_class[key])
    enabled_camera = enabled_class["camera"]
    disabled_camera = disabled_class["camera"]
    for key in (
        "rotation_matrix",
        "quaternion_xyzw",
        "view_direction",
        "projection_shift_pixels",
        "search_projection_shift_pixels",
    ):
        assert np.allclose(enabled_camera[key], disabled_camera[key])
    for key in (
        "in_plane_rotation_degrees",
        "match_score",
        "second_best_score",
        "score_margin",
        "matching_box_size",
        "matching_pixel_size_A",
        "search_box_size",
        "search_pixel_size_A",
        "search_evaluation_count",
    ):
        assert enabled_camera[key] == disabled_camera[key]
    assert enabled_camera["match_confidence"] == disabled_camera["match_confidence"]


def test_thumbnail_upload_failure_is_visible_without_losing_scientific_output(
    tmp_path,
):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )
    job.output_image_error = RuntimeError("thumbnail service unavailable")

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=64,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
    )

    assert "matched_projections" in job.saved_outputs
    assert job.output_images == {}
    assert any(
        "Could not attach matched_projections thumbnail" in message
        and "thumbnail service unavailable" in message
        for message in job.logs
    )


def test_tile_thumbnail_upload_failure_is_visible_without_losing_scientific_output(
    tmp_path,
):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )
    job.tile_image_error = RuntimeError("job tile service unavailable")

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=64,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
    )

    assert "matched_projections" in job.saved_outputs
    assert "search_projections" in job.saved_outputs
    assert job.output_images["matched_projections"] == (
        tmp_path / "renders" / "matched_projections_thumbnail.png"
    )
    assert job.tile_images == []
    assert any(
        "Could not attach matched_projections thumbnail to job tile" in message
        and "scientific output remains available" in message
        and "job tile service unavailable" in message
        for message in job.logs
    )


def test_external_job_records_automatic_native_rendering_grid_for_selected_map(
    tmp_path,
):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=64,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
    )

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    rendering = results["rendering"]
    assert rendering["map"] == "sharpened"
    assert rendering["original_shape"] == [6, 4, 3]
    assert rendering["requested_grid_size"] is None
    assert rendering["effective_grid_size"] == 6
    assert rendering["sampled_shape"] == [6, 4, 3]
    assert rendering["grid_size_was_automatic"] is True
    assert rendering["was_downsampled"] is False
    assert rendering["estimated_memory_bytes"] > 0
    assert rendering["memory_estimate_excludes"] == [
        "marching-cubes mesh",
        "plotting allocations",
    ]
    sampling_log = next(
        message for message in job.logs if message.startswith("Surface Sampling Grid:")
    )
    assert "original=6 x 4 x 3" in sampling_log
    assert "requested=native" in sampling_log
    assert "effective=6 x 4 x 3" in sampling_log
    assert "downsampled=no" in sampling_log
    assert "mesh and plotting allocations excluded" in sampling_log


def test_external_job_reports_surface_sampling_before_extraction(tmp_path, monkeypatch):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )
    from cryosparc_2d_projection import external_job

    real_build_surface_model = external_job.build_surface_model

    def assert_sampling_was_reported(*args, **kwargs):
        assert any(
            message.startswith("Surface Sampling Grid:") for message in job.logs
        )
        return real_build_surface_model(*args, **kwargs)

    monkeypatch.setattr(external_job, "build_surface_model", assert_sampling_was_reported)

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=64,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
    )


def test_external_job_records_manual_rendering_grid_override(tmp_path):
    project, _ = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )

    run_external_orientation_job(
        project,
        workspace_uid="W1",
        select_2d_source=SourceOutput("J10", "particles_selected"),
        select_templates_source=SourceOutput("J10", "templates_selected"),
        refinement_source=SourceOutput("J20", "particles"),
        volume_source=SourceOutput("J20", "volume"),
        symmetry="C1",
        render_options=ClassRenderOptions(
            map_name="sharpened",
            image_size=64,
            surface_level=0.5,
            grid_size=4,
        ),
        comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
    )

    results = json.loads((tmp_path / "class_orientations.json").read_text())
    rendering = results["rendering"]
    assert rendering["requested_grid_size"] == 4
    assert rendering["effective_grid_size"] == 4
    assert rendering["sampled_shape"] == [4, 3, 2]
    assert rendering["grid_size_was_automatic"] is False
    assert rendering["was_downsampled"] is True


def test_external_job_logs_actionable_surface_memory_failure(tmp_path, monkeypatch):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )
    failure = SurfaceRenderMemoryError(
        stage="surface extraction",
        sampling_grid=resolve_surface_sampling_grid(
            (512, 384, 256),
            requested_grid_size=512,
        ),
    )

    def fail_surface_build(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        "cryosparc_2d_projection.external_job.build_surface_model",
        fail_surface_build,
    )

    with pytest.raises(SurfaceRenderMemoryError) as raised:
        run_external_orientation_job(
            project,
            workspace_uid="W1",
            select_2d_source=SourceOutput("J10", "particles_selected"),
            select_templates_source=SourceOutput("J10", "templates_selected"),
            refinement_source=SourceOutput("J20", "particles"),
            volume_source=SourceOutput("J20", "volume"),
            symmetry="C1",
            render_options=ClassRenderOptions(
                map_name="sharpened",
                image_size=64,
                surface_level=0.5,
            ),
            comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
        )

    assert raised.value is failure
    assert job.logs[-1] == str(failure)
    assert "--render-grid-size 384" in job.logs[-1]


def test_external_job_rejects_inconsistent_native_class_boxes_before_search(
    tmp_path,
):
    project, job = _inconsistent_native_box_external_job(tmp_path)

    with pytest.raises(ValueError) as error:
        run_external_orientation_job(
            project,
            workspace_uid="W1",
            select_2d_source=SourceOutput("J10", "particles_selected"),
            select_templates_source=SourceOutput("J10", "templates_selected"),
            refinement_source=SourceOutput("J20", "particles"),
            volume_source=SourceOutput("J20", "volume"),
            symmetry="C1",
            render_options=ClassRenderOptions(
                map_name="sharpened",
                image_size=64,
                surface_level=0.5,
            ),
            comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
        )

    message = str(error.value)
    assert "Class 1" in message
    assert "(9, 9)" in message
    assert "Class 2" in message
    assert "(11, 11)" in message
    assert job.saved_outputs == {}
    assert not (tmp_path / "class_orientations.json").exists()
    assert not (tmp_path / "class_projections.mrcs").exists()


def test_external_job_rejects_inconsistent_native_class_pixel_sizes_before_search(
    tmp_path,
):
    project, job = _inconsistent_native_box_external_job(tmp_path)
    same_shape_stack = np.zeros((2, 9, 9), dtype=np.float32)
    same_shape_stack[1, 2:7, 2:7] = 1.0
    mrc.write(tmp_path / "templates_class_002.mrcs", same_shape_stack, 2.0)
    job.datasets["select_2d_templates"]["blob/psize_A"][1] = 2.0

    with pytest.raises(ValueError) as error:
        run_external_orientation_job(
            project,
            workspace_uid="W1",
            select_2d_source=SourceOutput("J10", "particles_selected"),
            select_templates_source=SourceOutput("J10", "templates_selected"),
            refinement_source=SourceOutput("J20", "particles"),
            volume_source=SourceOutput("J20", "volume"),
            symmetry="C1",
            render_options=ClassRenderOptions(
                map_name="sharpened",
                image_size=64,
                surface_level=0.5,
            ),
            comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
        )

    message = str(error.value)
    assert "Class 1" in message
    assert "1.5" in message
    assert "Class 2" in message
    assert "2" in message
    assert job.saved_outputs == {}
    assert not (tmp_path / "class_orientations.json").exists()


def test_external_job_fails_clearly_when_native_reprojection_fails(
    tmp_path, monkeypatch
):
    project, job = _native_grid_external_job(
        tmp_path,
        class_size=9,
        rendering_shape=(6, 4, 3),
    )

    def fail_native_reprojection(*args, **kwargs):
        raise MemoryError("injected native reprojection failure")

    monkeypatch.setattr(
        "cryosparc_2d_projection.external_job.project_native_matched_projection",
        fail_native_reprojection,
    )

    with pytest.raises(NativeReprojectionError) as raised:
        run_external_orientation_job(
            project,
            workspace_uid="W1",
            select_2d_source=SourceOutput("J10", "particles_selected"),
            select_templates_source=SourceOutput("J10", "templates_selected"),
            refinement_source=SourceOutput("J20", "particles"),
            volume_source=SourceOutput("J20", "volume"),
            symmetry="C1",
            render_options=ClassRenderOptions(
                map_name="sharpened",
                image_size=64,
                surface_level=0.5,
            ),
            comparison_options=ComparisonRenderOptions(dpi=100, page_size=1),
        )

    assert "Class 1" in str(raised.value)
    assert "bounded Search Projection was not substituted" in str(raised.value)
    assert str(raised.value) in job.logs
    assert "matched_projections" not in job.saved_outputs
    assert "search_projections" not in job.saved_outputs
    assert not (tmp_path / "class_projections.mrcs").exists()
    assert not (tmp_path / "search_projections.mrcs").exists()


def test_class_render_options_validate_the_whole_rendering_policy():
    import pytest

    with pytest.raises(ValueError, match="rendering map"):
        ClassRenderOptions(map_name="unknown")
    with pytest.raises(ValueError, match="background"):
        ClassRenderOptions(background="blue")
    assert ClassRenderOptions(grid_size=193).grid_size == 193
    with pytest.raises(ValueError, match="64"):
        ClassRenderOptions(image_size=63)
