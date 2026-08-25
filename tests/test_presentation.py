from cryosparc_2d_projection.presentation import ComparisonRenderOptions
import pytest


def test_comparison_resolution_auto_sizes_only_an_unrequested_camera_render():
    options = ComparisonRenderOptions(dpi=600, page_size=1)

    assert options.resolve_render_size(None) == 1800
    assert options.resolve_render_size(512) == 512


@pytest.mark.parametrize(
    "settings",
    [
        {"dpi": 0},
        {"dpi": 100.5},
        {"page_size": 0},
        {"page_size": 2.5},
    ],
)
def test_comparison_resolution_requires_positive_integer_settings(settings):
    with pytest.raises(ValueError):
        ComparisonRenderOptions(**settings)


def test_comparison_resolution_warns_without_overriding_high_quality_requests():
    resolved = ComparisonRenderOptions(dpi=601, page_size=10).resolve(
        class_count=45,
        requested_render_size=512,
    )

    assert resolved.effective_render_size == 512
    assert not resolved.render_size_was_automatic
    assert resolved.estimated_page_width_px == 5409
    assert resolved.estimated_page_height_px == 18030
    assert len(resolved.warnings) == 2
    assert "601 DPI" in resolved.warnings[0]
    assert "512" in resolved.warnings[1]
    assert "1803" in resolved.warnings[1]
