from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedComparisonRender:
    comparison_dpi: int
    preview_page_size: int
    requested_render_size: int | None
    effective_render_size: int
    render_size_was_automatic: bool
    estimated_page_width_px: int
    estimated_page_height_px: int
    estimated_page_rgba_memory_bytes: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonRenderOptions:
    """Presentation resolution policy for static three-column Class Results."""

    dpi: int = 100
    page_size: int = 10
    auto_crop_2d: bool = False

    def __post_init__(self):
        if type(self.dpi) is not int or self.dpi <= 0:
            raise ValueError("comparison DPI must be a positive integer")
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("preview page size must be a positive integer")
        if type(self.auto_crop_2d) is not bool:
            raise ValueError("auto-crop 2D flag must be boolean")

    def resolve_render_size(self, requested_render_size):
        if requested_render_size is not None:
            return int(requested_render_size)
        return max(1024, 3 * self.dpi)

    def resolve(self, *, class_count, requested_render_size):
        if type(class_count) is not int or class_count <= 0:
            raise ValueError("class count must be a positive integer")
        effective_render_size = self.resolve_render_size(requested_render_size)
        page_rows = min(self.page_size, class_count)
        width = 9 * self.dpi
        height = 3 * page_rows * self.dpi
        warnings = []
        if self.dpi > 600:
            memory_mib = width * height * 4 / (1024**2)
            warnings.append(
                f"Comparison output at {self.dpi} DPI creates preview pages up to "
                f"{width} x {height} pixels (approximately {memory_mib:.1f} MiB RGBA)."
            )
        recommended_render_size = 3 * self.dpi
        if (
            requested_render_size is not None
            and requested_render_size < recommended_render_size
        ):
            warnings.append(
                f"Requested Camera View Render size {requested_render_size} px is below "
                f"the {recommended_render_size} px recommended for {self.dpi} DPI; "
                "the third comparison column may appear blurred."
            )
        return ResolvedComparisonRender(
            comparison_dpi=self.dpi,
            preview_page_size=self.page_size,
            requested_render_size=requested_render_size,
            effective_render_size=effective_render_size,
            render_size_was_automatic=requested_render_size is None,
            estimated_page_width_px=width,
            estimated_page_height_px=height,
            estimated_page_rgba_memory_bytes=width * height * 4,
            warnings=tuple(warnings),
        )
