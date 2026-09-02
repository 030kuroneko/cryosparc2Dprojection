import argparse
import sys

from cryosparc_2d_projection.external_job import run_external_orientation_job
from cryosparc_2d_projection.external_job_adapter import ExternalJobSource
from cryosparc_2d_projection.surface_render import ClassRenderOptions
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.scoring import BandLimitedScoreConfig
from cryosparc_2d_projection.symmetry import SupportedSymmetry


def parse_class_numbers(value):
    try:
        numbers = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("Classes must be comma-separated positive integers") from error
    if not numbers or any(number < 1 for number in numbers):
        raise ValueError("Class numbers start at 1")
    if len(set(numbers)) != len(numbers):
        raise ValueError("Class numbers must not be repeated")
    return numbers


def parse_supported_symmetry(value):
    try:
        return SupportedSymmetry.parse(value).value
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _integer_at_least(minimum):
    def parse(value):
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(f"value must be at least {minimum}")
        return number

    return parse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a CryoSPARC 5.0.6 External Job for 2D class orientations."
    )
    parser.add_argument("--url", required=True, help="CryoSPARC web URL")
    parser.add_argument("--project", required=True, help="Project UID, e.g. P1")
    parser.add_argument("--workspace", required=True, help="Workspace UID, e.g. W1")
    parser.add_argument("--select-job", required=True, help="Select 2D job UID")
    parser.add_argument(
        "--select-output", default="particles_selected", help="Select 2D particle output"
    )
    parser.add_argument(
        "--templates-output",
        default="templates_selected",
        help="Select 2D class-average template output",
    )
    parser.add_argument("--refinement-job", required=True, help="NU or Local job UID")
    parser.add_argument(
        "--refinement-particles-output",
        default="particles",
        help="NU or Local particle output",
    )
    parser.add_argument(
        "--volume-output", default="volume", help="NU or Local volume output"
    )
    parser.add_argument(
        "--symmetry",
        type=parse_supported_symmetry,
        default="C1",
        help="Refinement symmetry (v0.1: C1 or I)",
    )
    parser.add_argument(
        "--classes",
        type=parse_class_numbers,
        help="One-based class numbers to create interactive volumes for, e.g. 3,8,12",
    )
    parser.add_argument(
        "--surface-level",
        type=float,
        help="Raw density contour for 3D surface renders (default: automatic)",
    )
    parser.add_argument(
        "--render-map",
        choices=("map", "sharpened"),
        default="map",
        help="Map used only for 3D rendering",
    )
    parser.add_argument(
        "--render-background",
        choices=("dark", "light"),
        default="dark",
        help="Background for static 3D renders",
    )
    parser.add_argument(
        "--render-size",
        type=_integer_at_least(64),
        help="Camera View Render PNG size (default: automatic from comparison DPI)",
    )
    parser.add_argument(
        "--render-grid-size",
        type=_integer_at_least(2),
        help=(
            "Maximum 3D grid size used to extract the rendering surface "
            "(default: complete native Rendering Map grid)"
        ),
    )
    parser.add_argument(
        "--comparison-dpi",
        type=_integer_at_least(1),
        default=100,
        help="DPI for all static three-column Class Result images",
    )
    parser.add_argument(
        "--preview-page-size",
        type=_integer_at_least(1),
        default=10,
        help="Number of classes per CryoSPARC preview page",
    )
    parser.add_argument(
        "--auto-crop-2d",
        action="store_true",
        help=(
            "Automatically crop 2D Class Average and Matched Projection panels "
            "in comparison previews"
        ),
    )
    parser.add_argument(
        "--diagnostic-low-resolution-A",
        type=float,
        default=80.0,
        help="Low-resolution edge in Angstrom for the diagnostic score",
    )
    parser.add_argument(
        "--diagnostic-high-resolution-A",
        type=float,
        default=15.0,
        help="High-resolution edge in Angstrom for the diagnostic score",
    )
    parser.add_argument(
        "--diagnostic-mask-radius-fraction",
        type=float,
        default=0.45,
        help="Full-weight mask radius as a fraction of matching box width",
    )
    parser.add_argument(
        "--diagnostic-mask-edge-fraction",
        type=float,
        default=0.05,
        help="Cosine mask edge width as a fraction of matching box width",
    )
    return parser


def main(argv=None, *, client_factory=None):
    args = build_parser().parse_args(argv)

    if client_factory is None:
        from cryosparc.tools import CryoSPARC

        client_factory = CryoSPARC

    client = client_factory(args.url)
    if not client.test_connection():
        raise ConnectionError(f"Could not connect to CryoSPARC at {args.url}")

    project = client.find_project(args.project)
    run_external_orientation_job(
        project,
        workspace_uid=args.workspace,
        select_2d_source=ExternalJobSource(args.select_job, args.select_output),
        select_templates_source=ExternalJobSource(
            args.select_job, args.templates_output
        ),
        refinement_source=ExternalJobSource(
            args.refinement_job, args.refinement_particles_output
        ),
        volume_source=ExternalJobSource(args.refinement_job, args.volume_output),
        symmetry=args.symmetry,
        interactive_class_numbers=args.classes or (),
        render_options=ClassRenderOptions(
            surface_level=args.surface_level,
            map_name=args.render_map,
            background=args.render_background,
            image_size=args.render_size,
            grid_size=args.render_grid_size,
        ),
        diagnostic_score_config=BandLimitedScoreConfig(
            low_resolution_A=args.diagnostic_low_resolution_A,
            high_resolution_A=args.diagnostic_high_resolution_A,
            mask_radius_fraction=args.diagnostic_mask_radius_fraction,
            mask_edge_fraction=args.diagnostic_mask_edge_fraction,
        ),
        comparison_options=ComparisonRenderOptions(
            dpi=args.comparison_dpi,
            page_size=args.preview_page_size,
            auto_crop_2d=args.auto_crop_2d,
        ),
        warning_callback=lambda message: print(
            f"WARNING: {message}", file=sys.stderr
        ),
        status_callback=print,
    )
    return 0
