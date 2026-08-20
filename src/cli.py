import argparse

from cryosparc_2d_projection.external_job import (
    SourceOutput,
    run_external_orientation_job,
)


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
        default="C1",
        help="Refinement symmetry: C<n>, D<n>, T, O, I, I1, or I2",
    )
    parser.add_argument(
        "--classes",
        type=parse_class_numbers,
        help="One-based class numbers to create interactive volumes for, e.g. 3,8,12",
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
        select_2d_source=SourceOutput(args.select_job, args.select_output),
        select_templates_source=SourceOutput(args.select_job, args.templates_output),
        refinement_source=SourceOutput(
            args.refinement_job, args.refinement_particles_output
        ),
        volume_source=SourceOutput(args.refinement_job, args.volume_output),
        symmetry=args.symmetry,
    )
    return 0
