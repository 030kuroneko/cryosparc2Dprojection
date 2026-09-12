"""Keep the optional Web entry point usable in a CLI-only installation."""
import sys


def main(argv=None):
    try:
        import flask  # noqa: F401
        import waitress  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name not in ('flask', 'waitress'):
            raise
        print('Web GUI is not installed. On a Linux server, run: '
              'cryosparc2d-service install\n'
              'For a manual Web session, install the project with its [web] extra.',
              file=sys.stderr)
        return 1
    from cryosparc_2d_projection.web import main as serve
    return serve(argv)


if __name__ == '__main__':
    raise SystemExit(main())
