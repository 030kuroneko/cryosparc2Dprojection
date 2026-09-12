"""Install and control the optional Linux Web service; imports only stdlib."""
import argparse
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
from urllib.request import urlopen


UNIT = 'cryosparc2d.service'
CONTROLS = ('start', 'stop', 'restart', 'status', 'enable', 'disable')


def main(argv=None, *, root=Path('/'), run=subprocess.run, prompt=input, opener=urlopen):
    parser = argparse.ArgumentParser(description='Manage the optional Web GUI service.')
    parser.add_argument('action', choices=(*CONTROLS, 'install', 'configure'))
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent,
                        help='Source checkout for installation')
    parser.add_argument('--uv', help='uv executable for installation')
    args = parser.parse_args(argv)
    try:
        if platform.system() != 'Linux' or not shutil.which('systemctl'):
            raise ValueError('The managed GUI requires Linux with systemd. '
                             'The scientific CLIs remain available on this platform.')
        if args.action == 'install':
            uv = args.uv or shutil.which('uv')
            if not uv and (Path(sys.executable).parent / 'uv').is_file():
                uv = str(Path(sys.executable).parent / 'uv')
            if not uv:
                raise ValueError('uv was not found. Run bash install.sh first or add uv to PATH.')
            if os.geteuid() != 0:
                return run(['sudo', sys.executable, '-m', 'cryosparc_2d_projection.service_cli',
                            'install', '--source', str(args.source.resolve()), '--uv', uv],
                           check=False).returncode
            from cryosparc_2d_projection.service_setup import Layout, install, setup_lock
            previous_umask = os.umask(0o022)
            try:
                with setup_lock(Layout(root)):
                    install(Layout(root), args.source, uv, run, prompt, opener)
            finally:
                os.umask(previous_umask)
            return 0
        if not (root / 'etc/systemd/system' / UNIT).exists():
            raise ValueError('Web service is not installed. Run: cryosparc2d-service install')
        if args.action == 'configure':
            from cryosparc_2d_projection.service_setup import Layout, configure, setup_lock
            layout = Layout(root)
            if os.geteuid() != 0:
                return run(['sudo', str(layout.python), '-m',
                            'cryosparc_2d_projection.service_cli', 'configure'], check=False).returncode
            with setup_lock(layout):
                configure(layout, run, prompt, opener)
            return 0
        command = [shutil.which('systemctl'), args.action]
        if args.action == 'status':
            command.append('--no-pager')
        command.append(UNIT)
        if os.geteuid() != 0:
            command.insert(0, 'sudo')
        return run(command, check=False).returncode
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'Service command failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
