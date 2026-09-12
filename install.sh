#!/usr/bin/env bash
# Install the CLI only. Web setup is a separate, explicit command.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo 'Usage: bash install.sh [--manager uv|miniforge] [--miniforge-prefix PATH]'
    echo 'Installs Python 3.12 and the CLI; uv is the default. Does not install a service.'
    echo 'Optional: CRYOSPARC2D_HOME and CRYOSPARC2D_BIN_DIR select installation paths.'
    exit 0
fi
manager=uv
miniforge_prefix=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manager|--miniforge-prefix)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            if [[ "$1" == --manager ]]; then manager="$2"; else miniforge_prefix="$2"; fi
            shift 2 ;;
        *) echo 'Unknown argument. Use --help.' >&2; exit 2 ;;
    esac
done
[[ "$manager" == uv || "$manager" == miniforge ]] || { echo 'Choose uv or miniforge.' >&2; exit 2; }
[[ "$manager" == miniforge || -z "$miniforge_prefix" ]] || { echo '--miniforge-prefix requires --manager miniforge.' >&2; exit 2; }

install_root="${CRYOSPARC2D_HOME:-$HOME/.local/share/cryosparc2d}"
command_dir="${CRYOSPARC2D_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$install_root" "$command_dir"
install_root="$(cd "$install_root" && pwd)"
command_dir="$(cd "$command_dir" && pwd)"

environment_dir="$install_root/venv"
if [[ -d "$environment_dir" ]]; then
    if { [[ "$manager" == uv ]] && [[ -d "$environment_dir/conda-meta" ]]; } ||
       { [[ "$manager" == miniforge ]] && [[ ! -d "$environment_dir/conda-meta" ]]; }; then
        echo 'This installation uses a different manager. Choose separate CRYOSPARC2D_HOME and CRYOSPARC2D_BIN_DIR paths.' >&2
        exit 1
    fi
fi

if [[ "$manager" == uv ]]; then
if command -v uv >/dev/null 2>&1; then
    uv_command="$(command -v uv)"
else
    command -v curl >/dev/null 2>&1 || { echo 'Install curl, then retry.' >&2; exit 1; }
    uv_installer="$(mktemp)"
    trap 'rm -f "$uv_installer"' EXIT
    curl --fail --silent --show-error --location https://astral.sh/uv/install.sh -o "$uv_installer"
    UV_UNMANAGED_INSTALL="$command_dir" sh "$uv_installer"
    uv_command="$command_dir/uv"
fi
fi

source_dir=''
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "$source_dir/pyproject.toml" || ! -f "$source_dir/uv.lock" ]]; then
    command -v git >/dev/null 2>&1 || { echo 'Install Git, then retry.' >&2; exit 1; }
    source_dir="$install_root/source"
    if [[ ! -d "$source_dir" ]]; then
        git clone https://github.com/030kuroneko/cryosparc2Dprojection.git "$source_dir"
    fi
fi

if [[ "$manager" == uv ]]; then
    export UV_PROJECT_ENVIRONMENT="$environment_dir"
    "$uv_command" sync --project "$source_dir" --locked --no-default-groups --python 3.12
else
    if [[ -z "$miniforge_prefix" ]]; then
        miniforge_prefix="$install_root/miniforge3"
        if [[ -x "$HOME/miniforge3/bin/conda" ]]; then miniforge_prefix="$HOME/miniforge3"; fi
    fi
    [[ "$miniforge_prefix" == /* && "$miniforge_prefix" != *' '* && "$environment_dir" != *' '* ]] || {
        echo 'Miniforge requires an absolute prefix and installation paths without spaces.' >&2; exit 1;
    }
    if [[ ! -x "$miniforge_prefix/bin/conda" ]]; then
        [[ ! -e "$miniforge_prefix" ]] || { echo "Not a usable Miniforge installation: $miniforge_prefix" >&2; exit 1; }
        command -v curl >/dev/null 2>&1 || { echo 'Install curl, then retry.' >&2; exit 1; }
        miniforge_os="$(uname -s)"
        [[ "$miniforge_os" != Darwin ]] || miniforge_os=MacOSX
        case "$miniforge_os" in Linux|MacOSX) ;; *) echo 'Use Linux or macOS for this installer.' >&2; exit 1 ;; esac
        miniforge_installer="$(mktemp)"
        trap 'rm -f "$miniforge_installer"' EXIT
        curl --fail --silent --show-error --location "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$miniforge_os-$(uname -m).sh" -o "$miniforge_installer"
        bash "$miniforge_installer" -b -p "$miniforge_prefix"
    fi
    conda_action=create
    [[ ! -d "$environment_dir/conda-meta" ]] || conda_action=install
    "$miniforge_prefix/bin/conda" "$conda_action" --yes --prefix "$environment_dir" --override-channels --channel conda-forge python=3.12 pip uv
    "$environment_dir/bin/python" -m pip install -e "$source_dir"
fi
for command_path in "$environment_dir"/bin/cryosparc2d*; do
    [[ -f "$command_path" ]] || continue
    destination="$command_dir/$(basename "$command_path")"
    if [[ -e "$destination" || -L "$destination" ]]; then
        if [[ ! -L "$destination" || "$(readlink "$destination")" != "$command_path" ]]; then
            echo "Cannot replace existing command: $destination" >&2
            exit 1
        fi
    else
        ln -s "$command_path" "$destination"
    fi
done

echo "CLI installed. Commands: $command_dir"
echo 'Try: cryosparc2d-projection --help'
echo 'Add the browser GUI and background service: cryosparc2d-service install'
case ":$PATH:" in
    *":$command_dir:"*) ;;
    *) printf 'Add this directory to PATH (or use the full command path):\nexport PATH=%q:"$PATH"\n' "$command_dir" ;;
esac
