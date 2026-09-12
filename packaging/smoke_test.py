"""Run from outside the checkout with the installed package's Python."""
import importlib.metadata
import importlib.resources
from pathlib import Path
import subprocess
import sys

assert importlib.metadata.version('cryosparc-2d-projection')
for command in ('cryosparc2d-projection', 'cryosparc2d-axis-search', 'cryosparc2d-service'):
    subprocess.run([str(Path(sys.executable).parent / command), '--help'], check=True)
assets = importlib.resources.files('cryosparc_2d_projection').joinpath('web_assets')
assert any(assets.iterdir()), 'Packaged Web assets are missing'
from cryosparc.dataset import Dataset
assert len(Dataset()) == 0
print('Installed package and CLI smoke checks passed.')
