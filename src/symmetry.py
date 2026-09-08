"""Point-group names and CryoSPARC coordinate conventions."""

from dataclasses import dataclass
import re
from typing import ClassVar

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SupportedSymmetry:
    """A validated point group in the tool's CryoSPARC convention."""

    value: str
    C1: ClassVar["SupportedSymmetry"]
    I: ClassVar["SupportedSymmetry"]

    def __post_init__(self):
        normalized = str(self.value).strip().upper()
        if not re.fullmatch(r"(?:[CD][1-9][0-9]*|T|O|I)", normalized):
            raise ValueError("Supported symmetry: Cn, Dn (positive integer n), T, O, I")
        object.__setattr__(self, "value", normalized)

    @classmethod
    def parse(cls, value):
        return value if isinstance(value, cls) else cls(value)


SupportedSymmetry.C1 = SupportedSymmetry("C1")
SupportedSymmetry.I = SupportedSymmetry("I")


def symmetry_operators(symmetry):
    """Return proper rotations in the declared CryoSPARC map coordinates."""
    name = SupportedSymmetry.parse(symmetry).value
    operators = Rotation.create_group(name, axis="Z").as_matrix()
    if name.startswith("D"):
        # SciPy puts the dyad on X; CryoSPARC puts it on Y.
        change = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        operators = change @ operators @ change.T
    elif name == "T":
        # Map SciPy's (1,1,1) threefold to Z and its X dyad to
        # (0, sqrt(2/3), sqrt(1/3)). See the convention research note.
        change = np.array([
            [0., -1 / np.sqrt(2), 1 / np.sqrt(2)],
            [2 / np.sqrt(6), -1 / np.sqrt(6), -1 / np.sqrt(6)],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        ])
        operators = change @ operators @ change.T
    return operators
