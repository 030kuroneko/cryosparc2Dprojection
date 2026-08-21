from enum import Enum


class SupportedSymmetry(str, Enum):
    """A symmetry convention verified for the version 0.1 workflow."""

    C1 = "C1"
    I = "I"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError("v0.1 only supports C1 and I") from error
