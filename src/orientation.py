import numpy as np


def mean_view_vector(vectors):
    """Calculate normalized mean viewing direction."""
    vectors = np.asarray(vectors, dtype=float)
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return mean / norm
