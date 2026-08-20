import numpy as np

from src.orientation import mean_view_vector


def test_mean_view_vector_returns_normalized_direction():
    vectors = np.array([
        [0, 0, 1],
        [0, 0, 1],
    ])

    result = mean_view_vector(vectors)

    assert np.allclose(result, [0, 0, 1])
    assert np.isclose(np.linalg.norm(result), 1.0)
