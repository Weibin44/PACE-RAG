import numpy as np

from pace.methods.utils import (
    minmax_normalize,
    normalize_nonnegative_by_max,
)


def test_normalize_nonnegative_by_max():
    result = normalize_nonnegative_by_max(
        np.array([-2.0, 1.0, 3.0])
    )

    np.testing.assert_allclose(result, [0.0, 1.0 / 3.0, 1.0])


def test_minmax_normalize():
    result = minmax_normalize(
        np.array([-2.0, 1.0, 3.0])
    )

    np.testing.assert_allclose(result, [0.0, 0.6, 1.0])


def test_minmax_constant_value_is_explicit():
    values = np.array([2.0, 2.0, 2.0])

    np.testing.assert_array_equal(
        minmax_normalize(values),
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        minmax_normalize(values, constant_value=1.0),
        [1.0, 1.0, 1.0],
    )