import numpy as np

from problems.linear_regression import linear_regression_gradient


def test_simple_linear_regression():
    X = np.array([[1, 1], [1, 2], [2, 3]])
    true_coefficients = np.array([1, 2])
    y = np.dot(X, true_coefficients)
    np.testing.assert_allclose(linear_regression_gradient(X, y, 0.01, 500), true_coefficients, rtol=0.1)
