import numpy as np

from problems.linear_regression import linear_regression_matrx_inv, linear_regression_gradient


def test_simple_linear_regression():
    X = np.array([[1, 1], [1, 2], [2, 3]])
    true_coefficients = np.array([1, 2])
    y = np.dot(X, true_coefficients)  # perfect linear relationship
    np.testing.assert_allclose(linear_regression_matrx_inv(X, y), true_coefficients, rtol=1e-5)
    np.testing.assert_allclose(linear_regression_gradient(X, y, 0.01, 1000), true_coefficients, rtol=0.1)


def test_multiple_features():
    X = np.array([[1, 2, 3], [1, 3, 5], [2, 3, 5], [3, 5, 8]])
    true_coefficients = np.array([0.5, 1, 1.5])
    y = np.dot(X, true_coefficients)  # perfect linear relationship
    np.testing.assert_allclose(linear_regression_matrx_inv(X, y), true_coefficients, rtol=1e-5)
    np.testing.assert_allclose(linear_regression_gradient(X, y, 0.01, 1000), true_coefficients, rtol=0.1)


def test_noisy_linear_relationship():
    np.random.seed(42)  # For reproducibility
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    true_coefficients = np.array([1.5, 2.5])
    noise = np.random.normal(0, 0.2, X.shape[0])  # Small noise added to the target
    y = X @ true_coefficients + noise
    estimated_coefficients = linear_regression_matrx_inv(X, y)
    # Check if the estimated coefficients are close to the true coefficients within a reasonable tolerance
    np.testing.assert_allclose(estimated_coefficients, true_coefficients, rtol=0.1)
    np.testing.assert_allclose(linear_regression_gradient(X, y, 0.01, 1000), true_coefficients, rtol=0.2)
