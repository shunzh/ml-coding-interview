import numpy as np


def linear_regression_matrx_inv(X, y):
    """
    Given a matrix X of features and a vector y of target values, return the
    coefficients of the linear regression model using matrix multiplication.

    Args:
        X: A 2D numpy array of shape (n, d) where n is the number of data points
            and d is the number of features.
        y: A 1D numpy array of shape (n,) containing the target values.
    """
    # YOUR CODE HERE
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta


def linear_regression_gradient(X, y, lr, epoches):
    """
    Given a matrix X of features and a vector y of target values, return the
    coefficients of the linear regression model using gradient descent.

    Initialize the coefficients to be zero.

    Args:
        X: A 2D numpy array of shape (n, d) where n is the number of data points
            and d is the number of features.
        y: A 1D numpy array of shape (n,) containing the target values.
        lr: The learning rate.
        epoches: The number of epoches to train the model.

    Returns:
        A 1D numpy array of shape (d,) containing the coefficients of the linear regression model.
    """
    # YOUR CODE HERE
    n, d = X.shape

    theta = np.zeros(d)

    for _ in range(epoches):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            gradient = (np.dot(xi, theta) - yi) * xi
            theta -= lr * gradient

    return theta
