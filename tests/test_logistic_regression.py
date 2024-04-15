import numpy as np
from problems.logistic_regression import logistic_regression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_training_accuracy():
    # Define a simple dataset
    X = np.array([
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5]
    ])
    y = np.array([0, 0, 1, 1])
    lr = 0.1
    epochs = 1000

    # Train the logistic regression model
    theta = logistic_regression(X, y, lr, epochs)

    # Predictions using the trained model
    predictions = sigmoid(np.dot(X, theta)) >= 0.5  # Convert probabilities to 0 or 1
    predictions = predictions.astype(int)  # Ensure the predictions are integers for comparison

    # Calculate accuracy
    accuracy = np.mean(predictions == y)

    # Check if the accuracy meets a reasonable threshold
    assert accuracy == 1, "The model should be able to achieve 100% accuracy on this simple dataset."


def test_random_synthetic_dataset():
    # Seed for reproducibility
    np.random.seed(42)

    # Generate random features and binary labels
    n_samples = 100
    n_features = 3
    X = np.random.randn(n_samples, n_features)  # Normally distributed features
    true_theta = np.random.randn(n_features)    # Random true coefficients
    z = np.dot(X, true_theta)                   # Linear combination
    probabilities = sigmoid(z)                  # Apply sigmoid to get probabilities
    y = (probabilities >= 0.5).astype(int)      # Threshold probabilities to get binary labels

    lr = 0.1
    epochs = 1000

    # Train the logistic regression model
    theta = logistic_regression(X, y, lr, epochs)

    # Predictions using the trained model
    predictions = sigmoid(np.dot(X, theta)) >= 0.5
    predictions = predictions.astype(int)

    # Calculate accuracy
    accuracy = np.mean(predictions == y)

    # Check if the accuracy meets a reasonable threshold
    assert accuracy >= 0.9, "The model should achieve at least 90% accuracy on this synthetic dataset"
