import numpy as np

def k_nearest_neighbors(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, k: int) -> np.ndarray:
    """
    Implement the k-nearest neighbors algorithm.

    Args:
        train_features: A 2D numpy array of features for training data.
        train_labels: A 1D numpy array of labels corresponding to the training data features.
        test_features: A 2D numpy array of features for test data.
        k: The number of nearest neighbors to consider.

    Returns:
        A 1D numpy array containing the predicted labels for each test point.
    """
    n_test = test_features.shape[0]
    predicted_labels = np.zeros(n_test, dtype=int)

    for i, test_sample in enumerate(test_features):
        distances = np.linalg.norm(train_features - test_sample, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        predicted_labels[i] = np.argmax(np.bincount(nearest_labels))

    return predicted_labels
