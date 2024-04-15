import numpy as np
from problems.knn import k_nearest_neighbors  # Adjust the import statement according to your project structure


def test_k_nearest_neighbors_with_generated_data():
    # Setting the random seed for reproducibility
    np.random.seed(42)

    # Generate Cluster 1
    cluster1 = np.random.randn(50, 2) * 0.3 + np.array([1, 1])
    labels1 = np.zeros(50, dtype=int)

    # Generate Cluster 2
    cluster2 = np.random.randn(50, 2) * 0.3 + np.array([4, 4])
    labels2 = np.ones(50, dtype=int)

    # Generate Cluster 3
    cluster3 = np.random.randn(50, 2) * 0.3 + np.array([8, 1])
    labels3 = np.full(50, 2, dtype=int)

    # Combine into a single dataset
    features = np.vstack([cluster1, cluster2, cluster3])
    labels = np.concatenate([labels1, labels2, labels3])

    # Shuffle the dataset
    indices = np.random.permutation(features.shape[0])
    features = features[indices]
    labels = labels[indices]

    # Define test features
    test_features = np.array([[0.9, 1.1], [4.1, 3.9], [8.2, 1.1], [1, 4]])  # The last is an outlier to check robustness

    # Number of nearest neighbors
    k = 3
    # Expected labels for the test set, assuming the last test point is closer to cluster 2
    expected_labels = np.array([0, 1, 2, 1])

    # Run k-nearest neighbors
    predicted_labels = k_nearest_neighbors(features, labels, test_features, k)

    # Assert the predicted labels match the expected labels
    np.testing.assert_array_equal(predicted_labels, expected_labels)
