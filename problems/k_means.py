import collections

import numpy as np


def k_means_clustering(data: np.ndarray, k: int, epsilon: float) -> np.ndarray:
    """
    Implement the k-means clustering algorithm.

    Args:
        data: A 2D numpy array of shape (n, d) where n is the number of data points and d is the number of features.
        k: The number of clusters.
        epsilon: The stopping criterion. The algorithm should stop when the maximum change in the centroids is less
            than epsilon.

    Returns:
        A 1D numpy array of shape (n,) containing the cluster labels of each data point.
    """
    # YOUR CODE HERE
    n = data.shape[0]

    # Initialize the centroids
    centroid_indices = np.random.choice(n, k, replace=False)
    centroids = data[centroid_indices]

    min_distance = np.inf
    while True:
        centroid_to_point = collections.defaultdict(list)
        point_to_centroid = np.zeros(n, dtype=int)
        total_distance = 0

        # E-step: Assign each data point to the nearest centroid
        for idx, sample in enumerate(data):
            distances = np.linalg.norm(centroids - sample, axis=1)
            closest_centroid = np.argmin(distances)
            total_distance += distances[closest_centroid]

            centroid_to_point[closest_centroid].append(sample)
            point_to_centroid[idx] = closest_centroid

        if total_distance < min_distance - epsilon:
            min_distance = total_distance
        else:
            break

        # M-step: Update the centroids
        for i in range(k):
            centroids[i] = np.mean(centroid_to_point[i], axis=0)

    return point_to_centroid
