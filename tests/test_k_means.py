import numpy as np
from problems.k_means import k_means_clustering


def assert_clusters(labels, expected_groups):
    found_groups = {label: set(np.where(labels == label)[0]) for label in np.unique(labels)}
    expected_set = [set(group) for group in expected_groups]

    assert len(found_groups) == len(expected_set)
    assert all(any(group == found_group for found_group in found_groups.values()) for group in expected_set)


# Use the function in your tests like this:
def test_basic_functionality():
    data = np.array([[1, 2], [1, 1], [10, 10], [10, 11]])
    k = 2
    epsilon = 0.01
    labels = k_means_clustering(data, k, epsilon)
    expected_groups = [[0, 1], [2, 3]]
    assert_clusters(labels, expected_groups)


def test_closely_packed_clusters():
    data = np.array([[1, 1], [2, 2], [1, 2], [5, 5], [6, 5], [5, 6]])
    k = 2
    epsilon = 0.01
    labels = k_means_clustering(data, k, epsilon)
    expected_groups = [[0, 1, 2], [3, 4, 5]]
    assert_clusters(labels, expected_groups)


def test_three_close_clusters():
    def generate_cluster(center, num_points, variance):
        """Generate 'num_points' points around a 'center' with 'variance'."""
        return np.random.normal(loc=center, scale=variance, size=(num_points, len(center)))

    np.random.seed(42)  # For reproducibility
    cluster1 = generate_cluster(center=[1, 1], num_points=100, variance=0.1)
    cluster2 = generate_cluster(center=[100, 1], num_points=100, variance=0.1)
    cluster3 = generate_cluster(center=[1, 100], num_points=100, variance=0.1)

    data = np.vstack([cluster1, cluster2, cluster3])
    k = 3
    epsilon = 0.01

    labels = k_means_clustering(data, k, epsilon)

    # Expected groups: clusters of indices
    expected_groups = [list(range(100)), list(range(100, 200)), list(range(200, 300))]

    # Using the assert_clusters utility function to check grouping correctness
    assert_clusters(labels, expected_groups)
