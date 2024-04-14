import numpy as np
from my_solutions.k_means import k_means_clustering  # Replace 'your_module' with the actual name of your Python file/module


def assert_clusters(data, labels, expected_groups):
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
    assert_clusters(data, labels, expected_groups)


def test_non_integer_data():
    data = np.array([[1.5, 2.5], [1.7, 1.8], [10.1, 10.3], [10.2, 11.4]])
    k = 2
    epsilon = 0.01
    labels = k_means_clustering(data, k, epsilon)
    expected_groups = [[0, 1], [2, 3]]
    assert_clusters(data, labels, expected_groups)


def test_closely_packed_clusters():
    data = np.array([[1, 1], [2, 2], [1, 2], [5, 5], [6, 5], [5, 6]])
    k = 2
    epsilon = 0.01
    labels = k_means_clustering(data, k, epsilon)
    expected_groups = [[0, 1, 2], [3, 4, 5]]
    assert_clusters(data, labels, expected_groups)
