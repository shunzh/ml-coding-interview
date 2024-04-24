import torch

from problems.deep_learning.tensor_ops import pairwise_diff, generate_grid


def test_pairwise_diff():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    expected = torch.tensor([
        [2.0, 2.2361],
        [4.4721, 5.0]
    ])
    result = pairwise_diff(x, y)
    torch.testing.assert_allclose(result, expected)


def test_generate_grid():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    expected = torch.tensor([
        [[1.0, 3.0], [1.0, 4.0]],
        [[2.0, 3.0], [2.0, 4.0]]
    ])
    result = generate_grid(x, y)
    torch.testing.assert_allclose(result, expected)
