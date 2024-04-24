"""
Some machine learning interviews focus on evaluating candidates' abilities to work with tensors.
"""
import torch


def pairwise_diff(x, y):
    """
    Compute the pairwise difference between two tensors.

    Args:
        x: (n_x, d) tensor
        y: (n_y, d) tensor

    Returns:
        A tensor with shape of (n_x, n_y), where the element at (i, j) is ||x[i] - y[j]||_2.
    """
    # YOUR CODE HERE
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.norm(x - y, dim=-1)


def generate_grid(x, y):
    """
    Generate a grid of points.

    Args:
        x: (n_x,) tensor
        y: (n_y,) tensor

    Returns:
        A tensor with shape of (n_x, n_y, 2), where the element at (i, j) is (x[i], y[j]).
    """
    # YOUR CODE HERE
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([grid_x, grid_y], dim=-1)
