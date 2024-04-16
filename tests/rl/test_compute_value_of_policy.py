import torch
from problems.rl.compute_value_of_policy import compute_value_of_policy


def test_basic_functionality():
    # Test the function with a basic input
    rewards = torch.tensor([1.0, 0.5, 2.0, 3.0])
    values = torch.tensor([0.5, 1.5, 1.0, 2.0])
    gamma = 0.9
    expected_output = torch.tensor([1.0 + 0.9 * 1.5, 0.5 + 0.9 * 1.0, 2.0 + 0.9 * 2.0, 3.0])
    result = compute_value_of_policy(rewards, values, gamma)
    assert torch.allclose(result, expected_output), "Basic functionality test failed."


def test_single_step_episode():
    # Test the function with a single step episode
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])  # This value should actually not matter
    gamma = 0.9
    expected_output = torch.tensor([1.0])
    result = compute_value_of_policy(rewards, values, gamma)
    assert torch.allclose(result, expected_output), "Single step episode test failed."


def test_zero_discount_factor():
    # Test the function with a zero discount factor
    rewards = torch.tensor([1.0, 0.5, 2.0, 3.0])
    values = torch.tensor([0.5, 1.5, 1.0, 2.0])
    gamma = 0.0
    expected_output = torch.tensor([1.0, 0.5, 2.0, 3.0])  # Values should be ignored
    result = compute_value_of_policy(rewards, values, gamma)
    assert torch.allclose(result, expected_output), "Zero discount factor test failed."


def test_full_discount_factor():
    # Test the function with a full discount factor (edge case)
    rewards = torch.tensor([1.0, 0.5, 2.0, 3.0])
    values = torch.tensor([0.5, 1.5, 1.0, 2.0])
    gamma = 1.0
    expected_output = torch.tensor([1.0 + 1.5, 0.5 + 1.0, 2.0 + 2.0, 3.0])
    result = compute_value_of_policy(rewards, values, gamma)
    assert torch.allclose(result, expected_output), "Full discount factor test failed."
