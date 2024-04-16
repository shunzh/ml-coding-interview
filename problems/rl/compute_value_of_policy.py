import torch


def compute_value_of_policy(
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
):
    """
    Suppose we rolled out a policy and obtained the rewards and values at each time step. Compute the value of the policy.

    Args:
        rewards: A 1D tensor of shape (T,) containing the rewards at each time step.
        values: A 1D tensor of shape (T,) containing the value of each state at each time step.
        gamma: The discount factor.

    Returns:
        A 1D tensor of shape (T,) containing the value of each state under the policy.
    """
    # YOUR CODE HERE
    T = len(rewards)

    # target_values[t] = rewards[t] + gamma * values[t + 1] for t < T
    # target_values[T] = rewards[T]
    shifted_values = torch.cat([values[1:], torch.zeros(1)])
    target_values = rewards + gamma * shifted_values

    return target_values
