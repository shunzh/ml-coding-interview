import torch


def compute_ppo_loss(
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float
):
    """
    Compute the PPO loss given the old and new log probabilities of actions,
    and the advantage estimates.

    Args:
        old_log_probs: A 1D tensor of shape (T,) containing the log probabilities of actions
                       taken by the old policy.
        new_log_probs: A 1D tensor of shape (T,) containing the log probabilities of actions
                       taken by the new policy after update.
        advantages: A 1D tensor of shape (T,) containing the advantage estimates.
        clip_ratio: A scalar indicating how much the new policy is allowed to deviate from the old one.

    Returns:
        A scalar tensor representing the PPO loss.
    """
    # YOUR CODE HERE
    # ratios = exp(new_log_probs - old_log_probs)
    ratios = torch.exp(new_log_probs - old_log_probs)
    # clipped_ratios = clip(ratios, 1 - clip_ratio, 1 + clip_ratio)
    clipped_ratios = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
    # loss = -min(ratios * advantages, clipped_ratios * advantages)
    loss = -torch.min(ratios * advantages, clipped_ratios * advantages)

    return loss.mean()
