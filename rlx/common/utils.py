import mlx.core as mx


def get_rewards_to_go(rewards, gamma):
    rewards_to_go = mx.zeros(len(rewards))
    next_reward_to_go = 0
    for t in reversed(range(len(rewards))):
        rewards_to_go[t] = rewards[t] + gamma * next_reward_to_go
        next_reward_to_go = rewards_to_go[t]
    return rewards_to_go
