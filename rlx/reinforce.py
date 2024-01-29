import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class Policy(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(input_dims, output_dims)
            for input_dims, output_dims in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def get_action(self, obs):
        logits = self(obs)
        action = mx.random.categorical(logits)
        return action.item()


def discount_cumsum(rewards, gamma):
    t_steps = mx.arange(rewards.size)
    r = rewards * mx.power(gamma, t_steps)
    r = r[::-1].cumsum()[::-1] / mx.power(gamma, t_steps)
    return r


def compute_loss(log_probs, rewards):
    loss = mx.sum(-log_probs * rewards)
    return loss


def get_log_probs(agent, observations, actions):
    logits = agent(observations)
    probs = nn.softmax(logits)[mx.arange(len(actions)), actions]
    log_probs = -mx.log(probs)
    return log_probs


def train(env, agent, optimizer, loss_and_grad_fn, n_episodes=1000):
    returns = mx.zeros(n_episodes)
    for episode in range(n_episodes):
        observations, actions, rewards = [], [], []
        obs, _ = env.reset()
        while True:
            observations.append(mx.array(obs))
            action = agent.get_action(mx.array(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(mx.array(action))
            rewards.append(mx.array(reward))
            if terminated or truncated:
                break
        observations = mx.array(observations)
        actions = mx.array(actions)
        rewards = mx.array(rewards)
        discounted_rewards = discount_cumsum(rewards, gamma=0.99)
        log_probs = get_log_probs(agent, observations, actions)
        loss, grads = loss_and_grad_fn(log_probs, discounted_rewards)
        optimizer.update(agent, grads)
        mx.eval(agent.parameters(), optimizer.state)
        returns[episode] = mx.sum(rewards)
        if episode % 10 == 0:
            print(f"Episode {episode} reward: {mx.mean(returns[:episode+1])}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    n_episodes = 10000

    input_dim = obs_dim
    num_layers = 1
    hidden_dim = 128
    output_dim = n_actions

    agent = Policy(
        num_layers=num_layers,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    mx.eval(agent.parameters())

    optimizer = optim.Adam(learning_rate=1e-3)

    loss_and_grad_fn = nn.value_and_grad(agent, compute_loss)
    train(env, agent, optimizer, loss_and_grad_fn, n_episodes=n_episodes)
