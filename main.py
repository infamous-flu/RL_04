import torch
import gymnasium as gym

from agents.dqn import DQN
from agents.ppo import PPO
from config import DQNConfig, PPOConfig


def main(agent_type):
    env = gym.make('LunarLander-v2', render_mode='human')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    match agent_type:
        case 'dqn':
            config = DQNConfig(enable_logging=False)
            agent = DQN(env, device, config)
            # agent.load_model('saved_models/dqn/model_trained.pth')
        case 'ppo':
            config = PPOConfig()
            agent = PPO(env, device, config)
            agent.load_model('saved_models/ppo/model_trained.pth')

    score = 0
    observation, _ = env.reset(seed=69420)
    for _ in range(10000):
        action = agent.select_action(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        score += reward
        if terminated or truncated:
            print(f'Score: {score:.2f}')
            score = 0
            observation, _ = env.reset()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main('ppo')
