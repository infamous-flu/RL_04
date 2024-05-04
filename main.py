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
            config = DQNConfig(log_dir='tmp/', epsilon=0, eps_min=0)
            agent = DQN(env, device, config)
            agent.load_model('saved_models/dqn/model_20240504005957.pth')
        case 'ppo':
            config = PPOConfig(log_dir='tmp/')
            agent = PPO(env, device, config)
            agent.load_model('saved_models/ppo/model_20240504104541.pth')
    observation, info = env.reset(seed=42)
    score = 0
    for _ in range(10000):
        action = agent.select_action(observation, True)
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        if terminated or truncated:
            print(f'Score: {score:.2f}')
            score = 0
            observation, info = env.reset()
    env.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main('ppo')
