import torch
import gymnasium as gym

from agents.dqn import DQN
from agents.ppo import PPO
from config import DQNConfig, PPOConfig


def main(agent_type):
    env = gym.make('LunarLander-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    match agent_type:
        case 'dqn':
            config = DQNConfig(enable_logging=False)
            agent = DQN(env, device, config)
            # agent.load_model('saved_models/dqn/model_20240504005957.pth')
        case 'ppo':
            config = PPOConfig()
            agent = PPO(env, device, config)
            # agent.load_model('saved_models/ppo/model_20240504104541.pth')
    agent.train(1e7)
    env.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main('ppo')
