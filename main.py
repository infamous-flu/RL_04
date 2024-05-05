import torch
import gymnasium as gym

from agents.dqn import DQN
from agents.ppo import PPO
from config import DQNConfig, PPOConfig


def main(agent_type):
    env_id = 'LunarLander-v2'
    env = gym.make(env_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 69420

    match agent_type:
        case 'dqn':
            config = DQNConfig(env_id=env_id)
            agent = DQN(env, device, config, seed)
        case 'ppo':
            config = PPOConfig(env_id=env_id)
            agent = PPO(env, device, config, seed)

    agent.train(1e7)
    env.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main('ppo')
