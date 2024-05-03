import torch
import gymnasium as gym

from agents.dqn import DQN
from agents.ppo import PPO
from config import DQNConfig, PPOConfig


def main(agent_type):
    env = gym.make('LunarLander-v2')
    match agent_type:
        case 'dqn':
            config = DQNConfig()
            agent = DQN(env, config)
        case 'ppo':
            config = PPOConfig()
            agent = PPO(env, config)
    agent.learn(1e7)
    env.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main('dqn')
