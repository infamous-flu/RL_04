import time
import random
from typing import Optional, Union

import numpy as np
import torch
import gymnasium as gym

from agents.dqn import DQN
from agents.ppo import PPO
from config.agents_config import DQNConfig, PPOConfig
from config.experiment_config import TrainingConfig, EvaluationConfig
from utils.helpers import nice_box


def train_agent(agent_config: Union[DQNConfig, PPOConfig], training_config: TrainingConfig,
                evaluation_config: Optional[EvaluationConfig] = None) -> Union[DQN, PPO]:
    """
    Train an RL agent over a number of timesteps based on the provided configuration.

    Args:
        agent_config (Union[DQNConfig, PPOConfig]): Configuration object specific to the chosen RL agent.
        training_config (TrainingConfig): ...
        evaluation_config (EvaluationConfig): ...        

    Returns:
        Union[DQN, PPO]: The trained agent object.
    """

    if training_config.evaluate_every > 0 and evaluation_config is None:
        raise ValueError('...')

    if training_config.env_id != evaluation_config.env_id:
        raise ValueError(f'...')

    if training_config.agent_type != evaluation_config.agent_type:
        raise ValueError(f'...')

    # Generate seed if it's not specified
    if training_config.seed is None:
        training_config.seed = int(time.time() * 1000) % (2 ** 32 - 1)

    # Set seeds for reproducibility
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    # Create the training environment
    env = gym.make(training_config.env_id, **training_config.kwargs)

    # Initialize the agent based on the agent type
    match training_config.agent_type:
        case 'dqn':
            if not isinstance(agent_config, DQNConfig):
                raise TypeError('Expected a DQNConfig instance for "dqn" agent type.')
            agent = DQN(env, training_config.device, agent_config, training_config.seed)
        case 'ppo':
            if not isinstance(agent_config, PPOConfig):
                raise TypeError('Expected a PPOConfig instance for "ppo" agent type.')
            agent = PPO(env, training_config.device, agent_config, training_config.seed)
        case _:
            raise ValueError(f'Unknown agent type: {training_config.agent_type}')

    # Train the agent
    agent.learn(training_config, evaluation_config)

    # Close the environment after training
    env.close()

    # Return the trained agent for evaluation
    return agent


def main():
    """Main function to set up configurations, train an agent, and evaluate its performance."""

    # Set up the general experiment configuration
    env_id = 'LunarLander-v2'                                              # The ID of the gym environment
    agent_type = 'dqn'  # or 'dqn'                                         # The type of RL agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # The computational device (CPU or GPU)

    box_width = 82

    # Set up the agent's configuration
    match agent_type:
        case 'dqn':
            agent_config = DQNConfig()
        case 'ppo':
            agent_config = PPOConfig()

    # Set up the training configuration
    training_seed = 73             # Seed for training reproducibility
    training_timesteps = int(1e6)  # Number of timesteps for training

    training_config = TrainingConfig(
        env_id=env_id,
        agent_type=agent_type,
        device=device,
        n_timesteps=training_timesteps,
        seed=training_seed,
    )

    # Set up the evaluation configuration
    evaluation_seed = training_seed + 37   # Different seed value for evaluation
    evaluation_episodes = 10               # Number of episodes for evaluation

    evaluation_config = EvaluationConfig(
        env_id=env_id,
        agent_type=agent_type,
        n_episodes=evaluation_episodes,
        seed=evaluation_seed,
        record_every=6
    )

    # Print a summary of the current configuration
    print('\n' + nice_box(
        width=box_width,
        contents=[(f'Deep Reinforcement Learning', 'c')]
        + [('─'*(box_width-6), 'c')]
        + [('  Training settings:', '')]
        + [(f'    - {key}: {value}', '') for key, value in vars(training_config).items()]
        + [('─'*(box_width-6), 'c')]
        + [('  Evaluation settings:', '')]
        + [(f'    - {key}: {value}', '') for key, value in vars(evaluation_config).items()
           if key not in ['env_id', 'agent_type', 'device', 'video_folder', 'name_prefix']]
        + [('─'*(box_width-6), 'c')]
        + [('  Agent hyperparameters:', '')]
        + [(f'    - {key}: {value}', '') for key, value in vars(agent_config).items()],
        padding=4, thick=True) + '\n'
    )

    print('    ' + 'Training and Evaluation Returns'.center(box_width) + '\n  ' + '─' * 88)

    # Train the agent based on the training configuration
    train_agent(agent_config, training_config, evaluation_config)

    print()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
