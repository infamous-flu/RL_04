import os
import re
import time
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from agents.dqn import DQN
from agents.ppo import PPO
from config.agents_config import DQNConfig, PPOConfig
from config.experiment_config import TrainingConfig, EvaluationConfig
from utils.helpers import nice_box


def evaluate_agent(agent: Any, evaluation_config: EvaluationConfig) -> float:
    """
    Evaluate an RL agent over a given number of episodes and report average rewards.

    Args:
        agent (Any): The trained agent object to be evaluated.
        evaluation_config (EvaluationConfig): Configuration for the evaluation, including environment and video settings.

    Returns:
        float: The average reward obtained by the agent across the evaluation episodes.
    """

    def extract_timestamp(agent: Any) -> str:
        """Extract the first timestamp from any of the provided paths or generate a new one."""

        timestamp_pattern = r'\d{8}\-\d{6}|\d{8}\d{6}'

        # Attempt to extract the timestamp from agent attributes
        for attr in ['video_folder', 'log_dir', 'save_path']:
            if hasattr(agent, attr):
                value = getattr(agent, attr, None)
                if value:
                    match = re.search(timestamp_pattern, value)
                    if match:
                        return match.group(0)

        # Fallback to a new timestamp if no match is found
        return datetime.now().strftime('%Y%m%d%H%M%S')

    match evaluation_config.agent_type:
        case 'dqn':
            if not isinstance(agent, DQN):
                raise TypeError(f'Expected a DQN agent for evaluation, but got {type(agent).__name__}.')
        case 'ppo':
            if not isinstance(agent, PPO):
                raise TypeError(f'Expected a PPO agent for evaluation, but got {type(agent).__name__}.')
        case _:
            raise ValueError(f"Unknown agent type: {evaluation_config.agent_type}")

    # Generate seed if it's not specified
    if evaluation_config.seed is None:
        evaluation_config.seed = int(time.time() * 1000) % (2 ** 32 - 1)

    # Set seeds for reproducibility
    random.seed(evaluation_config.seed)
    np.random.seed(evaluation_config.seed)
    torch.manual_seed(evaluation_config.seed)

    # Generate default video folder if not provided
    if evaluation_config.video_folder is None:
        timestamp = extract_timestamp(agent)
        evaluation_config.video_folder = os.path.join(
            'recordings', evaluation_config.env_id, evaluation_config.agent_type, timestamp, 'evaluation'
        )

    # Create the evaluation environment
    env = gym.make(evaluation_config.env_id, render_mode='rgb_array')

    # Enable video recording if specified
    if evaluation_config.enable_recording:
        env = RecordVideo(
            env,
            video_folder=evaluation_config.video_folder,
            name_prefix=evaluation_config.name_prefix,
            episode_trigger=lambda x: x % evaluation_config.record_every == 0,
            disable_logger=True
        )

    returns = []
    # Set the initial observation with the evaluation seed
    observation, _ = env.reset(seed=evaluation_config.seed)
    for episode in range(1, evaluation_config.n_episodes + 1):
        episode_return = 0
        done = False
        while not done:
            action = agent.select_action(
                observation, deterministic=evaluation_config.deterministic)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        observation, _ = env.reset()

    # Compute the average return across all episodes
    average_return = sum(returns) / evaluation_config.n_episodes
    res_str = f'Average return over {evaluation_config.n_episodes} episodes: {average_return:.3f}'
    print(res_str.center(60))

    return average_return


def train_agent(agent_config: Any, training_config: TrainingConfig) -> Any:
    """
    Train an RL agent based on the provided configuration.

    Args:
        agent_config (Any): Configuration object specific to the chosen RL agent (e.g., DQNConfig, PPOConfig).
        training_config (TrainingConfig): The training configuration.

    Returns:
        Any: The trained agent object.
    """

    # Generate seed if it's not specified
    if training_config.seed is None:
        training_config.seed = int(time.time() * 1000) % (2 ** 32 - 1)

    # Set seeds for reproducibility
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    # Create the training environment
    env = gym.make(training_config.env_id, render_mode='rgb_array')

    # Enable video recording if specified
    if training_config.enable_recording:
        env = RecordVideo(
            env,
            video_folder=training_config.video_folder,
            name_prefix=training_config.name_prefix,
            episode_trigger=lambda x: x % training_config.record_every == 0,
            disable_logger=True
        )

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
            raise ValueError(f"Unknown agent type: {training_config.agent_type}")

    # Train the agent
    agent.train(training_config)

    # Close the environment after training
    env.close()

    # Return the trained agent for evaluation
    return agent


def main():
    """Main function to set up configurations, train an agent, and evaluate its performance."""

    # Set up the general experiment configuration
    env_id = 'LunarLander-v2'                                              # The ID of the gym environment
    agent_type = 'dqn'  # or 'ppo'                                         # The type of RL agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # The computational device (CPU or GPU)

    box_width = 60

    # Set up the agent's configuration
    match agent_type:
        case 'dqn':
            agent_config = DQNConfig()
        case 'ppo':
            agent_config = PPOConfig()

    # Set up the training configuration
    training_seed = 69420          # Seed for training reproducibility
    training_timesteps = int(1e7)  # Number of timesteps for training

    training_config = TrainingConfig(
        env_id=env_id,
        agent_type=agent_type,
        device=device,
        n_timesteps=training_timesteps,
        seed=training_seed,
        enable_recording=True,
        record_every=100
    )

    # Set up the evaluation configuration
    evaluation_seed = training_seed + 37  # Different seed value for evaluation
    evaluation_episodes = 10              # Number of episodes for evaluation

    evaluation_config = EvaluationConfig(
        env_id=env_id,
        agent_type=agent_type,
        device=device,
        n_episodes=evaluation_episodes,
        seed=evaluation_seed,
        enable_recording=True,
        record_every=1
    )

    # Print a summary of the current configuration
    print('\n' + nice_box(
        width=box_width,
        contents=[(f'Deep Reinforcement Learning', 'c')]
        + [('─'*(box_width-6), 'c')]
        + [('Training settings:', '')]
        + [(f'  - {key}: {value}', '') for key, value in vars(training_config).items()]
        + [('Agent hyperparameters:', '')]
        + [(f'  - {key}: {value}', '') for key, value in vars(agent_config).items()],
        thick=True) + '\n'
    )

    print('Training'.center(box_width) + '\n' + ('─'*(box_width-6)).center(box_width))

    # Train the agent based on the training configuration
    # trained_agent = train_agent(agent_config, training_config)

    print('\n' + 'Evaluation'.center(box_width) + '\n' + ('─'*(box_width-6)).center(box_width))

    # Evaluate the agent based on the evaluation configuration
    # evaluate_agent(trained_agent, evaluation_config)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
