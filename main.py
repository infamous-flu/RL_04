import time
import random
from typing import Optional, Union

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from agents.dqn import DQN
from agents.ppo import PPO
from config.agents_config import DQNConfig, PPOConfig
from config.experiment_config import TrainingConfig, EvaluationConfig


def evaluate(config: EvaluationConfig) -> float:
    """Evaluate the agent over a given number of episodes and report average rewards."""

    # Generate seed if it's not specified
    if config.seed is None:
        config.seed = int(time.time() * 1000) % (2 ** 32 - 1)

    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    gym.utils.seeding.seed(config.seed)

    # Create the evaluation environment
    env = gym.make(config.env_id, render_mode='human')

    # Enable video recording if specified
    if config.enable_recording:
        env = RecordVideo(
            env,
            video_folder=config.video_folder,
            name_prefix=config.name_prefix,
            episode_trigger=lambda x: x % config.record_every == 0,
            disable_logger=True
        )

    # Optionally load a pre-trained model
    if config.model_path is not None:
        config.agent.load_model(config.model_path)

    returns = []
    # Set the initial observation with the evaluation seed
    observation, _ = env.reset(seed=config.seed)
    for episode in range(1, config.n_episodes + 1):
        episode_return = 0
        done = False
        while not done:
            action = config.agent.select_action(observation, deterministic=config.deterministic)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        print(f'Episode {episode}\tReturn: {episode_return:.3f}')
        returns.append(episode_return)
        observation, _ = env.reset()

    # Compute the average return across all episodes
    average_return = sum(returns) / config.n_episodes
    print(f'\nAverage return over {config.n_episodes} episodes: {average_return:.3f}')

    return average_return


def train(config: TrainingConfig):
    """Train a RL agent based on the provided configuration."""

    # Generate seed if it's not specified
    if config.seed is None:
        config.seed = int(time.time() * 1000) % (2 ** 32 - 1)

    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    gym.utils.seeding.seed(config.seed)

    # Set the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the training environment
    env = gym.make(config.env_id, render_mode='rgb_array')

    # Enable video recording if specified
    if config.enable_recording:
        env = RecordVideo(
            env,
            video_folder=config.video_folder,
            name_prefix=config.name_prefix,
            episode_trigger=lambda x: x % config.record_every == 0,
            disable_logger=True
        )

    # Initialize the agent based on the agent type
    match config.agent_type:
        case 'dqn':
            agent_config = DQNConfig()
            agent = DQN(env, device, agent_config, config.seed)
        case 'ppo':
            agent_config = PPOConfig()
            agent = PPO(env, device, agent_config, config.seed)
        case _:
            raise ValueError(f'Unknown agent type: {config.agent_type}')

    # Train the agent for a pre-defined number of timesteps
    agent.train(config.n_timesteps)

    # Close the environment after training
    env.close()

    # Return the trained agent for evaluation
    return agent


def main():
    # Set up the training configuration
    env_id = 'LunarLander-v2'      # The ID of the gym environment to be used
    agent_type = 'ppo'             # The type of agent to be trained
    training_seed = 69420          # Seed for training reproducibility
    training_timesteps = int(1e7)  # Number of timesteps for training

    training_config = TrainingConfig(
        env_id=env_id,
        agent_type=agent_type,
        n_timesteps=training_timesteps,
        seed=training_seed,
        enable_recording=True
    )

    # Train the agent based on the training configuration
    trained_agent = train(training_config)

    # Set up the evaluation configuration
    evaluation_seed = training_seed + 37  # Different seed value for evaluation
    evaluation_episodes = 10              # Number of episodes for evaluation

    evaluation_config = EvaluationConfig(
        env_id=env_id,
        agent=trained_agent,
        n_episodes=evaluation_episodes,
        seed=evaluation_seed
    )

    # Evaluate the agent based on the evaluation configuration
    evaluate(evaluation_config)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
