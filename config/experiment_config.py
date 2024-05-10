import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class BaseExperimentConfig:
    """
    Base configuration class for reinforcement learning experiments, shared by both training and evaluation settings.

    Attributes:
        env_id (str): The ID of the gym environment to be used. This is a required field.
        agent_type (str): The type of agent to use (e.g., 'dqn', 'ppo'). This is a required field.
        device (torch.device): The computation device, either CPU or GPU (CUDA), based on system availability.
        seed (Optional[int]): A global random seed for reproducibility. Defaults to `None`, which generates a seed.
        kwargs (Dict[str, Any]): Additional keyword arguments for gym environment customization.
    """

    env_id: str
    agent_type: str
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig(BaseExperimentConfig):
    """
    Configuration class for training reinforcement learning agents.

    Attributes:
        env_id (str): The ID of the gym environment to be used. This is a required field.
        agent_type (str): The type of agent to use (e.g., 'dqn', 'ppo'). This is a required field.
        device (torch.device): The computation device, either CPU or GPU (CUDA), based on system availability.
        kwargs (Dict[str, Any]): Additional keyword arguments for gym environment customization.
        seed (Optional[int]): A global random seed for reproducibility. Defaults to `None`, which generates a seed.
        n_timesteps (int): The total number of timesteps for training. Must be positive.
        evaluate_every (int): ...
        score_threshold (int): Defines when the environment is considered solved.
        window_size (int): The window size for calculating rolling average scores and average episode length. Must be positive.
        max_timesteps_per_episode (int): Maximum timesteps per episode. Must be positive.
        print_every (int): How often to print scores. Non-positive values disable printing.
        enable_logging (bool): Whether logging is enabled or not. Defaults to `True`. 
        log_dir (Optional[str]): Directory for storing TensorBoard logs. If not provided, a default is generated.
        checkpoint_frequency (Optional[int]): Specifies how frequently to save models. If `None`, automatic generation is triggered; non-positive values disable saving.
        save_path (Optional[str]): Path to save the model. If not provided, a default path is generated.
    """

    n_timesteps: int = 10000
    evaluate_every: int = 10000
    score_threshold: int = 200
    window_size: int = 100
    max_timesteps_per_episode: int = 1000
    print_every: int = 10000
    enable_logging: bool = True
    log_dir: Optional[str] = None
    checkpoint_frequency: Optional[int] = None
    save_path: Optional[str] = None

    def __post_init__(self):
        # Validate that `env_id` and `agent_type` are set correctly
        if not self.env_id:
            raise ValueError('Environment ID (env_id) must be specified.')
        if not self.agent_type:
            raise ValueError('Agent type (agent_type) must be specified.')
        if self.agent_type not in ['dqn', 'ppo']:
            raise ValueError(f'Invalid agent type: {self.agent_type}. Must be one of "dqn" or "ppo".')

        # Validate numeric fields
        if self.n_timesteps <= 0:
            raise ValueError('Number of timesteps (n_timesteps) must be positive.')
        if self.max_timesteps_per_episode <= 0:
            raise ValueError('Max timesteps per episode (max_timesteps_per_episode) must be positive.')
        if self.window_size <= 0:
            raise ValueError('Window size (window_size) must be positive.')

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Generate default log directory if not provided
        if self.log_dir is None:
            self.log_dir = os.path.join('runs', self.env_id, self.agent_type, timestamp)

        # Generate default model saving path if not provided
        if self.save_path is None:
            self.save_path = os.path.join('saved_models', self.env_id, self.agent_type, f'model_{timestamp}.pth')
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Automatically determine the checkpoint frequency if it's None
        if self.checkpoint_frequency is None:
            match self.agent_type:
                case 'dqn':
                    self.checkpoint_frequency = 100
                case 'ppo':
                    self.checkpoint_frequency = 10
                case _:
                    raise ValueError(f'Unknown agent type: {self.agent_type}')


@dataclass
class EvaluationConfig(BaseExperimentConfig):
    """
    Configuration class for evaluation settings.

    Attributes:
        env_id (str): The ID of the gym environment to be used. This is a required field.
        agent_type (str): The type of agent to use (e.g., 'dqn', 'ppo'). This is a required field.
        device (torch.device): The computation device, either CPU or GPU (CUDA), based on system availability.
        seed (Optional[int]): A global random seed for reproducibility. Defaults to `None`, which generates a seed.
        kwargs (Dict[str, Any]): Additional keyword arguments for gym environment customization.
        n_episodes (int): Number of episodes to run during evaluation. Must be positive.
        deterministic (bool): Whether to select actions deterministically or allow exploration. Defaults to True.
        record_every (int): Specifies the interval at which episodes should be recorded. Non-positive values disable recording.
        video_folder (Optional[str]): The directory to store recorded videos. If not provided, a default is generated.
        name_prefix (Optional[str]): A prefix for naming video files to differentiate experiments. Defaults to an empty string.
    """

    n_episodes: int = 10
    deterministic: bool = True
    record_every: int = 10
    video_folder: Optional[str] = None
    name_prefix: Optional[str] = None

    def __post_init__(self):
        # Validate that `env_id` and `agent_type` are set correctly
        if not self.env_id:
            raise ValueError('Environment ID (env_id) must be specified.')
        if not self.agent_type:
            raise ValueError('Agent type (agent_type) must be specified.')
        if self.agent_type not in ['dqn', 'ppo']:
            raise ValueError(f'Invalid agent type: {self.agent_type}. Must be one of "dqn" or "ppo".')

        # Validate numeric fields
        if self.n_episodes <= 0:
            raise ValueError('Number of episodes (n_episodes) must be positive.')
