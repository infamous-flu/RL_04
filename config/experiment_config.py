import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TrainingConfig:
    """
    Configuration class for training settings.

    Attributes:
        env_id (str): The ID of the gym environment to be used.
        agent_type (str): The type of agent (e.g., 'dqn', 'ppo').
        n_timesteps (int): The total number of timesteps for training.
        seed (int): Global random seed for reproducibility.
        enable_recording (bool): Whether or not to record training videos.
        record_every (int): Interval for video recording episodes.
        video_folder (str): Folder to save recorded videos.
        name_prefix (str): Prefix for video recording file names.
        score_threshold (int): Defines when the environment is considered solved.
        scores_window_size (int): Window size for calculating rolling average scores.
        max_timesteps_per_episode (int): Maximum timesteps per episode.
        print_every (int): How often to print scores. Negative values disable printing.
        enable_logging (bool): Whether logging is enabled or not.
        log_dir (str): Directory for storing Tensorboard logs.
        checkpoint_frequency (int): How frequently to save models. A value of `0` will
        automatically select the frequency based on the agent type. Positive values define
        a custom interval, and negative values disable model saving.
        save_path (str): Path to save the model.
    """

    env_id: str
    agent_type: str
    n_timesteps: int = int(1e7)
    seed: Optional[int] = None
    enable_recording: bool = True
    record_every: int = 100
    video_folder: Optional[str] = None
    name_prefix: str = field(default='')
    score_threshold: int = 200
    scores_window_size: int = 100
    max_timesteps_per_episode: int = 1000
    print_every: int = 10000
    enable_logging: bool = True
    log_dir: Optional[str] = None
    checkpoint_frequency: int = 0
    save_path: Optional[str] = None

    def __post_init__(self):
        # Validate agent type
        if self.agent_type not in ['dqn', 'ppo']:
            raise ValueError(f'Unsupported agent type: {self.agent_type}.')

        # Validate numeric ranges for certain attributes
        if self.n_timesteps <= 0:
            raise ValueError('n_timesteps must be positive.')
        if self.record_every <= 0:
            raise ValueError('record_every must be positive.')
        if self.max_timesteps_per_episode <= 0:
            raise ValueError('max_timesteps_per_episode must be positive.')
        if self.score_threshold < 0:
            raise ValueError('score_threshold cannot be negative.')

        # Automatically select checkpoint frequency
        if self.checkpoint_frequency == 0:
            match self.agent_type:
                case 'dqn':
                    self.checkpoint_frequency = 100
                case 'ppo':
                    self.checkpoint_frequency = 10

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Ensure video folder is set up correctly
        if self.video_folder is None:
            self.video_folder = os.path.join('recordings', self.env_id, self.agent_type, timestamp, 'training')

        # Ensure logging directory is set up correctly
        if self.log_dir is None:
            self.log_dir = os.path.join('runs', self.env_id, self.agent_type, timestamp)

        # Ensure save path is properly defined
        if self.save_path is None:
            self.save_path = os.path.join('saved_models', self.env_id, self.agent_type, f'model_{timestamp}.pth')

        # Create the directory for saving models if it doesn't exist
        save_dir = os.path.dirname(self.save_path)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create directory {save_dir}: {e}")


@dataclass
class EvaluationConfig:
    """
    Configuration class for evaluation settings.

    Attributes:
        env_id (str): The ID of the environment to use for evaluation.
        agent (Any): The trained agent to evaluate.
        n_episodes (int): Number of episodes to run during evaluation.
        seed (Optional[int]): Random seed for reproducibility.
        deterministic (bool): Whether to select actions deterministically or allow exploration.
        model_path (Optional[str]): Path to load a pre-trained model, if available.
        enable_recording (bool): Whether to enable video recording during evaluation.
        record_every (int): Frequency at which to record episodes.
        video_folder (str): Folder path to save recorded evaluation videos.
        name_prefix (str): Prefix for the recorded video file names.
    """

    env_id: str
    agent: Any
    n_episodes: int = 10
    seed: Optional[int] = None
    deterministic: bool = True
    model_path: Optional[str] = None
    enable_recording: bool = False
    record_every: int = 2
    video_folder: Optional[str] = None
    name_prefix: str = field(default='')

    def __post_init__(self):
        from agents.dqn import DQN
        from agents.ppo import PPO

        if isinstance(self.agent, DQN):
            agent_type = 'dqn'
        elif isinstance(self.agent, PPO):
            agent_type = 'ppo'
        else:
            raise TypeError(f'Unsupported agent type: {type(self.agent)}')

        if self.video_folder is None:
            timestamp_match = None
            if hasattr(self.agent, 'save_path') and self.agent.save_path:
                timestamp_match = re.search(r'(\d{8}\d{6})', self.agent.save_path)
            timestamp = timestamp_match.group(1) if timestamp_match \
                else datetime.now().strftime('%Y%m%d%H%M%S')
            self.video_folder = os.path.join('recordings', self.env_id, agent_type, timestamp, 'evaluation')
