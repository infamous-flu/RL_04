import re
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Optional

from agents.dqn import DQN
from agents.ppo import PPO


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
        print_every (int): How often to print scores.
        enable_logging (bool): Whether logging is enabled or not.
        log_dir (str): Directory for storing Tensorboard logs.
        checkpoint_frequency (int): How frequently to save models.
        save_path (str): Path to save the model.
    """

    env_id: str
    agent_type: str
    n_timesteps: int = int(1e7)
    seed: Optional[int] = None
    enable_recording: bool = True
    record_every: int = 100
    video_folder = ''
    name_prefix: str = ''
    score_threshold: int = 200
    scores_window_size: int = 100
    max_timesteps_per_episode: int = 1000
    print_every: int = 10000
    enable_logging: bool = True
    log_dir: str = ''
    checkpoint_frequency: int = 20
    save_path: str = ''

    def __post_init__(self):
        current_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        if not self.video_folder:
            self.video_folder = f'recordings/{self.env_id}/{self.agent_type}/{current_timestamp}/training'
        if not self.log_dir:
            self.log_dir = f'runs/{self.env_id}/{self.agent_type}/{current_timestamp}'
        if not self.save_path:
            self.save_path = f'saved_models/{self.env_id}/{self.agent_type}/model_{current_timestamp}.pth'


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
    record_every: int = 1
    video_folder: str = ''
    name_prefix: str = ''

    def __post_init__(self):
        if isinstance(self.agent, DQN):
            agent_type = 'dqn'
        elif isinstance(self.agent, PPO):
            agent_type = 'ppo'
        else:
            raise TypeError(f'Unsupported agent type: {type(self.agent)}')

        if not self.video_folder:
            timestamp_match = None
            if hasattr(self.agent, 'save_path') and self.agent.save_path:
                timestamp_match = re.search(r'(\d{8}\d{6})', self.agent.save_path)
            timestamp = timestamp_match.group(1) if timestamp_match \
                else datetime.now().strftime('%Y%m%d%H%M%S')
            self.video_folder = f'recordings/{self.env_id}/{agent_type}/{timestamp}/evaluation'
