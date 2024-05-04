from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    '''
    Configuration class for DQN agent settings.

    Attributes:
        batch_size (int): Size of the batch used in the learning process.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Initial value for the epsilon in the epsilon-greedy policy.
        eps_min (float): Minimum value of epsilon.
        eps_decay (float): Decay rate of epsilon per episode.
        gamma (float): Discount factor for future rewards.
        tau (float): Interpolation parameter for updating the target network.
        memory_size (int): Size of the memory buffer.
        score_to_beat (int): Score threshold after which the environment is considered solved.
        scores_window_size (int): The rolling window size for averaging scores.
        max_timesteps_per_episode (int): Maximum number of timesteps per episode.
        model_save_frequency (int): Frequency of saving the model (in terms of episodes).
        log_dir (str): Directory for storing logs.
        save_path (str): Path to save the model.
    '''
    batch_size: int = 64
    learning_rate: float = 5e-4
    epsilon: float = 1.0
    eps_min: float = 0.01
    eps_decay: float = 0.995
    gamma: float = 0.99
    tau: float = 1e-3
    memory_size: int = int(1e5)
    score_to_beat: int = 200
    scores_window_size: int = 100
    max_timesteps_per_episode: int = 1000
    model_save_frequency: int = 10
    log_dir: str = \
        field(default_factory=lambda: f'runs/dqn/{datetime.now().strftime("%Y%m%d%H%M%S")}')
    save_path: str = \
        field(default_factory=lambda: f'saved_models/dqn/model_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth')


@ dataclass
class PPOConfig:
    max_timesteps_per_batch: int = 4000
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.99
    n_epochs: int = 4
    n_minibatches: int = 4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 1e-4
    score_to_beat: int = 200
    scores_window_size: int = 100
    max_timesteps_per_episode: int = 1000
    model_save_frequency: int = 10
    log_dir: str = \
        field(default_factory=lambda: f'runs/ppo/{datetime.now().strftime("%Y%m%d%H%M%S")}')
    save_path: str = \
        field(default_factory=lambda: f'saved_models/ppo/model_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth')
