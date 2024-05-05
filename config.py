from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    """
    Configuration class for DQN agent settings.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        buffer_size (int): Size of the memory buffer.
        learning_starts (int): How many steps before learning starts
        batch_size (int): Size of the batch used in the learning process.
        tau (float): Interpolation parameter for updating the target network.
        gamma (float): Discount factor for future rewards.
        learn_frequency (int): Number of steps between each learning.
        target_update_interval (int): Number of steps between target network updates.
        epsilon (float): Initial value for the epsilon in the epsilon-greedy policy.
        eps_final (float): Final value of epsilon.
        eps_decay (float): Decay rate of epsilon per episode.
        score_threshold (int): Score threshold after which the environment is considered solved.
        std_deviation_factor (float): The number of standard deviations to subtract from the mean score.
        scores_window_size (int): The rolling window size for averaging scores.
        max_timesteps_per_episode (int): Maximum number of timesteps per episode.
        model_save_frequency (int): Frequency of saving the model (in terms of episodes).
        print_every (int): Frequency of printing the average score.
        enable_logging (bool): Flag to enable or disable logging.
        log_dir (str): Directory for storing logs.
        save_path (str): Path to save the model.
    """
    learning_rate: float = 5e-4
    buffer_size: int = int(1e6)
    learning_starts: int = 100
    batch_size: int = 64
    tau: float = 1e-3
    gamma: float = 0.99
    learn_frequency: int = 1
    epsilon: float = 1.0
    eps_final: float = 0.01
    eps_decay: float = 0.995
    score_threshold: int = 200
    std_deviation_factor: float = 1.5
    scores_window_size: int = 100
    max_timesteps_per_episode: int = 1000
    model_save_frequency: int = 10
    print_every: int = int(1e5)
    enable_logging: bool = True
    log_dir: str = \
        field(default_factory=lambda: f'runs/dqn/{datetime.now().strftime("%Y%m%d%H%M%S")}')
    save_path: str = \
        field(default_factory=lambda: f'saved_models/dqn/model_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth')


@ dataclass
class PPOConfig:
    """
    Configuration class for DQN agent settings.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        max_timesteps_per_batch (int): Maximum number of timesteps per batch.
        n_minibatches (int): Number of minibatches to split the batch into for training.
        n_epochs (int): Number of epochs to train over a single batch.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        clip_range (float): Clipping parameter for PPO to limit the policy update step.
        normalize_advantage (bool): Whether to normalize advantage or not.
        value_coef (float): Coefficient for the value loss in the total loss calculation.
        entropy_coef (float): Coefficient for the entropy bonus in the total loss calculation.
        score_threshold (int): Score threshold after which the environment is considered solved.
        std_deviation_factor (float): The number of standard deviations to subtract from the mean score.
        scores_window_size (int): The rolling window size for averaging scores.
        max_timesteps_per_episode (int): Maximum number of timesteps per episode.
        model_save_frequency (int): Frequency of saving the model (in terms of batch iterations).
        print_every (int): Frequency of printing the average score.
        enable_logging (bool): Flag to enable or disable logging.
        log_dir (str): Directory for storing logs.
        save_path (str): Path to save the model.
    """
    learning_rate: float = 5e-4
    max_timesteps_per_batch: int = 4000
    n_minibatches: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    normalize_advantage: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    score_threshold: int = 200
    std_deviation_factor: float = 1.5
    scores_window_size: int = 100
    max_timesteps_per_episode: int = 1000
    model_save_frequency: int = 10
    print_every: int = int(1e5)
    enable_logging: bool = True
    log_dir: str = \
        field(default_factory=lambda: f'runs/ppo/{datetime.now().strftime("%Y%m%d%H%M%S")}')
    save_path: str = \
        field(default_factory=lambda: f'saved_models/ppo/model_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth')
