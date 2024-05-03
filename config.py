from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class DQNConfig:
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
