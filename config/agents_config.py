from dataclasses import dataclass


@dataclass
class DQNConfig:
    """
    Configuration class for DQN agent settings.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        buffer_size (int): Size of the memory buffer.
        learning_starts (int): How many steps before learning starts
        minibatch_size (int): Size of the minibatch used in the learning process.
        tau (float): Interpolation parameter for updating the target network.
        gamma (float): Discount factor for future rewards.
        learn_frequency (int): Number of steps between each learning.
        epsilon (float): Initial value for the epsilon in the epsilon-greedy policy.
        eps_final (float): Final value of epsilon.
        eps_decay (float): Decay rate of epsilon per episode.
    """

    learning_rate: float = 5e-4
    buffer_size: int = int(1e5)
    learning_starts: int = 1000
    minibatch_size: int = 64
    tau: float = 1e-3
    gamma: float = 0.99
    learn_frequency: int = 1
    epsilon: float = 1.0
    eps_final: float = 0.01
    eps_decay: float = 0.995


@ dataclass
class PPOConfig:
    """
    Configuration class for PPO agent settings.

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
