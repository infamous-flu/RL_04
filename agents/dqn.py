import os
import re
import time
import random
import warnings
from datetime import datetime
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from config.agents_config import DQNConfig
from config.experiment_config import TrainingConfig, EvaluationConfig

# Defining the Experience namedtuple
Experience = namedtuple('Experience', ('observation', 'action', 'next_observation', 'reward', 'done'))


class ReplayMemory:
    """
    A class used to store replay experiences for a DQN agent.

    Attributes:
        capacity (int): The maximum number of experiences the memory can hold.
        memory (deque): A deque that stores the experiences up to the specified capacity.
    """

    def __init__(self, capacity: int):
        """
        Initializes the ReplayMemory with a specified capacity.

        Args:
            capacity (int): The maximum number of experiences to store.
        """

        self.memory = deque([], maxlen=capacity)

    def push(self, observation: NDArray[np.float32], action: int,
             next_observation: NDArray[np.float32], reward: float, done: bool):
        """
        Adds an experience to the memory. Each experience is constructed from the given parameters
        and stored as a tuple.

        Args:
            observation (NDArray[np.float32]): The current environment observation.
            action (int): The action taken based on the observation.
            next_observation (NDArray[np.float32]): The subsequent environment observation following the action.
            reward (float): The reward received after taking the action.
            done (bool): A boolean indicating whether the episode has ended.
        """

        self.memory.append(Experience(observation, action, next_observation, reward, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Randomly samples a batch of experiences from the memory.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            List[Experience]: A list of randomly sampled experiences.
        """

        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current size of the internal memory.

        Returns:
            int: The number of experiences stored in the memory.
        """

        return len(self.memory)


class QNetwork(nn.Module):
    """
    A neural network class for approximating Q-values in a reinforcement learning context.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer, outputs Q-values for each action.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initializes the QNetwork with specified dimensions for input and output layers.

        Args:
            n_observations (int): Number of observation inputs expected by the network.
            n_actions (int): Number of possible actions the agent can take.
        """

        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)             # Second fully connected layer
        self.fc3 = nn.Linear(128, n_actions)       # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor containing the batch of observations.

        Returns:
            torch.Tensor: The tensor containing Q-values for each action.
        """

        x = F.relu(self.fc1(x))  # Activation function after the first layer
        x = F.relu(self.fc2(x))  # Activation function after the second layer
        return self.fc3(x)       # Output from the final layer


class DQN:
    """
    A class implementing the Deep Q-Network (DQN) reinforcement learning algorithm.

    Attributes:
        env (gym.Env): The gym environment in which the agent will operate.
        device (torch.device): The device (CPU or GPU) to perform computations.
        agent_config (DQNConfig): Configuration object containing hyperparameters for the DQN agent.
        seed (Optional[int]): Optional seed value for initialization to ensure reproducibility.
    """

    def __init__(self, env: gym.Env, device: torch.device, agent_config: DQNConfig, seed: Optional[int] = None):
        """
        Initializes the DQN agent with the environment, device, and configuration.

        Args:
            env (gym.Env): The gym environment in which the agent will operate.
            device (torch.device): The device (CPU or GPU) to perform computations.
            agent_config (DQNConfig): Configuration object containing hyperparameters for the DQN agent.
            seed (Optional[int]): Optional seed value for initialization to ensure reproducibility.
        """

        self.env = env                                        # The gym environment where the agent will interact
        self.device = device                                  # The computation device (CPU or GPU)
        self.agent_config = agent_config                      # Configuration for agent hyperparameters
        if seed is not None:
            self._set_seed(seed)                              # Set the seed in various components
        self.n_observations = env.observation_space.shape[0]  # Number of features in the observation space
        self.n_actions = env.action_space.n                   # Number of possible actions
        self._init_hyperparameters()                          # Initialize the hyperparameters
        self._init_networks()                                 # Set up the neural network architecture
        self.memory = ReplayMemory(self.buffer_size)          # Initialize the replay memory

    def learn(self, training_config: TrainingConfig, evaluation_config: Optional[EvaluationConfig] = None):
        """
        Train the DQN agent using the given training configuration and optionally evaluate during training. 
        Raises errors if evaluation is required by the training configuration but not provided, or if there
        are mismatches in environment IDs or agent types between training and evaluation configurations.

        Args:
            training_config (TrainingConfig): The configuration containing training parameters.
            evaluation_config (Optional[EvaluationConfig]): Optional configuration for evaluation parameters.
        """

        if training_config.evaluate_every > 0 and evaluation_config is None:
            raise ValueError('Evaluation configuration is required because `evaluate_every` is set.')

        if training_config.env_id != evaluation_config.env_id:
            raise ValueError('Training and evaluation environment IDs do not match.')

        if training_config.agent_type != evaluation_config.agent_type:
            raise ValueError('Training and evaluation agent types do not match.')

        self.training_config = training_config                    # Configuration for training parameters
        if evaluation_config:
            self.evaluation_config = evaluation_config            # Configuration for evaluation parameters
        self._init_training_parameters()                          # Initialize training parameters
        self._set_seed(self.training_config.seed)                 # Set the seed in various components
        self._init_writer()                                       # Prepare the TensorBoard writer for logging

        self.t = 0                                                # Initialize global timestep counter
        self.episode_i = 0                                        # Initialize episode counter
        self.returns_window = deque([], maxlen=self.window_size)  # Used for tracking the average returns
        self.lengths_window = deque([], maxlen=self.window_size)  # Used for tracking the average episode lengths

        while self.t < self.n_timesteps:
            self.rollout()  # Perform one episode of interaction with the environment

        # Final save and close the logger
        if self.checkpoint_frequency > 0:
            self.save_model(self.save_path)
        if self.writer is not None:
            self.writer.close()

    def rollout(self):
        """
        Executes one episode of interaction with the environment, collecting experiences and updating the agent's
        policy. The episode is constrained by a maximum number of timesteps. During the episode, actions are
        selected based on the current policy, experiences are stored, and the agent is trained periodically
        according to specified intervals.

        This method also handles logging, model checkpointing, and evaluation. It updates exploration rate and
        records performance metrics like episode return and length. It checks and logs when the agent reaches
        the predefined  performance threshold.
        """

        episode_return = 0                 # Initialize return for the episode
        observation, _ = self.env.reset()  # Reset the environment and get the initial observation

        for episode_t in range(self.max_timesteps_per_episode):                         # Iterate over allowed timesteps
            self.t += 1                                                                 # Increment the global timestep counter
            action = self.select_action(observation)                                    # Select an action based on the current observation
            next_observation, reward, terminated, truncated, _ = self.env.step(action)  # Take the action and observe the outcome
            done = terminated or truncated                                              # Determine if the episode has ended

            episode_return += reward                                                    # Update the episode return
            self.memory.push(observation, action, next_observation, reward, done)       # Store teh experience in memory

            # Train the Q-Network at specified intervals, after an initial delay
            if self.t >= self.learning_starts and self.t % self.learn_every == 0:
                self.train()

            # Save the model at specified intervals
            if self.checkpoint_frequency > 0 and self.t % self.checkpoint_frequency == 0:
                self.save_model(self.save_path)

            # Evaluate the agent periodically and log the results if enabled
            if self.evaluate_every > 0 and self.t % self.evaluate_every == 0:
                average_evaluation_return = self.evaluate(self.evaluation_config)
                if self.enable_logging:
                    self.writer.add_scalar('Evaluation/AverageEvaluationReturn', average_evaluation_return, self.t)
            else:
                average_evaluation_return = None

            # Print progress periodically
            if self.print_every > 0 and self.t % self.print_every == 0:
                res1 = f'Timestep {self.t:>7}'
                res2 = f'Training Return: {np.mean(self.returns_window):.3f}'
                res3 = ''
                if average_evaluation_return is not None:
                    res3 = f'Evaluation Return: {average_evaluation_return:.3f}'
                print('    ' + res1.ljust(16) + '        ' + res2.ljust(25) + '        ' + res3.ljust(27))

            if done:
                break  # If the episode is finished, exit the loop

            observation = next_observation  # Update the observation for the next timestep

        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_final)  # Update the exploration rate

        self.episode_i += 1                         # Increment the episode counter
        self.returns_window.append(episode_return)  # Record the episode return
        self.lengths_window.append(episode_t + 1)   # Record the episode length

        if self.enable_logging:
            if self.episode_i >= self.window_size:
                self.writer.add_scalar(
                    'Training/AverageTrainingReturn', np.mean(self.returns_window), self.t)       # Log average training return
                self.writer.add_scalar(
                    'Training/AverageEpisodeLength', np.mean(self.lengths_window), self.t)        # Log the average episode length
            self.writer.add_scalar('Episodic/EpisodeReturn', episode_return, self.episode_i)  # Log the return of the current episode
            self.writer.add_scalar('Episodic/EpisodeLength', episode_t + 1, self.episode_i)   # Log the length of the current episode
            self.writer.add_scalar('DQN/ExplorationRate', self.epsilon, self.t)               # Log the exploration rate

    def train(self):
        """
        Performs a learning update using the Deep Q-Learning (DQN) algorithm. This method handles the entire
        process of fetching a batch of experiences, processing them through the network, calculating the loss,
        and updating the network parameters. It ensures that updates occur only when enough experiences are
        available, and it periodically synchronizes the target network with the policy network.

        The training step uses the Bellman equation to update the policy network's Q-values towards their target
        Q-values, which are estimated using the rewards and the next state's maximum Q-value as predicted by the
        target network.
        """

        # Ensure there are enough samples in memory to form a complete minibatch
        if len(self.memory) < self.minibatch_size:
            return

        # Sample a minibatch of experiences from memory
        experiences = self.memory.sample(self.minibatch_size)
        observations, actions, next_observations, rewards, dones = zip(*experiences)

        # Convert data into tensors suitable for network processing
        observations, actions, next_observations, rewards, dones \
            = self._prepare_tensors(observations, actions, next_observations, rewards, dones)

        # Compute the Q-values for the actions taken, as predicted by the policy network
        current_q_values = self.policy_net(observations).gather(1, actions.unsqueeze(1)).squeeze(-1)

        # Compute the next state's maximum Q-values from the target network, used for computing the Bellman update
        with torch.no_grad():
            max_q_values, _ = self.target_net(next_observations).max(dim=1)

        # Compute the target Q-values using the reward and the discounted max Q-values from the target network
        target_q_values = rewards + self.gamma * max_q_values * (1 - dones)

        # Compute the loss as the mean squared error between the current and target Q-values
        loss = self.MSELoss(current_q_values, target_q_values)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target network with weights from the policy network
        self.update_target_network()

        # Log training loss if logging is enabled
        if self.enable_logging:
            self.writer.add_scalar('DQN/Loss', loss, self.t)  # Log the loss

    def select_action(self, observation: NDArray[np.float32], deterministic: bool = False) -> int:
        """
        Selects an action based on the current observation using an epsilon-greedy policy. The method chooses
        the action with the highest Q-value from the policy network if acting deterministically or under the
        conditions of exploitation. Otherwise, it randomly selects an action with a probability defined by the
        epsilon value, promoting exploration.

        Args:
            observation (NDArray[np.float32]): The current state observation from the environment.
            deterministic (bool): If True, always selects the action with the highest Q-value (exploitation). \
                                  If False, occasionally selects a random action based on epsilon (exploration).

        Returns:
            int: The action selected by the agent, either deterministically or stochastically.
        """

        if np.random.random() > self.epsilon or deterministic:
            # Prepare the observation: convert to tensor, add batch dimension, and transfer to the device
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Compute Q-values for all actions using the policy network
            with torch.no_grad():
                q_values = self.policy_net(observation).squeeze()
            # Select the action with the maximum Q-value
            action = torch.argmax(q_values).item()
        else:
            # Randomly select an action from the available action space
            action = self.env.action_space.sample()

        return action

    def update_target_network(self):
        """
        Updates the target network by partially copying the weights from the policy network, implementing a soft
        update. The soft update helps to stabilize training by blending weights of the policy network with the 
        previous weights of the target network, controlled by a factor tau.

        This method uses the weighted sum of the corresponding weights from both networks, where tau is the
        blend ratio.
        """

        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def evaluate(self, evaluation_config: EvaluationConfig) -> float:
        """
        Evaluates the DQN agent by running it through a series of episodes in an evaluation environment,
        according to the provided configuration. It temporarily alters random states for reproducibility,
        records evaluation episodes if configured, and safely restores states afterwards.

        Args:
            evaluation_config (EvaluationConfig): Configuration object containing evaluation parameters.

        Returns:
            float: Average return across all evaluation episodes, calculated as the sum of rewards obtained per \
            episode divided by the number of episodes.
        """

        def extract_timestamp() -> str:
            """Extract the first timestamp from any of the provided paths or generate a new one."""
            timestamp_pattern = r'\d{8}\-\d{6}|\d{8}\d{6}'
            for attr in ['log_dir', 'save_path']:
                if hasattr(self, attr):
                    value = getattr(self, attr, None)
                    if value:
                        match = re.search(timestamp_pattern, value)
                        if match:
                            return match.group(0)
            return datetime.now().strftime('%Y%m%d%H%M%S')

        # Save the current random states to avoid interference
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()

        try:
            # Generate seed if it's not specified
            if evaluation_config.seed is None:
                evaluation_config.seed = int(time.time() * 1000) % (2 ** 32 - 1)

            # Create dummy timestep if needed:
            if not hasattr(self, 't'):
                self.t = 0

            # Calculate the initial seed for this evaluation
            eval_seed = evaluation_config.seed + self.t // 10000

            # Set seeds for reproducibility
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)

            # Generate default video folder if not provided
            if evaluation_config.video_folder is None:
                timestamp = extract_timestamp()
                video_folder = os.path.join('recordings', evaluation_config.env_id, 'dqn', timestamp)
            else:
                video_folder = evaluation_config.video_folder

            # Generate default name prefix if not provided
            if evaluation_config.name_prefix is None:
                name_prefix = f'timestep-{self.t:07d}'
            else:
                name_prefix = evaluation_config.name_prefix

            # Create the evaluation environment
            eval_env = gym.make(evaluation_config.env_id, render_mode='rgb_array', **evaluation_config.kwargs)

            # Enable video recording if specified
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                if evaluation_config.record_every > 0:
                    eval_env = RecordVideo(
                        eval_env,
                        video_folder=video_folder,
                        name_prefix=name_prefix,
                        episode_trigger=lambda x: x % evaluation_config.record_every == 0,
                        disable_logger=True
                    )

            returns = []

            # Evaluate the agent over a specified number of episodes
            for episode in range(evaluation_config.n_episodes):
                episode_return = 0
                observation, _ = eval_env.reset(seed=eval_seed)
                terminated, truncated = False, False
                for _ in range(evaluation_config.max_timesteps_per_episode):
                    action = self.select_action(
                        observation, deterministic=evaluation_config.deterministic
                    )
                    observation, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_return += reward
                    if terminated or truncated:
                        break
                returns.append(episode_return)
                eval_seed += 1  # Increment the seed for the next episode

            eval_env.close()

        finally:
            # Restore the original random states to ensure no training impact
            random.setstate(random_state)
            np.random.set_state(np_random_state)
            torch.set_rng_state(torch_random_state)

        # Compute the average return across all episodes
        average_evaluation_return = sum(returns) / evaluation_config.n_episodes

        return average_evaluation_return

    def save_model(self, file_path: str, save_optimizer: bool = True):
        """
        Saves the policy network's model parameters to the specified file path. Optionally, it also saves
        the optimizer state. This function is useful for checkpointing the model during training or saving
        the final trained model.

        Args:
            file_path (str): The path to the file where the model parameters are to be saved.
            save_optimizer (bool): If True, saves the optimizer state along with the model parameters.
        """

        model_and_or_optim = {'model_state_dict': self.policy_net.state_dict()}
        if save_optimizer:
            model_and_or_optim['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(model_and_or_optim, file_path)

    def load_model(self, file_path: str, load_optimizer: bool = True):
        """
        Loads model parameters into the policy network from a specified file and also ensures the target
        network is synchronized with the policy network. Optionally, it can also load the optimizer state
        if it was saved.

        Args:
            file_path (str): The path to the file from which to load the model parameters.
            load_optimizer (bool): If True, also loads the optimizer state, assuming it is available.
        """

        model_and_or_optim = torch.load(file_path)
        self.policy_net.load_state_dict(model_and_or_optim['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if load_optimizer and 'optimizer_state_dict' in model_and_or_optim:
            self.optimizer.load_state_dict(model_and_or_optim['optimizer_state_dict'])

    def _prepare_tensors(self, observations: Tuple[NDArray[np.float32], ...], actions: Tuple[int, ...],
                         next_observations: Tuple[NDArray[np.float32], ...], rewards: Tuple[float, ...],
                         dones: Tuple[bool, ...]):
        """
        Converts lists of observations, actions, log probabilities, and advantages into tensors
        and transfers them to the specified computing device. This step is critical for batching
        and processing the data efficiently in neural network operations.

        Args:
            observations (List[NDArray[np.float32]]): List of environment observations, each as a NumPy array.
            actions (List[int]): List of actions taken by the agent, each an integer representing a discrete action.
            log_probs (List[torch.Tensor]): List of log probabilities of the actions taken, each a single-element tensor.
            advantages (List[torch.Tensor]): List of advantage estimates used for updating the policy, each a single-element tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors of observations, actions, log \
                probabilities,  and advantages, all formatted and moved to the appropriate device for training.
        """

        observations = torch.tensor(np.stack(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(actions), dtype=torch.int64).to(self.device)
        next_observations = torch.tensor(np.stack(next_observations), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.stack(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.stack(dones), dtype=torch.float32).to(self.device)
        return observations, actions, next_observations, rewards, dones

    def _set_seed(self, seed: int):
        """Set seeds for reproducibility in various components."""

        # Seed the environment
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        # Seed Python, NumPy, and PyTorch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    def _init_hyperparameters(self):
        """
        Initializes hyperparameters from the configuration.
        """

        self.learning_rate = self.agent_config.learning_rate
        self.buffer_size = self.agent_config.buffer_size
        self.learning_starts = self.agent_config.learning_starts
        self.minibatch_size = self.agent_config.minibatch_size
        self.tau = self.agent_config.tau
        self.gamma = self.agent_config.gamma
        self.learn_every = self.agent_config.learn_every
        self.epsilon = self.agent_config.epsilon
        self.eps_final = self.agent_config.eps_final
        self.eps_decay = self.agent_config.eps_decay

    def _init_training_parameters(self):
        """
        Initializes training parameters from the configuration.
        """

        self.env_id = self.training_config.env_id
        self.n_timesteps = self.training_config.n_timesteps
        self.evaluate_every = self.training_config.evaluate_every
        self.score_threshold = self.training_config.score_threshold
        self.window_size = self.training_config.window_size
        self.max_timesteps_per_episode = self.training_config.max_timesteps_per_episode
        self.print_every = self.training_config.print_every
        self.enable_logging = self.training_config.enable_logging
        self.log_dir = self.training_config.log_dir
        self.checkpoint_frequency = self.training_config.checkpoint_frequency
        self.save_path = self.training_config.save_path

    def _init_networks(self):
        """
        Initializes the policy and target neural networks and sets up the optimizer and loss function.
        """

        self.policy_net = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.MSELoss = nn.MSELoss()

    def _init_writer(self):
        """
        Initializes Tensorboard writer for logging if logging is enabled.
        Additionally, logs all agent hyperparameters from the agent's configuration.
        """

        if self.enable_logging:
            self.writer = SummaryWriter(self.log_dir)
            agent_hyperparameters = vars(self.agent_config)
            hyperparameters_str = '\n'.join(
                [f'{key}: {value}' for key, value in agent_hyperparameters.items()])
            self.writer.add_text('DQN/Hyperparameters', hyperparameters_str)
        else:
            self.writer = None
