import random
from typing import List, Tuple
from collections import deque, namedtuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


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

    def push(self, *args: Tuple):
        """
        Adds an experience to the memory. Each experience is expected to be a tuple
        containing the observation, action, next observation, reward, and done signal.

        Args:
            *args (Tuple): A tuple representing an experience.
        """

        self.memory.append(Experience(*args))

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
        env (gym.Env): The environment in which the agent operates.
        device (torch.device): The device (CPU or GPU) on which the computations are performed.
        config (DQNConfig): Configuration parameters for setting up the DQN agent.
        seed (int): Seed for the pseudo random generators.
    """

    def __init__(self, env: gym.Env, device: torch.device, config, seed: int = None):
        """
        Initializes the DQN agent with the environment, device, and configuration.

        Args:
            env (gym.Env): The gym environment.
            device (torch.device): The device (CPU or GPU) to perform computations.
            config (DQNConfig): A dataclass containing all necessary hyperparameters.
            seed (int): Seed for the pseudo random generators.
        """

        self.env = env                                        # The gym environment where the agent will interact
        self.device = device                                  # The computation device (CPU or GPU)
        self.config = config                                  # Configuration containing all hyperparameters
        self.seed = seed                                      # Seed for the pseudo random generators
        self.n_observations = env.observation_space.shape[0]  # Number of features in the observation space
        self.n_actions = env.action_space.n                   # Number of possible actions
        if self.seed is not None:
            self._set_seed(self.seed)                         # Set the seed in various components
        self._init_hyperparameters()                          # Initialize the hyperparameters based on the configuration
        self._init_networks()                                 # Set up the neural network architecture
        self._init_writer()                                   # Prepare the TensorBoard writer for logging
        self.memory = ReplayMemory(self.buffer_size)          # Initialize the replay memory

    def train(self, n_timesteps: int):
        """
        Trains the DQN agent for a given number of timesteps.

        Args:
            n_timesteps (int): The number of timesteps to train the agent.
        """

        self.t = 0                                                      # Initialize global timestep counter
        self.episode_i = 0                                              # Initialize episode counter
        self.scores_window = deque([], maxlen=self.scores_window_size)  # Used for tracking the average score

        while self.t < n_timesteps:
            self.rollout()  # Perform one episode of interaction with the environment

            if self.is_environment_solved():
                print(f'\nEnvironment solved in {self.t} timesteps!', end='\t')
                print(f'Average Score: {np.mean(self.scores_window):.3f}')
                break

            # Save the model at specified intervals
            if self.episode_i % self.model_save_frequency == 0:
                self.save_model(self.save_path)

        # Final save and close the logger
        self.save_model(self.save_path)
        if self.writer is not None:
            self.writer.close()

    def rollout(self):
        """
        Executes one episode of interaction with the environment, collecting experience
        and training the Q network.
        """

        score = 0                          # Initialize score for the episode
        observation, _ = self.env.reset()  # Reset the environment and get the initial observation

        for episode_t in range(self.max_timesteps_per_episode):                         # Iterate over allowed timesteps
            self.t += 1                                                                 # Increment the global timestep counter
            action = self.select_action(observation)                                    # Select an action
            next_observation, reward, terminated, truncated, _ = self.env.step(action)  # Take the action
            done = terminated or truncated                                              # Determine if the episode has ended

            score += reward                                                        # Update the score
            self.memory.push(observation, action, next_observation, reward, done)  # Remember the experience

            if self.t >= self.learning_starts and self.t % self.learn_frequency == 0:
                self.learn()  # Learn using the collected experiences

            # Print progress periodically
            if self.t % self.print_every == 0:
                print(f'Timestep {self.t}\tAverage Score: {np.mean(self.scores_window):.3f}')

            if done:
                break  # If the episode is finished, exit the loop

            observation = next_observation  # Update the observation

        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_final)  # Update the exploration rate

        self.episode_i += 1               # Increment the episode counter
        self.scores_window.append(score)  # Record the score

        if self.enable_logging:
            self.writer.add_scalar('Common/EpisodeReturn', score, self.t)                       # Log the total score
            self.writer.add_scalar('Common/EpisodeLength', episode_t + 1, self.episode_i)       # Log the episode length
            self.writer.add_scalar('Common/AverageScore', np.mean(self.scores_window), self.t)  # Log the average score
            self.writer.add_scalar('DQN/ExplorationRate', self.epsilon, self.t)                 # Log epsilon

    def learn(self):
        """
        Performs a learning update using the Deep Q-Learning (DQN) algorithm.
        """

        # Check if enough samples are available in memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from memory
        experiences = self.memory.sample(self.batch_size)
        observations, actions, next_observations, rewards, dones = zip(*experiences)

        # Prepare the data by converting to PyTorch tensors for neural network processing
        observations, actions, next_observations, rewards, dones = \
            self._prepare_tensors(observations, actions, next_observations, rewards, dones)

        # Compute current Q-values from policy network for the actions taken
        current_q_values = self.policy_net(observations).gather(1, actions.unsqueeze(1)).squeeze(-1)

        # Compute target Q-values from target network for next observations
        with torch.no_grad():
            max_q_values, _ = self.target_net(next_observations).max(dim=1)

        # Compute the target Q-values using the Bellman equation
        target_q_values = rewards + self.gamma * max_q_values * (1 - dones)

        # Calculate loss using the Mean Squared Error loss function
        loss = self.MSELoss(current_q_values, target_q_values)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network parameters
        self.update_target_network()

        if self.enable_logging:
            self.writer.add_scalar('DQN/Loss', loss, self.t)  # Log the loss

    def select_action(self, observation: NDArray[np.float32], deterministic: bool = False) -> int:
        """
        Selects an action based on the current observation using an epsilon-greedy policy.

        Args:
            observation (NDArray[np.float32]): The current state observation from the environment.
            deterministic (bool): If True, the action choice is deterministic (the max value action).
                                  If False, the action is chosen stochastically with epsilon-greedy.

        Returns:
            int: The action selected by the agent.
        """

        # Decide whether to select the best action based on model prediction or sample randomly
        if np.random.random() > self.epsilon or deterministic:
            # Convert the observation to a tensor and add a batch dimension (batch size = 1)
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Model prediction: Compute Q-values for all actions, without gradient computation
            with torch.no_grad():
                q_values = self.policy_net(observation).squeeze()
            # Select the action with the highest Q-value
            action = torch.argmax(q_values).item()
        else:
            # Randomly sample an action from the action space
            action = self.env.action_space.sample()
        return action

    def calculate_return(self, episode_rewards: List[float]) -> float:
        """
        Calculates the discounted return for an episode.

        Args:
            episode_rewards (List[float]): The rewards collected during the episode.

        Returns:
            float: The total discounted return for the episode.
        """

        episode_return = 0
        # Calculate the return using the rewards obtained, applying discount factor gamma
        for reward in reversed(episode_rewards):
            episode_return = reward + self.gamma * episode_return
        return episode_return

    def update_target_network(self):
        """
        Updates the target network by partially copying the weights from the policy network.
        """

        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def is_environment_solved(self):
        """
        Check if the environment is solved.

        Returns:
            bool: True if the environment is solved, False otherwise.
        """

        if len(self.scores_window) < self.scores_window_size:
            return False  # Not enough scores for a valid evaluation

        # Calculate the mean score and standard deviation from the window
        mean_score = np.mean(self.scores_window)
        std_dev = np.std(self.scores_window, ddof=1)
        lower_bound = mean_score - std_dev

        return lower_bound > self.score_threshold

    def save_model(self, file_path: str, save_optimizer: bool = True):
        """
        Saves the policy network's model parameters to the specified file path.
        Optionally, it also saves the optimizer state.

        Args:
            file_path (str): The path to the file where the model parameters are to be saved.
            save_optimizer (bool): If True, saves the optimizer state as well. Defaults to False.
        """

        checkpoint = {
            'model_state_dict': self.policy_net.state_dict()
        }
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, file_path)

    def load_model(self, file_path: str, load_optimizer: bool = True):
        """
        oads model parameters into the policy network and copies them to the target network.
        Optionally, it also loads the optimizer state.

        Args:
            file_path (str): The path to the file from which to load the model parameters.
            load_optimizer (bool): If True, loads the optimizer state as well. Defaults to False.
        """

        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _prepare_tensors(self, observations, actions, next_observations, rewards, dones):
        """
        Converts lists of observations, actions, next_observations, rewards, and dones
        into tensors and sends them to the specified device.

        Args:
            observations: List of observations from the environment.
            actions: List of actions taken.
            next_observations: List of observations following the actions.
            rewards: List of received rewards.
            dones: List indicating whether an episode is finished.

        Returns:
            Tuple of Tensors: Prepared tensors of observations, actions, next_observations, rewards, and dones.
        """

        observations = torch.tensor(np.stack(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(actions), dtype=torch.int64).to(self.device)
        next_observations = torch.tensor(np.stack(next_observations), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.stack(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.stack(dones), dtype=torch.float32).to(self.device)
        return observations, actions, next_observations, rewards, dones

    def _set_seed(self, seed: int):
        """
        Sets the seed for reproducibility in various components.

        Args:
            seed (int): The seed value to be used.
        """

        # Seed the Gym environment
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        # Seed Python's random module
        random.seed(seed)
        # Seed NumPy's random generator
        np.random.seed(seed)
        # Seed PyTorch for reproducibility in computations
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    def _init_hyperparameters(self):
        """
        Initializes hyperparameters from the configuration.
        """

        self.learning_rate = self.config.learning_rate
        self.buffer_size = self.config.buffer_size
        self.learning_starts = self.config.learning_starts
        self.batch_size = self.config.batch_size
        self.tau = self.config.tau
        self.gamma = self.config.gamma
        self.learn_frequency = self.config.learn_frequency
        self.epsilon = self.config.epsilon
        self.eps_final = self.config.eps_final
        self.eps_decay = self.config.eps_decay
        self.score_threshold = self.config.score_threshold
        self.scores_window_size = self.config.scores_window_size
        self.max_timesteps_per_episode = self.config.max_timesteps_per_episode
        self.model_save_frequency = self.config.model_save_frequency
        self.print_every = self.config.print_every
        self.enable_logging = self.config.enable_logging
        self.log_dir = self.config.log_dir
        self.save_path = self.config.save_path

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
        """

        if self.enable_logging:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
