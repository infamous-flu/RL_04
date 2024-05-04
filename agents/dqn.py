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
    '''
    A class used to store replay experiences for a DQN agent.

    Attributes:
        capacity (int): The maximum number of experiences the memory can hold.
        memory (deque): A deque that stores the experiences up to the specified capacity.
    '''

    def __init__(self, capacity: int):
        '''
        Initializes the ReplayMemory with a specified capacity.

        Args:
            capacity (int): The maximum number of experiences to store.
        '''
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Tuple) -> None:
        '''
        Adds an experience to the memory. Each experience is expected to be a tuple
        containing the observation, action, next observation, reward, and done signal.

        Args:
            *args (Tuple): A tuple representing an experience.
        '''
        self.memory.append(Experience(*args))

    def sample(self, batch_size: int) -> List[Experience]:
        '''
        Randomly samples a batch of experiences from the memory.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            List[Experience]: A list of randomly sampled experiences.
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        '''
        Returns the current size of the internal memory.

        Returns:
            int: The number of experiences stored in the memory.
        '''
        return len(self.memory)


class QNetwork(nn.Module):
    '''
    A neural network class for approximating Q-values in a reinforcement learning context.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer, outputs Q-values for each action.
    '''

    def __init__(self, n_observations: int, n_actions: int):
        '''
        Initializes the QNetwork with specified dimensions for input and output layers.

        Args:
            n_observations (int): The number of observation inputs the network will receive.
            n_actions (int): The number of actions the network needs to provide Q-values for.
        '''
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)             # Second hidden layer
        self.fc3 = nn.Linear(128, n_actions)       # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor containing the batch of observations.

        Returns:
            torch.Tensor: The tensor containing Q-values for each action.
        '''
        x = F.relu(self.fc1(x))  # ReLU activation function applied to the first layer's output
        x = F.relu(self.fc2(x))  # ReLU activation function applied to the second layer's output
        return self.fc3(x)       # The final layer outputs the Q-values directly


class DQN:
    '''
    A class implementing the Deep Q-Network (DQN) reinforcement learning algorithm.

    Attributes:
        env (gym.Env): The environment in which the agent operates.
        device (torch.device): The device (CPU or GPU) on which the computations are performed.
        config (DQNConfig): Configuration parameters for setting up the DQN.
    '''

    def __init__(self, env: gym.Env, device: torch.device, config):
        '''
        Initializes the DQN agent with the environment, device, and configuration.

        Args:
            env (gym.Env): The gym environment.
            device (torch.device): The device (CPU or GPU) to perform computations.
            config: A dataclass containing all necessary hyperparameters.
        '''
        self.env = env  # The gym environment where the agent will interact
        self.device = device  # The computation device (CPU or GPU)
        self.config = config  # Configuration containing all hyperparameters
        self.n_observations = env.observation_space.shape[0]  # Number of features in the observation space
        self.n_actions = env.action_space.n  # Number of possible actions
        self._init_hyperparameters()  # Initialize the hyperparameters based on the configuration
        self._init_networks()  # Set up the neural network architecture
        self.writer = SummaryWriter(self.log_dir)  # Prepare the TensorBoard writer for logging
        self.memory = ReplayMemory(self.memory_size)  # Initialize the replay memory

    def learn(self, n_timesteps: int):
        '''
        Trains the agent for a given number of timesteps.

        Args:
            n_timesteps (int): The number of timesteps to train the agent.

        This method handles the training loop, including the rollout of episodes,
        updating the network, and logging. It checks for the condition to terminate
        training if the agent's performance meets the target criteria.
        '''
        self.i = 0  # Initialize episode counter
        self.t = 0  # Initialize global timestep counter
        scores_window = deque([], maxlen=self.scores_window_size)  # Used for tracking the average score

        while self.t < n_timesteps:
            score = self.rollout()  # Perform one episode of interaction with the environment
            scores_window.append(score)  # Record the score
            average_score = np.mean(scores_window)  # Calculate the moving average score

            # Check for early stopping if the environment is considered solved
            if average_score >= self.score_to_beat:
                print(f'\nEnvironment solved in {self.i - self.scores_window_size} episodes!', end='\t')
                print(f'Average Score: {average_score: .4f}')
                break

            # Logging progress periodically
            if self.i % self.scores_window_size == 0:
                print(f'Episode {self.i}\tAverage Score: {average_score:.4f}')

            # Save the model at specified intervals
            if self.i % self.model_save_frequency == 0:
                self.save_model(self.save_path)

            # Update the exploration rate
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        # Final save and close the logger
        self.save_model(self.save_path)
        self.writer.close()

    def rollout(self):
        '''
        Executes one episode of interaction with the environment, collecting
        experience and training the Q network.

        Returns:
            float: The total reward accumulated over the episode.
        '''
        score = 0  # Initialize score for the episode
        observation, _ = self.env.reset()  # Reset the environment and get the initial observation

        for episode_t in range(self.max_timesteps_per_episode):  # Iterate over allowed timesteps
            self.t += 1   # Increment the global timestep counter
            action = self.select_action(observation)  # Select an action
            next_observation, reward, terminated, truncated, _ = self.env.step(action)  # Take the action
            done = terminated or truncated  # Determine if the episode has ended

            score += reward  # Update the score
            self.memory.push(observation, action, next_observation, reward, done)  # Remember the experience

            loss = self.train()  # Train the network using the stored experiences
            if loss is not None:
                self.writer.add_scalar('Loss', loss, self.t)  # Log the loss

            if done:  # If the episode is finished, exit the loop
                break

            observation = next_observation  # Update the observation

        self.i += 1  # Increment the episode counter
        self.writer.add_scalar('Episode/Score', score, self.i)  # Log the score
        self.writer.add_scalar('Episode/Length', episode_t + 1, self.i)  # Log the episode length

        return score

    def select_action(self, observation: NDArray[np.float32]) -> int:
        '''
        Selects an action based on the current observation using an epsilon-greedy policy.

        Args:
            observation (NDArray[np.float32]): The current state observation from the environment.

        Returns:
            int: The action to be taken.
        '''
        # Decide whether to select the best action based on model prediction or sample randomly.
        if random.random() > self.epsilon:
            # Convert the observation to a tensor and add a batch dimension (batch size = 1).
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Model prediction: Compute Q-values for all actions, without gradient computation.
            with torch.no_grad():
                q_values = self.policy_net(observation).squeeze()
            # Select the action with the highest Q-value.
            action = torch.argmax(q_values).item()
        else:
            # Randomly sample an action from the action space.
            action = self.env.action_space.sample()
        return action

    def train(self):
        '''
        Trains the Q network using a batch of experiences from memory.

        Returns:
            float or None: The loss for the training step, or None if not enough samples are available.
        '''
        # Check if enough samples are available in memory
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of experiences from memory
        experiences = self.memory.sample(self.batch_size)
        observations, actions, next_observations, rewards, dones = zip(*experiences)

        # Prepare tensors from the sampled experiences
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

        return loss.item()

    def update_target_network(self):
        '''
        Updates the target network by partially copying the weights from the policy network.
        This helps stabilize the learning process.
        '''
        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def calculate_return(self, episode_rewards: List[float]):
        '''
        Calculates the discounted return for an episode.

        Args:
            episode_rewards (list of float): The rewards collected during the episode.

        Returns:
            float: The total discounted return for the episode.
        '''
        episode_return = 0
        # Calculate the return using the rewards obtained, applying discount factor gamma
        for reward in reversed(episode_rewards):
            episode_return = reward + self.gamma * episode_return
        return episode_return

    def save_model(self, file_path: str):
        '''
        Saves the policy network's model parameters to the specified file path.

        Args:
            file_path (str): The path to the file where the model parameters are to be saved.
        '''
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, file_path: str):
        '''
        Loads model parameters into the policy network and copies them to the target network.

        Args:
            file_path (str): The path to the file from which to load the model parameters.
        '''
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _prepare_tensors(self, observations, actions, next_observations, rewards, dones):
        '''
        Converts lists of numpy arrays into tensors and sends them to the specified device.

        Args:
            observations: List of observations from the environment.
            actions: List of actions taken.
            next_observations: List of observations following the actions.
            rewards: List of received rewards.
            dones: List indicating whether an episode is finished.

        Returns:
            Tuple of Tensors: Prepared tensors of observations, actions, next_observations, rewards, and dones.
        '''
        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        return observations, actions, next_observations, rewards, dones

    def _init_hyperparameters(self):
        '''
        Initializes hyperparameters from the configuration.
        '''
        self.batch_size = self.config.batch_size
        self.learning_rate = self.config.learning_rate
        self.epsilon = self.config.epsilon
        self.eps_min = self.config.eps_min
        self.eps_decay = self.config.eps_decay
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.memory_size = self.config.memory_size
        self.score_to_beat = self.config.score_to_beat
        self.scores_window_size = self.config.scores_window_size
        self.max_timesteps_per_episode = self.config.max_timesteps_per_episode
        self.model_save_frequency = self.config.model_save_frequency
        self.log_dir = self.config.log_dir
        self.save_path = self.config.save_path

    def _init_networks(self):
        '''
        Initializes the policy and target neural networks and sets up the optimizer and loss function.
        '''
        self.policy_net = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.MSELoss = nn.MSELoss()
