import random
from collections import deque
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


class BaseNetwork(nn.Module):
    """
    A simple feedforward neural network that serves as the base for both the actor and critic networks.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output fully connected layer.
    """

    def __init__(self, input_dims: int, output_dims: int):
        """
        Initializes the network with specified dimensions for input and output layers.

        Args:
            input_dims (int): Dimensionality of the input features.
            output_dims (int): Dimensionality of the output layer.
        """

        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)   # First fully connected layer
        self.fc2 = nn.Linear(128, 128)          # Second fully connected layer
        self.fc3 = nn.Linear(128, output_dims)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network.
        """

        x = F.relu(self.fc1(x))  # Activation function after the first layer
        x = F.relu(self.fc2(x))  # Activation function after the second layer
        return self.fc3(x)       # Output from the final layer


class ActorCriticNetwork(nn.Module):
    """
    A network composed of two separate networks (actor and critic) for use in actor-critic methods.

    Attributes:
        actor (BaseNetwork): The actor network that outputs actions.
        critic (BaseNetwork): The critic network that outputs value estimates.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initializes the ActorCriticNetwork with specified dimensions for input and output layers.

        Args:
            n_observations (int): Number of observation inputs expected by the network.
            n_actions (int): Number of possible actions the agent can take.
        """

        super(ActorCriticNetwork, self).__init__()
        self.actor = BaseNetwork(n_observations, n_actions)  # Actor network initialization
        self.critic = BaseNetwork(n_observations, 1)         # Critic network initialization

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns the outputs from both the actor and the critic networks.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Contains the output from the actor and the critic networks.
        """

        return self.actor(x), self.critic(x)  # Return actor and critic network outputs


class PPO:
    """
    A class implementing the Proximal Policy Optimization (PPO) reinforcement learning algorithm.

    Attributes:
        env (gym.Env): The environment in which the agent operates.
        device (torch.device): The device (CPU or GPU) on which the computations are performed.
        config (PPOConfig): Configuration parameters for setting up the PPO agent.
        seed (int): Seed for the pseudo random generators.
    """

    def __init__(self, env: gym.Env, device: torch.device, agent_config, experiment_config):
        """
        Initializes the PPO agent with the environment, device, and configuration.

        Args:
            env (gym.Env): The gym environment.
            device (torch.device): The device (CPU or GPU) to perform computations.
            agent_config (PPOConfig): A dataclass containing the agent's hyperparameter.
        """

        self.env = env                                        # The gym environment where the agent will interact
        self.device = device                                  # The computation device (CPU or GPU)
        self.agent_config = agent_config                      # Configuration for agent hyperparameters
        self.experiment_config = experiment_config            # Configuration for experiment settings
        self.seed = self.experiment_config.seed               # Seed for the pseudo random generators
        if self.seed is not None:
            self._set_seed(self.seed)                         # Set the seed in various components
        self.n_observations = env.observation_space.shape[0]  # Number of features in the observation space
        self.n_actions = env.action_space.n                   # Number of possible actions
        self._init_hyperparameters()                          # Initialize the hyperparameters based on the configuration
        self._init_network()                                  # Set up the neural network architecture
        self._init_writer()                                   # Prepare the TensorBoard writer for logging

    def train(self, n_timesteps: int):
        """
        Trains the PPO agent for a given number of timesteps.

        Args:
            n_timesteps (int): The number of timesteps to train the agent.
        """

        self.t = 0                                                      # Initialize global timestep counter
        self.batch_i = 0                                                # Initialize batch counter
        self.episode_i = 0                                              # Initialize episode counter
        self.scores_window = deque([], maxlen=self.scores_window_size)  # Used for tracking the average score

        while self.t < n_timesteps:
            observations, actions, log_probs, rewards, values, dones = self.rollout()  # Collect a batch of trajectories
            self.batch_i += 1

            # Check for early stopping if the environment is considered solved
            if self.is_environment_solved():
                print(f'\nEnvironment solved in {self.t} timesteps!', end='\t')
                print(f'Average Score: {np.mean(self.scores_window):.3f}')
                break

            # Learn using the collected batch of trajectories
            self.learn(observations, actions, log_probs, rewards, values, dones)

            # Save the model at specified intervals
            if self.batch_i % self.checkpoint_frequency == 0:
                self.save_model(self.save_path)

        # Final save and close the logger
        self.save_model(self.save_path)
        if self.writer is not None:
            self.writer.close()

    def rollout(self) -> Tuple[List[np.ndarray], List[int], List[torch.Tensor],
                               List[List[float]], List[List[torch.Tensor]], List[List[bool]]]:
        """
        Executes one rollout to collect training data until a batch of trajectories is filled
        or the environment is considered solved.

        Returns:
            A tuple containing:
            - observations (List[np.ndarray]): The environment's states observed by the agent.
            - actions (List[int]): Actions taken by the agent.
            - log probabilities (List[torch.Tensor]): Log probabilities of the actions taken.
            - rewards (List[List[float]]): Rewards received after taking actions.
            - values (List[List[torch.Tensor]]): Estimated values after taking actions.
            - dones (List[List[bool]]): Boolean flags indicating if an episode has ended.
        """

        batch_t = 0  # Initialize batch timestep counter
        observations, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        while batch_t < self.max_timesteps_per_batch:
            score = 0                                                    # Initialize score for the episode
            episode_rewards, episode_values, episode_dones = [], [], []
            observation, _ = self.env.reset()                            # Reset the environment and get the initial observation
            done = False

            for episode_t in range(self.max_timesteps_per_episode):      # Iterate over allowed timesteps
                self.t += 1                                              # Increment the global timestep counter
                batch_t += 1                                             # Increment the batch timestep counter
                episode_dones.append(done)
                action, log_prob, V = self.select_action(observation)                       # Select an action
                next_observation, reward, terminated, truncated, _ = self.env.step(action)  # Take the action
                done = terminated or truncated                                              # Determine if the episode has ended

                score += reward                   # Update the score
                observations.append(observation)
                actions.append(action)
                log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_values.append(V)

                # Print progress periodically
                if self.t % self.print_every == 0:
                    print(f'Timestep {self.t}\tAverage Score: {np.mean(self.scores_window):.3f}')

                if done:
                    break  # If the episode is finished, exit the loop

                observation = next_observation  # Update the observation

            rewards.append(episode_rewards)
            values.append(episode_values)
            dones.append(episode_dones)

            self.episode_i += 1               # Increment the episode counter
            self.scores_window.append(score)  # Record the score

            if self.enable_logging:
                self.writer.add_scalar('Common/AverageReturn', np.mean(self.scores_window), self.t)  # Log the average score
                self.writer.add_scalar('Common/EpisodeReturn', score, self.t)                        # Log the score
                self.writer.add_scalar('Common/EpisodeLength', episode_t + 1, self.episode_i)        # Log the episode length

            # Stop the rollout if environment is considered solved
            if self.is_environment_solved():
                break

        return observations, actions, log_probs, rewards, values, dones

    def learn(self, observations: List[np.ndarray], actions: List[int], log_probs: List[torch.Tensor],
              rewards: List[List[float]], values: List[List[torch.Tensor]], dones: List[List[bool]]):
        """
        Performs a learning update using the Proximal Policy Optimization (PPO) algorithm.

        Args:
            observations (List[np.ndarray]): The environment's states observed by the agent.
            actions (List[int]): Actions taken by the agent.
            log probabilities (List[torch.Tensor]): Log probabilities of the actions taken.
            rewards (List[List[float]]): Lists of rewards received after taking actions.
            values (List[List[torch.Tensor]]): Lists of estimated values after taking actions.
            dones (List[List[bool]]): Lists of boolean flags indicating if an episode has ended.
        """

        # Calculate the advantage estimates using Generalized Advantage Estimation (GAE)
        advantages = self.calculate_gae(rewards, values, dones)

        # Prepare the data by converting to PyTorch tensors for neural network processing
        observations, actions, log_probs, advantages = \
            self._prepare_tensors(observations, actions, log_probs, advantages)

        # Use the actor-critic network to predict the current value estimates
        with torch.no_grad():
            _, V = self.actor_critic(observations)
        returns = advantages + V.squeeze()  # Combine advantages with value estimates to get the returns

        # Setup for minibatch learning
        batch_size = len(observations)
        remainder = batch_size % self.n_minibatches
        minibatch_size = batch_size // self.n_minibatches
        minibatch_sizes = [minibatch_size + 1 if i < remainder else
                           minibatch_size for i in range(self.n_minibatches)]

        cumulative_policy_loss, cumulative_value_loss, cumulative_entropy_loss = 0.0, 0.0, 0.0
        for _ in range(self.n_epochs):                            # Loop over the number of specified epochs
            indices = torch.randperm(batch_size).to(self.device)  # Shuffle indices for minibatch creation
            start = 0
            for minibatch_size in minibatch_sizes:
                # Slice the next minibatch
                end = start + minibatch_size
                mini_indices = indices[start:end]
                mini_observations = observations[mini_indices]
                mini_actions = actions[mini_indices]
                mini_log_probs = log_probs[mini_indices]
                mini_advantages = advantages[mini_indices]
                mini_returns = returns[mini_indices]

                # Normalize advantages to reduce variance and improve training stability
                if self.normalize_advantage:
                    mini_advantages = (mini_advantages - mini_advantages.mean()) / (mini_advantages.std() + 1e-10)

                # Evaluate the current policy's performance on the minibatch to get new values, log probs, and entropy
                new_V, new_log_probs, entropy = self.evaluate(mini_observations, mini_actions)

                # Calculate the ratios of the new log probabilities to the old log probabilities
                ratios = torch.exp(new_log_probs - mini_log_probs)

                # Calculate the first part of the surrogate loss
                surrogate_1 = ratios * mini_advantages

                # Calculate the second part of the surrogate loss, applying clipping to reduce variability
                surrogate_2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * mini_advantages

                # Calculate the final policy loss using the clipped and unclipped surrogate losses
                policy_loss = (-torch.min(surrogate_1, surrogate_2)).mean()

                # Calculate the value loss using Mean Squared Error between predicted and actual returns
                value_loss = self.MSELoss(new_V.squeeze(), mini_returns)

                # Combine the policy and value losses, adjusting for entropy to promote exploration
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                # Accumulate losses for monitoring
                cumulative_policy_loss += policy_loss
                cumulative_value_loss += value_loss
                cumulative_entropy_loss += entropy.mean()

                # Perform backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                start = end  # Update the start index for the next minibatch

        if self.enable_logging:
            mean_policy_loss = cumulative_policy_loss / (self.n_epochs * self.n_minibatches)
            mean_value_loss = cumulative_value_loss / (self.n_epochs * self.n_minibatches)
            mean_entropy_loss = cumulative_entropy_loss / (self.n_epochs * self.n_minibatches)

            self.writer.add_scalar('PPO/PolicyLoss', mean_policy_loss, self.t)    # Log the policy loss
            self.writer.add_scalar('PPO/ValueLoss', mean_value_loss, self.t)      # Log the value loss
            self.writer.add_scalar('PPO/EntropyLoss', mean_entropy_loss, self.t)  # Log the entropy loss

    def select_action(self, observation: NDArray[np.float32], deterministic: bool = False):
        """
        Selects an action based on the current observation using the policy defined by the actor-critic network.

        Args:
            observation (NDArray[np.float32]): The current state observation from the environment.
            deterministic (bool): If True, the action choice is deterministic (the max probability action).
                                  If False, the action is sampled stochastically according to the policy distribution.

        Returns:
            int: The action selected by the agent when deterministic is True.
            tuple: A tuple containing the following when deterministic is False:
                - int: The action selected by the agent.
                - torch.Tensor: The log probability of the selected action.
                - torch.Tensor: The value estimate from the critic network.
        """

        # Convert the observation to a tensor and add a batch dimension (batch size = 1)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Model prediction: Get logits and state value estimate
        with torch.no_grad():
            logits, V = self.actor_critic(observation)
        # Decide whether to select the best action based on model prediction or sample stochastically
        if deterministic:
            # Select the action with the highest probability
            return torch.argmax(logits).item()
        else:
            # Create a probability distribution over actions based on the logits and sample an action
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, V.squeeze()

    def evaluate(self, observations: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the given observations and actions using the actor-critic network
        to obtain log probabilities, values, and entropy.

        Args:
            observations (torch.Tensor): The batch of observations.
            actions (torch.Tensor): The batch of actions taken.

        Returns:
            A tuple containing:
                torch.Tensor: The value estimates for the given observations.
                torch.Tensor: The log probabilities of the taken actions.
                torch.Tensor: The entropy of the policy distribution.
        """

        logits, V = self.actor_critic(observations)    # Forward pass through the actor-critic network
        dist = Categorical(logits=logits)              # Create a categorical distribution based on the logits
        log_probs = dist.log_prob(actions)             # Calculate the log probabilities of the actions
        return V.squeeze(), log_probs, dist.entropy()

    def calculate_gae(self, rewards: List[List[float]], values: List[List[torch.Tensor]],
                      dones: List[List[bool]]) -> List[torch.Tensor]:
        """
        Calculates the Generalized Advantage Estimation (GAE) for a set of rewards, values, and done signals.

        Args:
            rewards (List[List[float]]): Rewards obtained from the environment.
            values (List[List[torch.Tensor]]): Value estimates from the critic.
            dones (List[List[bool]]): Done signals indicating the end of an episode.

        Returns:
            List[torch.Tensor]: The list of advantage estimates.
        """

        advantages = []  # List to store the computed advantages for the batch

        # Iterate through each episode's rewards, values, and done signals
        for episode_rewards, episode_values, episode_dones in zip(rewards, values, dones):
            episode_advantages = []  # List to store the current episode's advantages
            last_advantage = 0       # Initialize the last advantage to zero

            # Process each timestep in reverse to compute the advantage values
            for i in reversed(range(len(episode_rewards))):
                if i + 1 < len(episode_rewards):
                    # Compute the temporal difference (delta) for the current step
                    delta = episode_rewards[i] + self.gamma * \
                        episode_values[i+1] * (1 - episode_dones[i+1]) - episode_values[i]
                else:
                    # Compute the difference between the reward and current value for the last step
                    delta = episode_rewards[i] - episode_values[i]

                # Compute the advantage value using the GAE formula
                advantage = delta + self.gamma * self.gae_lambda * (1 - episode_dones[i]) * last_advantage
                last_advantage = advantage               # Update the last advantage for the next timestep
                episode_advantages.insert(0, advantage)  # Insert at the front of the list

            advantages.extend(episode_advantages)  # Extend the main advantages with the current episode's

        return advantages

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
        Saves the actor-critic network's model parameters to the specified file path.
        Optionally, it also saves the optimizer state.

        Args:
            file_path (str): The path to the file where the model parameters are to be saved.
            save_optimizer (bool): If True, saves the optimizer state as well. Defaults to False.
        """

        model_and_or_optim = {'model_state_dict': self.actor_critic.state_dict()}
        if save_optimizer:
            model_and_or_optim['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(model_and_or_optim, file_path)

    def load_model(self, file_path: str, load_optimizer: bool = True):
        """
        Loads model parameters into the actor-critic network, and optionally loads the optimizer state.

        Args:
            file_path (str): The path to the file from which to load the model parameters.
            load_optimizer (bool): If True, loads the optimizer state as well. Defaults to False.
        """

        model_and_or_optim = torch.load(file_path)
        self.actor_critic.load_state_dict(model_and_or_optim['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in model_and_or_optim:
            self.optimizer.load_state_dict(model_and_or_optim['optimizer_state_dict'])

    def _prepare_tensors(self, observations: List[np.ndarray], actions: List[int],
                         log_probs: List[torch.Tensor], advantages: List[torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts lists of observations, actions, log probs, and advantages into tensors
        and sends them to the specified device.

        Args:
            observations: List of observations from the environment.
            actions: List of actions taken.
            log_probs: List of log probabilities of the actions taken.
            advantages: List of advantage estimates.

        Returns:
            Tuple of Tensors: Prepared tensors of observations, actions, log_probs, and advantages.
        """

        observations = torch.tensor(np.stack(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(actions), dtype=torch.int64).to(self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return observations, actions, log_probs, advantages

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

        # Agent hyperparameters
        self.learning_rate = self.agent_config.learning_rate
        self.max_timesteps_per_batch = self.agent_config.max_timesteps_per_batch
        self.n_minibatches = self.agent_config.n_minibatches
        self.n_epochs = self.agent_config.n_epochs
        self.clip_range = self.agent_config.clip_range
        self.gamma = self.agent_config.gamma
        self.gae_lambda = self.agent_config.gae_lambda
        self.normalize_advantage = self.agent_config.normalize_advantage
        self.value_coef = self.agent_config.value_coef
        self.entropy_coef = self.agent_config.entropy_coef

        # Experiment settings
        self.score_threshold = self.experiment_config.score_threshold
        self.scores_window_size = self.experiment_config.scores_window_size
        self.max_timesteps_per_episode = self.experiment_config.max_timesteps_per_episode
        self.checkpoint_frequency = self.experiment_config.checkpoint_frequency
        self.print_every = self.experiment_config.print_every
        self.enable_logging = self.experiment_config.enable_logging
        self.log_dir = self.experiment_config.log_dir
        self.save_path = self.experiment_config.save_path

    def _init_network(self):
        """
        Initializes the actor-critic neural network and sets up the optimizer and loss function.
        """

        self.actor_critic = ActorCriticNetwork(self.n_observations, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
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
            self.writer.add_text('PPO/Hyperparameters', hyperparameters_str)
        else:
            self.writer = None
