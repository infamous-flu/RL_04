import os
import re
import time
import random
import warnings
from datetime import datetime
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from config.agents_config import PPOConfig
from config.experiment_config import TrainingConfig, EvaluationConfig


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
        env (gym.Env): The gym environment in which the agent will operate.
        device (torch.device): The device (CPU or GPU) to perform computations.
        agent_config (DQNConfig): Configuration object containing hyperparameters for the PPO agent.
        seed (Optional[int]): Optional seed value for initialization to ensure reproducibility.
    """

    def __init__(self, env: gym.Env, device: torch.device, agent_config: PPOConfig, seed: Optional[int] = None):
        """
        Initializes the PPO agent with the environment, device, and configuration.

        Args:
            env (gym.Env): The gym environment in which the agent will operate.
            device (torch.device): The device (CPU or GPU) to perform computations.
            agent_config (DQNConfig): Configuration object containing hyperparameters for the PPO agent.
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
        self._init_network()                                  # Set up the neural network architecture

    def learn(self, training_config: TrainingConfig, evaluation_config: Optional[EvaluationConfig] = None):
        """
        Train the PPO agent using the given training configuration and optionally evaluate during training. 
        Raises errors if evaluation is required by the training configuration but not provided, or if there
        are mismatches in environment IDs or agent types between training and evaluation configurations.

        Args:
            training_config (TrainingConfig): The configuration containing training parameters such as the
            number of timesteps, the environment ID, the agent type, and the seed.
            evaluation_config (Optional[EvaluationConfig]): Optional configuration for evaluation parameters
            which includes settings like the environment ID and the agent type. Necessary if `evaluate_every`
            is set in the training configuration.
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
            observations, actions, log_probs, rewards, values = self.rollout()  # Collect a batch of trajectories

            # Learn using the collected batch of trajectories
            self.train(observations, actions, log_probs, rewards, values)

        # Final save and close the logger
        if self.checkpoint_frequency > 0:
            self.save_model(self.save_path)
        if self.writer is not None:
            self.writer.close()

    def rollout(self) -> Tuple[List[NDArray[np.float32]], List[int], List[torch.Tensor],
                               List[List[float]], List[List[torch.Tensor]]]:
        """
        Executes one rollout to collect training data until a batch of trajectories is filled. This
        method captures the sequence of observations, actions, corresponding log probabilities of
        actions, rewards, and estimated values from the environment until the specified batch size
        is reached.

        Returns:
            A tuple containing:
            - observations (List[NDArray[np.float32]]): States observed by the agent during the rollout.
            - actions (List[int]): Actions taken by the agent.
            - log probabilities (List[torch.Tensor]): Log probabilities of the actions taken.
            - rewards (List[List[float]]): Nested list of rewards received.
            - values (List[List[torch.Tensor]]): Nested list of value estimates.
        """

        batch_t = 0  # Initialize batch timestep counter
        observations, actions, log_probs, rewards, values = [], [], [], [], []

        while batch_t < self.max_timesteps_per_batch:
            episode_return = 0                                           # Initialize return for the episode
            episode_rewards, episode_values = [], []
            observation, _ = self.env.reset()                            # Reset the environment and get the initial observation
            done = False

            for episode_t in range(self.max_timesteps_per_episode):                         # Iterate over allowed timesteps
                self.t += 1                                                                 # Increment the global timestep counter
                batch_t += 1                                                                # Increment the batch timestep counter
                action, log_prob, V = self.select_action(observation)                       # Select an action
                next_observation, reward, terminated, truncated, _ = self.env.step(action)  # Take the action
                done = terminated or truncated                                              # Determine if the episode has ended

                episode_return += reward          # Update the episode return
                observations.append(observation)
                actions.append(action)
                log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_values.append(V)

                # Evaluate the agent periodically
                if self.evaluate_every > 0 and (self.t % self.evaluate_every == 0 or self.t == 1):
                    average_evaluation_return = self.evaluate(self.evaluation_config)
                    if self.enable_logging:
                        self.writer.add_scalar('Evaluation/AverageEvaluationReturn', average_evaluation_return, self.t)
                else:
                    average_evaluation_return = None

                # Save the model at specified intervals
                if self.checkpoint_frequency > 0 and self.t % self.checkpoint_frequency == 0:
                    self.save_model(self.save_path)

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

                observation = next_observation  # Update the observation

            rewards.append(episode_rewards)
            values.append(episode_values)

            self.episode_i += 1                         # Increment the episode counter
            self.returns_window.append(episode_return)  # Record the episode return
            self.lengths_window.append(episode_t + 1)   # Record the episode length

            if self.enable_logging:
                if self.episode_i >= self.window_size:
                    self.writer.add_scalar(
                        'Training/AverageTrainingReturn', np.mean(self.returns_window), self.t)   # Log the average return
                    self.writer.add_scalar(
                        'Training/AverageEpisodeLength', np.mean(self.lengths_window), self.t)    # Log the average episode length
                self.writer.add_scalar('Episodic/EpisodeReturn', episode_return, self.episode_i)  # Log the episode return
                self.writer.add_scalar('Episodic/EpisodeLength', episode_t + 1, self.episode_i)   # Log the episode length

        return observations, actions, log_probs, rewards, values

    def train(self, observations: List[NDArray[np.float32]], actions: List[int],
              log_probs: List[torch.Tensor], rewards: List[List[float]],
              values: List[List[torch.Tensor]]):
        """
        Performs a learning update using the Proximal Policy Optimization (PPO) algorithm. This method applies
        Generalized Advantage Estimation (GAE) to compute advantage estimates and subsequently performs several
        epochs of minibatch updates on shuffled data to optimize the policy and value functions.

        Args:
            observations (List[NDArray[np.float32]]): States observed by the agent.
            actions (List[int]): Actions taken by the agent.
            log probabilities (List[torch.Tensor]): Log probabilities of the actions at the time they were taken.
            rewards (List[List[float]]): Nested list of rewards received.
            values (List[List[torch.Tensor]]): Nested list of value estimates.

        This method updates the policy by performing several passes over the data, using minibatch SGD to find an
        optimal update that minimizes the policy loss while maximizing expected returns as dictated by PPO's
        clipped objective. It also logs various metrics such as policy loss, value loss, and entropy to monitor
        training progress.
        """

        # Calculate the advantage estimates using Generalized Advantage Estimation (GAE)
        advantages = self.calculate_gae(rewards, values)

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
                new_V, new_log_probs, entropy = self.get_values(mini_observations, mini_actions)

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
        This method supports both deterministic and stochastic action selections depending on the mode specified.
        Deterministic selection chooses the action with the highest probability, suitable for evaluation,
        while stochastic selection samples from the policy distribution, suitable for exploration during training.

        Args:
            observation (NDArray[np.float32]): The current state observation from the environment.
            deterministic (bool): If True, the action choice is deterministic (the max probability action). \
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

    def get_values(self, observations: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the given observations and actions using the actor-critic network to obtain values,
        log probabilities, and entropy of the policy distribution.

        Args:
            observations (torch.Tensor): A batch of observations from the environment, formatted as tensors.
            actions (torch.Tensor): A batch of actions taken by the agent, formatted as tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - The value estimates (V) for the given observations from the critic part of the network.
                - The log probabilities of the actions taken, derived from the actor part of the network.
                - The entropy of the policy distribution, used to encourage exploration.
        """

        logits, V = self.actor_critic(observations)    # Forward pass through the actor-critic network
        dist = Categorical(logits=logits)              # Create a categorical distribution based on the logits
        log_probs = dist.log_prob(actions)             # Calculate the log probabilities of the actions
        return V.squeeze(), log_probs, dist.entropy()

    def calculate_gae(self, rewards: List[List[float]], values: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Calculates the Generalized Advantage Estimation (GAE) for a set of rewards and values. GAE is used to
        reduce the variance of advantage estimates without increasing bias, which is crucial for stable policy
        updates in PPO.

        Args:
            rewards (List[List[float]]): Nested list of rewards obtained from the environment.
            values (List[List[torch.Tensor]]): Nested list of value estimates from the critic.

        Returns:
            List[torch.Tensor]: The list of advantage estimates, each tensor corresponding to an episode's advantages.
        """

        advantages = []  # List to store the computed advantages for the batch

        # Iterate through each episode's rewards, amd values.
        for episode_rewards, episode_values in zip(rewards, values):
            episode_advantages = []  # List to store the current episode's advantages
            last_advantage = 0       # Initialize the last advantage to zero

            # Process each timestep in reverse to compute the advantage values
            for i in reversed(range(len(episode_rewards))):
                if i + 1 < len(episode_rewards):
                    # Compute the temporal difference (delta) for the current step
                    delta = episode_rewards[i] + self.gamma * episode_values[i+1] - episode_values[i]
                else:
                    # Compute the difference between the reward and current value for the last step
                    delta = episode_rewards[i] - episode_values[i]

                # Compute the advantage value using the GAE formula
                advantage = delta + self.gamma * self.gae_lambda * last_advantage
                last_advantage = advantage               # Update the last advantage for the next timestep
                episode_advantages.insert(0, advantage)  # Insert at the front of the list

            advantages.extend(episode_advantages)  # Extend the main advantages with the current episode's

        return advantages

    def evaluate(self, evaluation_config: EvaluationConfig) -> float:
        """
        Evaluates the PPO agent by running it through a series of episodes in an evaluation environment,
        according to the provided configuration. It temporarily alters random states for reproducibility,
        records evaluation episodes if configured, and safely restores states afterwards.

        Args:
            evaluation_config (EvaluationConfig): Configuration object containing evaluation parameters.

        Returns:
            float: Average return across all evaluation episodes, calculated as the sum of rewards obtained per 
            episode divided by the number of episodes.
        """

        def extract_timestamp() -> str:
            """Extract the first timestamp from any of the provided paths or generate a new one."""
            timestamp_pattern = r'\d{8}\-\d{6}|\d{8}\d{6}'
            # Attempt to extract the timestamp from agent attributes
            for attr in ['log_dir', 'save_path']:
                if hasattr(self, attr):
                    value = getattr(self, attr, None)
                    if value:
                        match = re.search(timestamp_pattern, value)
                        if match:
                            return match.group(0)
            # Fallback to a new timestamp if no match is found
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
                video_folder = os.path.join('recordings', evaluation_config.env_id, 'ppo', timestamp)
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
                done = False
                for _ in range(evaluation_config.max_timesteps_per_episode):
                    action = self.select_action(
                        observation, deterministic=evaluation_config.deterministic
                    )
                    observation, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_return += reward
                    done = terminated or truncated
                    if done:
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

        model_and_or_optim = {'model_state_dict': self.actor_critic.state_dict()}
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
        self.actor_critic.load_state_dict(model_and_or_optim['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in model_and_or_optim:
            self.optimizer.load_state_dict(model_and_or_optim['optimizer_state_dict'])

    def _prepare_tensors(self, observations: List[NDArray[np.float32]], actions: List[int],
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

    def _init_training_parameters(self):
        """
        Initializes training parameters from the configuration.
        """

        self.env_id = self.training_config.env_id
        self.max_timesteps_per_episode = self.training_config.max_timesteps_per_episode
        self.n_timesteps = self.training_config.n_timesteps
        self.evaluate_every = self.training_config.evaluate_every
        self.window_size = self.training_config.window_size
        self.print_every = self.training_config.print_every
        self.enable_logging = self.training_config.enable_logging
        self.log_dir = self.training_config.log_dir
        self.checkpoint_frequency = self.training_config.checkpoint_frequency
        self.save_path = self.training_config.save_path

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
