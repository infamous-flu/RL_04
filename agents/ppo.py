import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


class BaseNetwork(nn.Module):

    def __init__(self, input_dims, output_dims) -> None:
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.actor = BaseNetwork(n_observations, n_actions)
        self.critic = BaseNetwork(n_observations, 1)

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPO:

    def __init__(self, env, device, config):
        self.env = env
        self.device = device
        self.config = config
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self._init_hyperparameters()
        self._init_network()
        self.writer = SummaryWriter(self.log_dir)

    def learn(self, n_timesteps):
        i = 0
        t = 0
        self.episode_i = 0
        self.scores_window = deque([], maxlen=self.scores_window_size)
        while t < n_timesteps:
            observations, actions, log_probs, rewards, \
                values, dones, episode_lengths = self.rollout()
            batch_size = sum(episode_lengths)
            i += 1
            t += batch_size
            for episode_rewards, episode_length in zip(rewards, episode_lengths):
                self.episode_i += 1
                score = sum(episode_rewards)
                self.writer.add_scalar('Episode/Score', sum(episode_rewards), self.episode_i)
                self.writer.add_scalar('Episode/Length', episode_length, self.episode_i)
                self.scores_window.append(score)
                average_score = np.mean(self.scores_window)
                if average_score >= self.score_to_beat:
                    break
                if self.episode_i % self.scores_window_size == 0:
                    print(f'Episode {self.episode_i}\tAverage Score: {average_score:.4f}')
            if average_score >= self.score_to_beat:
                print(f'\nEnvironment solved in {self.episode_i - self.scores_window_size} episodes!\tAverage Score: {average_score:.4f}')
                break
            A_k = self.calculate_gae(rewards, values, dones)
            observations, actions, log_probs, A_k = \
                self._prepare_tensors(observations, actions, log_probs, A_k)
            with torch.no_grad():
                _, V = self.actor_critic(observations)
            G_k = A_k + V.squeeze()
            avg_policy_loss, avg_value_loss = \
                self.train(observations, actions, log_probs, A_k, G_k, batch_size)
            self.writer.add_scalar('Loss/Policy', avg_policy_loss, t)
            self.writer.add_scalar('Loss/Value', avg_value_loss, t)
            if i % self.mode_save_frequency == 0:
                self.save_model(self.save_path)
        self.save_model(self.save_path)
        self.writer.close()

    def rollout(self):
        observations, actions, log_probs, rewards, values, \
            dones, episode_lengths = [], [], [], [], [], [], []
        batch_t = 0
        while batch_t < self.max_timesteps_per_batch:
            episode_rewards, episode_values, episode_dones = [], [], []
            observation, _ = self.env.reset()
            done = False
            for episode_t in range(self.max_timesteps_per_episode):
                batch_t += 1
                episode_dones.append(done)
                action, log_prob, V = self.select_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                observations.append(observation)
                actions.append(action)
                log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_values.append(V)
                if done:
                    break
                observation = next_observation
            rewards.append(episode_rewards)
            values.append(episode_values)
            dones.append(episode_dones)
            episode_lengths.append(episode_t + 1)
        return observations, actions, log_probs, rewards, values, dones, episode_lengths

    def train(self, observations, actions, log_probs, A_k, G_k, batch_size):
        indices = list(range(batch_size))
        minibatch_size = batch_size // self.n_minibatches
        cumulative_policy_loss, cumulative_value_loss = 0.0, 0.0
        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]
                mini_observations, mini_actions, mini_log_probs, mini_advantage, mini_returns = \
                    observations[idx], actions[idx], log_probs[idx], A_k[idx], G_k[idx]
                V, new_log_probs, entropy = self.evaluate(mini_observations, mini_actions)
                mini_advantage = (mini_advantage - mini_advantage.mean()) / (mini_advantage.std() + 1e-8)
                ratios = torch.exp(new_log_probs - mini_log_probs)
                surrogate_1 = ratios * mini_advantage
                surrogate_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                policy_loss = (-torch.min(surrogate_1, surrogate_2)).mean()
                value_loss = self.MSELoss(V, mini_returns)
                cumulative_policy_loss += policy_loss
                cumulative_value_loss += value_loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return cumulative_policy_loss / (epoch + 1), cumulative_value_loss / (epoch + 1)

    def select_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, V = self.actor_critic(observation)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, V.squeeze()

    def evaluate(self, observations, actions):
        logits, V = self.actor_critic(observations)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return V.squeeze(), log_probs, dist.entropy()

    def calculate_return(self, episode_rewards):
        episode_return = 0
        for reward in reversed(episode_rewards):
            episode_return = reward + self.gamma * episode_return
        return episode_return

    def calculate_gae(self, rewards, values, dones):
        advantages = []
        for episode_rewards, episode_values, episode_dones in zip(rewards, values, dones):
            episode_advantages = []
            last_advantage = 0
            for i in reversed(range(len(episode_rewards))):
                if i + 1 < len(episode_rewards):
                    delta = episode_rewards[i] + self.gamma * \
                        episode_values[i+1] * (1 - episode_dones[i+1]) - episode_values[i]
                else:
                    delta = episode_rewards[i] - episode_values[i]
                advantage = delta + self.gamma * self.gae_lambda * (1 - episode_dones[i]) * last_advantage
                last_advantage = advantage
                episode_advantages.insert(0, advantage)
            advantages.extend(episode_advantages)
        return advantages

    def save_model(self, file_path):
        torch.save(self.actor_critic.state_dict(), file_path)

    def load_model(self, file_path):
        self.actor_critic.load_state_dict(torch.load(file_path))

    def _prepare_tensors(self, observations, actions, log_probs, A_k):
        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        A_k = torch.tensor(A_k, dtype=torch.float32).to(self.device)
        return observations, actions, log_probs, A_k

    def _init_hyperparameters(self):
        self.max_timesteps_per_batch = self.config.max_timesteps_per_batch
        self.learning_rate = self.config.learning_rate
        self.gamma = self.config.gamma
        self.gae_lambda = self.config.gae_lambda
        self.n_epochs = self.config.n_epochs
        self.n_minibatches = self.config.n_minibatches
        self.clip = self.config.clip
        self.value_coef = self.config.value_coef
        self.entropy_coef = self.config.entropy_coef
        self.score_to_beat = self.config.score_to_beat
        self.scores_window_size = self.config.scores_window_size
        self.max_timesteps_per_episode = self.config.max_timesteps_per_episode
        self.mode_save_frequency = self.config.model_save_frequency
        self.log_dir = self.config.log_dir
        self.save_path = self.config.save_path

    def _init_network(self):
        self.actor_critic = ActorCriticNetwork(self.n_observations, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.MSELoss = nn.MSELoss()
