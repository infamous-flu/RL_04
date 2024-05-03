import os
import random
from datetime import datetime
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


Experience = namedtuple('Experience', ('observation', 'action', 'next_observation', 'reward', 'done'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:

    def __init__(self, env, config):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self._init_hyperparameters()
        self._init_networks()
        self.writer = SummaryWriter(self.log_dir)
        self.memory = ReplayMemory(self.memory_size)

    def learn(self, n_timesteps):
        self.i = 0
        self.t = 0
        scores_window = deque([], maxlen=self.scores_window_size)
        while self.t < n_timesteps:
            score = self.rollout()
            scores_window.append(score)
            average_score = np.mean(scores_window)
            if average_score >= self.score_to_beat:
                print(f'\nEnvironment solved in {self.i - self.scores_window_size} episodes!\tAverage Score: {average_score:.4f}')
                break
            if self.i % self.scores_window_size == 0:
                print(f'Episode {self.i}\tAverage Score: {average_score:.4f}')
            if self.i % self.model_save_frequency == 0:
                self.save_model(self.save_path)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        self.save_model(self.save_path)
        self.writer.close()

    def rollout(self):
        score = 0
        observation, _ = self.env.reset()
        for episode_t in range(self.max_timesteps_per_episode):
            self.t += 1
            action = self.select_action(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            score += reward
            self.memory.push(observation, action, next_observation, reward, done)
            loss = self.train()
            if loss is not None:
                self.writer.add_scalar('Loss', loss, self.t)
            if done:
                break
            observation = next_observation
        self.i += 1
        self.writer.add_scalar('Episode/Score', score, self.i)
        self.writer.add_scalar('Episode/Length', episode_t + 1, self.i)
        return score

    def select_action(self, observation):
        if random.random() > self.epsilon:
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(observation).squeeze()
            action = torch.argmax(q_values).item()
        else:
            action = self.env.action_space.sample()
        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        experiences = self.memory.sample(self.batch_size)
        observations, actions, next_observations, rewards, dones = zip(*experiences)
        observations, actions, next_observations, rewards, dones = \
            self._prepare_tensors(observations, actions, next_observations, rewards, dones)
        current_q_values = self.policy_net(observations).gather(1, actions.unsqueeze(1)).squeeze(-1)
        with torch.no_grad():
            max_q_values, _ = self.target_net(next_observations).max(dim=1)
        target_q_values = rewards + self.gamma * max_q_values * (1 - dones)
        loss = self.MSELoss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()
        return loss.item()

    def update_target_network(self):
        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def calculate_return(self, episode_rewards):
        episode_return = 0
        for reward in reversed(episode_rewards):
            episode_return = reward + self.gamma * episode_return
        return episode_return

    def save_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _prepare_tensors(self, observations, actions, next_observations, rewards, dones):
        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        return observations, actions, next_observations, rewards, dones

    def _init_hyperparameters(self):
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
        self.policy_net = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.MSELoss = nn.MSELoss()
