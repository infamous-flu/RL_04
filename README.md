# Reinforcement Learning Assignment 4: DQN vs. PPO

This project demonstrates Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents for reinforcement learning (RL) experiments using the LunarLander environment. The agents are trained and evaluated through configurable settings, with the results recorded and logged.

## Project Structure

- **`agents/`**: Contains implementations of the DQN and PPO agents.
  - `dqn.py`: DQN agent implementation.
  - `ppo.py`: PPO agent implementation.

- **`config/`**: Holds configuration classes for agents and experiments.
  - `agents_config.py`: Agent-specific hyperparameters.
  - `experiment_config.py`: Settings for training and evaluation.

- **`utils/`**: Utility functions.
  - `helpers.py`: Various utility functions.

- **`main.py`**: The main entry point of the project, containing the training and evaluation pipeline.

- **`environment.yml`**: Contains all dependencies required for Conda environment setup.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:infamous-flu/RL_04.git
   ```

2. **Create a Conda environment**:
    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the Environment**:
    ```bash
    conda activate deep_rl_env  # Or replace with the environment name in the YAML file
    ```

## Usage

1. **Run the Training and Evaluation**:
Use the **`main.py`** script to train and evaluate the agents:
    ```bash
    python main.py
    ```

2. **Customize Training**:
- Modify the **`main.py`** file to adjust the environment parameters via the **`kwargs`** argument.
- Update the agent configuration (e.g. `DQNConfig` or `PPOConfig`) as needed.

3. **Visualize Results**:
- Check the recordings folder for training and evaluation videos (if recording is enabled).
- Use TensorBoard for training progress visualization:

    ```bash
    tensorboard --logdir=runs
    ```
