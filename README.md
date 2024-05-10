# Reinforcement Learning Assignment 4: DQN vs. PPO

This project demonstrates Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents for reinforcement learning (RL) experiments using gymnasium environments. The agents are trained and evaluated through configurable settings, with the results recorded and logged.

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

### Run the Training and Evaluation ###

Use the main.py script to train and evaluate the agents with customizable command-line arguments:

```bash
python main.py --env_id LunarLander-v2 --agent_type dqn --device cuda --n_timesteps 300000
```

This will run the DQN agent on the LunarLander-v2 environment with 300,000 timesteps on a CUDA device, if available.

### Command Line Options ###

- Environment ID (--env_id): Specify the gym environment ID.
- Agent Type (--agent_type): Choose between 'dqn' and 'ppo' for the type of RL agent.
- Computation Device (--device): Choose 'cuda' or 'cpu' depending on available resources.
- Number of Training Timesteps (--n_timesteps): Set the number of timesteps for training.
- Seed (--seed): Provide a seed for reproducibility of training results.

### Additional Customizations ###

- Update the agent configuration (`DQNConfig` or `PPOConfig`) directly in their respective configuration classes as needed.
- Update the experiment configuration (`TrainingConfig` or `EvaluationConfig`) directly in their configuration classes as needed.

### Visualize Results ###

- Check the recordings folder for training and evaluation videos (if recording is enabled).
- Use TensorBoard for training progress visualization:

  ```bash
  tensorboard --logdir=runs
  ```
