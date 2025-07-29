import os

import config as c
import gymnasium as gym
import helpers as h
import log as l
import plot as p
import rl_env_cl as rl_cl
from sbx import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


def train():
    """
    Start training of the RL QAS Agent
    """
    
    # Create relevant directories and save config file
    path_run = c.PATH_RUN
    path_config = path_run + '/config.json'
    path_model = path_run + c.PATH_MODEL
    path_log = path_run + c.PATH_LOG
    path_data = path_run + c.PATH_DATA
    path_plot = path_run + c.PATH_PLOT
    h.create_directory(path_run)
    h.create_directory(path_model)
    h.create_directory(path_log)
    h.create_directory(path_data)
    h.create_directory(path_plot)
    l.save_config(path_config)
    
    if c.ML_TASK == "classification":
        env = gym.make("QuantumCircuitCl-v0")
    elif c.ML_TASK == "rl":
        env = gym.make("QuantumCircuitRl-v0")
    
    # Get hyperparameters set in 'config.py'; all other hyperparameters are set to default values
    hyperparameters = h.get_rl_hyperparameters()
    
    if c.RL_ALG == "ppo":
        model = PPO("MlpPolicy",
                env=env,
                tensorboard_log=path_log,
                **hyperparameters
                )
    elif c.RL_ALG == "a2c":
        model = A2C("MlpPolicy",
                env=env,
                tensorboard_log=path_log,
                policy_kwargs=dict(net_arch=[64,64]), # Statically set until bug in data logging for A2C policy_kwargs fixed
                **hyperparameters
                )
    else:
        raise ValueError(f"Invalid RL algorithm: {c.RL_ALG}")

    custom_callback = TensorboardCallback()
    
    for episode in range(1, c.EPISODES + 1):
        model.learn(
                total_timesteps=c.TIMESTEPS,
                callback=custom_callback,
                log_interval=1,
                tb_log_name="tensorboard",
                reset_num_timesteps=False
        )
        model.save(f"{path_model}/model_{episode*c.TIMESTEPS}")
        l.save_data_log(f"{path_data}data_log_{episode*c.TIMESTEPS}.json", env.unwrapped.data_log_run)
        
        # For debugging purposes (policy_kwargs not serializable for A2C algorithm)
        # my_dict = env.unwrapped.data_log_run
        # for key, value in my_dict.items():
        #     try:
        #         json.dumps(value)  # Versucht, den Wert zu serialisieren
        #     except TypeError:
        #         print(f"Der Wert für {key} ist nicht serialisierbar: {value}")
    
    # Save shared cache before closing environment (in case there are unsychronized data)
    env.unwrapped.circuit_accuracy_cache.save_cache()
    
    # Save data log    
    # l.save_data_log(f"{path_data}data_log_{c.EPISODES*c.TIMESTEPS}.json", env.unwrapped.data_log_run)

    # Generate and save plots
    metrics = ["episode_reward", "classification_accuracy_last", "steps", "circuit_depth"]
    for metric in metrics:
        p.plot_run(f"{path_data}data_log_{c.EPISODES*c.TIMESTEPS}.json", metric=metric)
    env.close()
    
    
def create_random_baseline():
    """
    Create a random baseline for the RL QAS Agent 
    """
    
    # Create relevant directories and save config file
    path_run = c.PATH_RUN
    path_config = path_run + '/config.json'
    path_model = path_run + c.PATH_MODEL
    path_log = path_run + c.PATH_LOG
    path_data = path_run + c.PATH_DATA
    path_plot = path_run + c.PATH_PLOT
    h.create_directory(path_run)
    h.create_directory(path_model)
    h.create_directory(path_log)
    h.create_directory(path_data)
    h.create_directory(path_plot)
    l.save_config(path_config)
    
    if c.ML_TASK == "classification":
        env = gym.make("QuantumCircuitCl-v0")
    elif c.ML_TASK == "rl":
        env = gym.make("QuantumCircuitRl-v0")
        
    env.reset()
    total_steps = c.EPISODES * c.TIMESTEPS
    terminated, truncated = False, False
    
    for step in range(1, total_steps + 1):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            terminated, truncated = False, False
    
    # Save shared cache before closing environment (in case there are unsychronized data)
    env.unwrapped.circuit_accuracy_cache.save_cache()
    
    l.save_data_log(f"{path_data}data_log_{total_steps}.json", env.unwrapped.data_log_run)
    
    # Generate and save plots
    metrics = ["episode_reward", "classification_accuracy_last", "steps", "circuit_depth"]
    for metric in metrics:
        p.plot_run(f"{path_data}data_log_{total_steps}.json", metric=metric)
    env.close()


def evaluate(path_model: str, episodes: int = 20):
    """
    Evaluate the model at the given path for the given number of episodes.
    """
    
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"The model path {path_model} does not exist.")
        
    env = gym.make("QuantumCircuit-v0")

    model = PPO.load(path=path_model, env=env)
    l.console_log(f"Model loaded from {path_model}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes)
    l.console_log(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    return mean_reward, std_reward


def test(path_model: str, steps: int = 1000):
    """
    Test the model at the given pat for the given number of steps.
    """
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"The model path {path_model} does not exist.")
    
    env = gym.make("QuantumCircuit-v0")
    
    model = PPO.load(path=path_model,env=env)
    l.console_log(f"Model loaded from {path_model}")

    obs, info = env.reset()

    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated: 
            obs, info = env.reset()
                        
    env.close()


# TODO: Doesn't work properly yet
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        if hasattr(env, 'accuracy_prev'):
            accuracy_prev = env.accuracy_prev
        else:
            # Falls "accuracy" nicht direkt zugänglich ist, nutze `get_attr`
            accuracy_prev = self.training_env.get_attr("accuracy_prev")[0]
        self.logger.record("custom/accuracy_prev", accuracy_prev)
        
        if hasattr(env, 'accuracy_average'):
            accuracy_average = env.accuracy_average
        else:
            # Falls "accuracy" nicht direkt zugänglich ist, nutze `get_attr`
            accuracy_average = self.training_env.get_attr("accuracy_average")[0]
        self.logger.record("custom/accuracy_average", accuracy_average)
        
        return True
    


if __name__ == "__main__":

    pass