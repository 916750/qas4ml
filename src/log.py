"""Functions for logging data and messages"""

import json
from datetime import datetime

import config as c


def get_config():
    
    config_dict = {
        "DEVICE": c.DEVICE,
        "COMP_UNIT": c.COMP_UNIT,
        "CONFIG_NR": c.CONFIG_NR,
        "RUN_NR": c.RUN_NR,
        "LOG_DATA_LEVEL": c.LOG_DATA_LEVEL,
        "SEED": c.SEED,
        "ACTION_SPACE": c.ACTION_SPACE,
        "PENALTY_ILLEGAL_ACTION": c.PENALTY_ILLEGAL_ACTION,
        "TIMESTEPS": c.TIMESTEPS,
        "EPISODES": c.EPISODES,
        "REWARD_MODE": c.REWARD_MODE,
        "RL_ALG": c.RL_ALG,
        "CIRCUIT_MEASUREMENT_MODE": c.CIRCUIT_MEASUREMENT_MODE,
        "CIRCUIT_DEPTH_MAX": c.CIRCUIT_DEPTH_MAX,
        "CIRCUIT_DEPTH_DYNAMIC": c.CIRCUIT_DEPTH_DYNAMIC,
        "GATE_PARAM_MODE": c.GATE_PARAM_MODE,
        "OPT_JAX": c.OPT_JAX,
        "OPT": c.OPT,
        "OPT_DYNAMIC_LEARNING_RATE": c.OPT_DYNAMIC_LEARNING_RATE,
        "OPT_LEARNING_RATE": c.OPT_LEARNING_RATE,
        "OPT_EPOCHS": c.OPT_EPOCHS,
        "OPT_BATCH_MODE": c.OPT_BATCH_MODE,
        "OPT_USE_VALIDATION_BATCH": c.OPT_USE_VALIDATION_BATCH,
        # "OPT_CONV_TOL": c.OPT_CONV_TOL, # Commented out as not used in current implementation
        "ML_TASK": c.ML_TASK,
        "CLASS_TASK": c.CLASS_TASK,
        "PATH_RUN": c.PATH_RUN,
        "PERFORMANCE_TARGET": c.PERFORMANCE_TARGET,
        "TEST_SIZE": c.TEST_SIZE,
        "TEST_RANDOM": c.TEST_RANDOM,
        "BONUS_REWARD": c.BONUS_REWARD
    }

    if c.RL_ALG == "ppo":
        config_dict["PPO_HYPERPARAMETERS"] = c.PPO_HYPERPARAMETERS
    elif c.RL_ALG == "a2c":
        config_dict["A2C_HYPERPARAMETERS"] = c.A2C_HYPERPARAMETERS
        
    if c.CIRCUIT_DEPTH_DYNAMIC:
        config_dict["CIRCUIT_DEPTH_DYNAMIC_INIT"] = c.CIRCUIT_DEPTH_DYNAMIC_INIT
        config_dict["CIRCUIT_DEPTH_DYNAMIC_INCR"] = c.CIRCUIT_DEPTH_DYNAMIC_INCR
        config_dict["CIRCUIT_DEPTH_STAGNATION_THRESHOLD"] = c.CIRCUIT_DEPTH_STAGNATION_THRESHOLD
        config_dict["CIRCUIT_DEPTH_WINDOW_SIZE"] = c.CIRCUIT_DEPTH_WINDOW_SIZE
        
    if c.GATE_PARAM_MODE == "static": 
        config_dict["GATE_PARAM_DEFAULT"] = c.GATE_PARAM_DEFAULT
    elif c.GATE_PARAM_MODE == "random":
        config_dict["GATE_PARAM_VALUE_RANGE"] = c.GATE_PARAM_VALUE_RANGE
        config_dict["GATE_PARAM_SEED_COUNT"] = c.GATE_PARAM_SEED_COUNT
        
    if c.OPT_BATCH_MODE == "fixed": 
        config_dict["OPT_BATCH_SIZE"] = c.OPT_BATCH_SIZE
    elif c.OPT_BATCH_MODE == "window":
        config_dict["OPT_BATCH_SIZE_WINDOW"] = c.OPT_BATCH_SIZE_WINDOW
    elif c.OPT_BATCH_MODE == "list":
        config_dict["OPT_BATCH_SIZE_LIST"] = c.OPT_BATCH_SIZE_LIST
    elif c.OPT_BATCH_MODE == "list_random":
        config_dict["OPT_BATCH_SIZE_LIST_RANDOM_LEN"] = c.OPT_BATCH_SIZE_LIST_RANDOM_LEN
        
    if c.OPT_USE_VALIDATION_BATCH: 
        config_dict["OPT_VALIDATION_BATCH_SIZE"] = c.OPT_VALIDATION_SET
        
    if c.BONUS_REWARD:
        config_dict["BONUS_REWARD_THRESHOLD"] = c.BONUS_REWARD_THRESHOLD
        
    return config_dict
         
            
def save_config(path: str): 
    """
    Saves the configuration to a JSON file.
    """
    if c.FLAG_LOG_DATA:
        
        config_dict = get_config()

        with open(path, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)
            
        console_log(f"Config.py saved as JSON file at '{path}'.")

    
def initialize_data_log_run():
    """
    Initializes the data log dictionary for a new run.
    """
    if c.FLAG_LOG_DATA:
        
        data_log_run = {
            "start_time": datetime.now().strftime("%d.%m.%Y:%H:%M:%S"),
            "config": get_config(),
            "episodes": {},
            "summary": {
                "episodes": 0,
                "steps": 0,
                "steps_episode_avg": 0,     # Average number of steps per episode over all episodes
                "steps_episode_max": 0,     # Maximum number of steps per episode over all episodes
                "steps_episode_min": 0,     # Minimum number of steps per episode over all episodes
                "reward_episode_avg": 0,    # Average reward per episode over all episodes
                "reward_episode_max": 0,    # Maximum reward per episode over all episodes
                "reward_episode_min": 0,    # Minimum reward per episode over all episodes
                "accuracy_episode_avg": 0,  # Average classification accuracy per episode over all episodes
                "accuracy_episode_max": 0,  # Maximum classification accuracy per episode over all episodes
                "accuracy_episode_min": 0,  # Minimum classification accuracy per episode over all episodes
                "circuit_depth_avg": 0,     # Average depth of circuits
                "circuit_depth_max": 0,     # Maximum depth of circuits
                "circuit_depth_min": 0,     # Minimum depth of circuits
                # "circuits_unique": 0,     # Number of unique circuits
                },
            "end_time": None
        }
    else:
        data_log_run = {}

    return data_log_run


def initialize_data_log_episode():
    """
    Initializes the data log dictionary for a new episode.
    """
    if c.FLAG_LOG_DATA:
        data_log_episode = {
            "start_time": datetime.now().strftime("%d.%m.%Y:%H:%M:%S"),
            "steps": {},
            "summary": {
                "steps": 0,
                "terminated": False,
                "truncated": False,
                "episode_reward": 0,
                "classification_accuracy_avg": 0,
                "classification_accuracy_max": 0,
                "classification_accuracy_min": 0,
                "classification_accuracy_last": 0,  # Classification accuracy of last valid step
                "observation": [],
                "circuit": [],
                "circuit_depth": 0,
                "weights": []
            },
            "end_time": None
        }
    else:
        data_log_episode = {}

    return data_log_episode


def initialize_data_log_step():
    """
    Initializes the data log dictionary for a new step.
    """
    if c.FLAG_LOG_DATA:
        data_log_step = {
            "start_time": datetime.now().strftime("%d.%m.%Y:%H:%M:%S"),
            "action": None,
            "gate": None,
            "illegal_actions": [],
            "reward": 0,
            "cumulative_reward": 0,
            "classification_accuracy": None,
            "terminated": False,
            "truncated": False,
            "end_time": None
        }
    else:
        data_log_step = {}
    
    return data_log_step
    

def update_data_log_run(data_log_run: dict, data_log_episode: dict, episode_idx: int):
    """
    Updates the data log dictionary for a run with the data log dictionary for an episode.
    """
    if c.FLAG_LOG_DATA and c.LOG_DATA_LEVEL != "run":
        new_episode_key = "episode_" + str(episode_idx)
        data_log_run["episodes"][new_episode_key] = data_log_episode
        
        
def update_data_log_run_summary(data_log_run: dict, **kwargs):
    """
    Updates the summary of data_log_run dictionary if run is terminated or truncated.
    """
    if c.FLAG_LOG_DATA:
        
        episodes_num_prev = data_log_run["summary"]["episodes"]
        data_log_run["summary"]["episodes"] +=  1
        episodes_num_post = data_log_run["summary"]["episodes"]
        
        for key, value in kwargs.items():
            # ! Has to be updated / adapted according to used reward shaping (e.g., max / min calculation) in order for values to be updated correctly
            if key == "steps":
                data_log_run["summary"]["steps"] += value
                data_log_run["summary"]["steps_episode_avg"] = (data_log_run["summary"]["steps_episode_avg"] * episodes_num_prev + value) / episodes_num_post
                data_log_run["summary"]["steps_episode_max"] = max(data_log_run["summary"]["steps_episode_max"], value)
                if episodes_num_prev == 0:
                    data_log_run["summary"]["steps_episode_min"] = value
                else:
                    data_log_run["summary"]["steps_episode_min"] = min(data_log_run["summary"]["steps_episode_min"], value)
            elif key == "reward":
                data_log_run["summary"]["reward_episode_avg"] = (data_log_run["summary"]["reward_episode_avg"] * episodes_num_prev + value) / episodes_num_post
                if episodes_num_prev == 0:
                    data_log_run["summary"]["reward_episode_max"] = value
                else:
                    data_log_run["summary"]["reward_episode_max"] = max(data_log_run["summary"]["reward_episode_max"], value)
                if episodes_num_prev == 0:
                    data_log_run["summary"]["reward_episode_min"] = value
                else:
                    data_log_run["summary"]["reward_episode_min"] = min(data_log_run["summary"]["reward_episode_min"], value)
            elif key == "accuracy_episode_avg":
                data_log_run["summary"]["accuracy_episode_avg"] = (data_log_run["summary"]["accuracy_episode_avg"] * episodes_num_prev + value) / episodes_num_post
            elif key == "accuracy_episode_max":
                data_log_run["summary"]["accuracy_episode_max"] = max(data_log_run["summary"]["accuracy_episode_max"], value)
            elif key == "accuracy_episode_min":
                if episodes_num_prev == 0:
                    data_log_run["summary"]["accuracy_episode_min"] = value
                else:
                    data_log_run["summary"]["accuracy_episode_min"] = min(data_log_run["summary"]["accuracy_episode_min"], value)
            elif key == "circuit_depth":
                data_log_run["summary"]["circuit_depth_avg"] = (data_log_run["summary"]["circuit_depth_avg"] * episodes_num_prev + value) / episodes_num_post
                data_log_run["summary"]["circuit_depth_max"] = max(data_log_run["summary"]["circuit_depth_max"], value)
                if episodes_num_prev == 0:
                    data_log_run["summary"]["circuit_depth_min"] = value
                else:
                    data_log_run["summary"]["circuit_depth_min"] = min(data_log_run["summary"]["circuit_depth_min"], value)
            else:
                raise KeyError(f"Key '{key}' does not exist in the dictionary.")
            
        data_log_run["end_time"] = datetime.now().strftime("%d.%m.%Y:%H:%M:%S")
    

def update_data_log_episode(data_log_episode: dict, data_log_step: dict, step_idx: int):
    """
    Updates the data log dictionary for an episode with the data log dictionary for a step.
    """
    if c.FLAG_LOG_DATA: 
        if c.LOG_DATA_LEVEL == "step":
            new_step_key = "step_" + str(step_idx)
            data_log_episode["steps"][new_step_key] = data_log_step
        
        # Updated with each step to avoid iteration over all steps  
        steps_num_prev = data_log_episode["summary"]["steps"]
        data_log_episode["summary"]["steps"] += 1
        steps_num_post = data_log_episode["summary"]["steps"]
        
        # Update episode summary with new step data if step is not illegal action
        if data_log_step["classification_accuracy"] is not None:
            data_log_episode["summary"]["classification_accuracy_avg"] = (data_log_episode["summary"]["classification_accuracy_avg"] * steps_num_prev + data_log_step["classification_accuracy"]) / steps_num_post
            data_log_episode["summary"]["classification_accuracy_max"] = max(data_log_episode["summary"]["classification_accuracy_max"], data_log_step["classification_accuracy"])
            if steps_num_prev == 0:
                data_log_episode["summary"]["classification_accuracy_min"] = data_log_step["classification_accuracy"]
            else:
                data_log_episode["summary"]["classification_accuracy_min"] = min(data_log_episode["summary"]["classification_accuracy_min"], data_log_step["classification_accuracy"])
            data_log_episode["summary"]["classification_accuracy_last"] = data_log_step["classification_accuracy"]
            data_log_episode["summary"]["circuit"].append(data_log_step["gate"])


def update_data_log_episode_summary(data_log_episode: dict, **kwargs):
    """
    Updates the summary of data_log_episode dictionary if episode is terminated or truncated.
    """
    if c.FLAG_LOG_DATA and c.LOG_DATA_LEVEL != "run":
        for key, value in kwargs.items():
            if key in data_log_episode["summary"]:
                data_log_episode["summary"][key] = value
            else:
                raise KeyError(f"Key '{key}' does not exist in the dictionary.")
            
        data_log_episode["end_time"] = datetime.now().strftime("%d.%m.%Y:%H:%M:%S")


def update_data_log_step(data_log_step: dict, **kwargs):
    """
    Updates the data log dictionary for a step with the given data.
    """
    # if 'a' not in kwargs:
    #    raise KeyError("Key 'a' is required but not provided.")

    if c.FLAG_LOG_DATA:
        for key, value in kwargs.items():
            if key in data_log_step:
                data_log_step[key] = value
            else:
                raise KeyError(f"Key '{key}' does not exist in the dictionary.")


def console_log(message: str, flag: bool = c.FLAG_LOG_CLI):
    """
    Logs a message to the console.
    """
    if flag:
        print("[Log] " + datetime.now().time().strftime('%H:%M:%S') + ": " + message)
        
        
def save_data_log(path: str, data_log: dict):
    """
    Saves the data log to a JSON file.
    """
    if c.FLAG_LOG_DATA:
        with open(path,'w') as f:
            json.dump(data_log, f, indent=4)
        console_log(f"Data log saved as JSON file at '{path}'.")
