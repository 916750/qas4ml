"""
Configuration file for the RL QAS framework
"""

import argparse

# Example for passing arguments via command line:
# python .\main.py --ct 'iris' --cn 0 --rn 1 --cd 4 --rm 0 --ts 150000 --ep 2 --st 1024 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm 'fixed' --obs_win 1 9 --obs_list 2 --obs_fix 20 --br 'false' --brt 0.95 --olr 0.01 --opt 'adam' --oep 1000 --gpm 'random' --gpv 1.0 --gps 5 --mm 'minimum' --alg 'random'


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration for the QAS RL Env")
    parser.add_argument("--ct", type=str, required=True, help="Classification task")
    parser.add_argument(
        "--cn", type=int, required=True, help="Number of hyperparameter configuration"
    )
    parser.add_argument("--rn", type=int, required=True, help="Number of current run")
    parser.add_argument("--cd", type=int, required=True, help="Maximum circuit depth")
    parser.add_argument(
        "--rm", type=int, required=True, help="Index of Reward mode for RL algorithm"
    )
    parser.add_argument(
        "--ts", type=int, required=True, help="Number of timesteps per episode"
    )
    parser.add_argument(
        "--ep", type=int, required=True, help="Number of episodes to run"
    )
    parser.add_argument(
        "--st", type=int, required=True, help="n_steps for RL algorithm"
    )
    parser.add_argument(
        "--bs", type=int, required=True, help="Batch size for RL algorithm"
    )
    parser.add_argument(
        "--lr", type=float, required=True, help="Learning rate for rl algorithm"
    )
    parser.add_argument(
        "--ga", type=float, required=True, help="Gamma for rl algorithm"
    )
    parser.add_argument(
        "--cr", type=float, required=True, help="Clip range for rl algorithm"
    )
    parser.add_argument(
        "--ec", type=float, required=True, help="Entropy coefficient for rl algorithm"
    )
    parser.add_argument(
        "--vf",
        type=float,
        required=True,
        help="Value function coefficient for rl algorithm",
    )
    parser.add_argument(
        "--na", type=int, required=True, help="Index of net_arch configuration"
    )
    parser.add_argument(
        "--bm", type=str, required=True, help="Batch mode for parameter optimization"
    )
    parser.add_argument(
        "--obs_win",
        type=int,
        nargs=2,
        required=True,
        help="Batch size window for optimization",
    )
    parser.add_argument(
        "--obs_list",
        type=int,
        required=True,
        help="Index for batch list for param optimization",
    )
    parser.add_argument(
        "--obs_fix",
        type=int,
        required=True,
        help="Fixed batch size for param optimization",
    )
    parser.add_argument(
        "--br",
        type=str,
        required=True,
        help="Bonus reward for performance above threshold",
    )
    parser.add_argument(
        "--brt", type=float, required=True, help="Threshold for bonus reward"
    )
    parser.add_argument(
        "--olr",
        type=float,
        required=True,
        help="Optimizer learning rate for param optimization",
    )
    parser.add_argument(
        "--opt", type=str, required=True, help="Optimizer for parameter optimization"
    )
    parser.add_argument(
        "--oep",
        type=int,
        required=True,
        help="Number of epochs for parameter optimization",
    )
    parser.add_argument(
        "--gpm",
        type=str,
        required=True,
        help="Parameter initialization mode for quantum gates",
    )
    parser.add_argument(
        "--gpv",
        type=float,
        required=True,
        help="Parameter value for initialization of quantum gates. If mode 'random', value is the range for random initialization. If mode 'static', value is the static parameter value.",
    )
    parser.add_argument(
        "--gps",
        type=int,
        required=True,
        help="Number of seeds for parameter initialization",
    )
    parser.add_argument(
        "--mm", type=str, required=True, help="Measurement mode for quantum circuit"
    )
    parser.add_argument("--alg", type=str, required=True, help="RL Algorithm")
    args = parser.parse_args()
    return args


args = parse_arguments()


####
## General constants
####

DEVICE = "thinkpad workstation"  # Set this accordningly if you have other devices executing this program on
COMP_UNIT = "CPU"  # No nother options implemented yet
CONFIG_NR = (
    args.cn
)  # Configuration number for logging and saving data; increment for new configuration
RUN_NR = args.rn

####
## Logging
####
FLAG_LOG_CLI = True
FLAG_LOG_DATA = True
LOG_DATA_LEVEL = "episode"  # Valid values: [step, episode, run]


####
## Constants reinforcement learning environment
####
SEED = (
    41 + RUN_NR
)  # Seed for RL algorithm and splitting classification data in train and test set;
# None for no / random seed; seed used for reproducibility; run_nr used so every run has different seed
ACTION_SPACE = ["RX", "RY", "RZ", "CNOT"]
PENALTY_ILLEGAL_ACTION = -0.01
TIMESTEPS = (
    args.ts
)  # Number of timesteps per episode; every TIMESTEPS model and data log is saved
EPISODES = (
    args.ep
)  # Number of episodes to run; training will run for EPISODES * TIMESTEPS timesteps

REWARD_MODES = [
    "performance_complexity",
    "performance_complexity_positive",
    "performance_complexity_positive_max",
    "performance_complexity_scaled",
    "performance",
    "performance_delta",
    "performance_delta_positive",
]
REWARD_MODE = REWARD_MODES[args.rm]  # Valid values:
#   - Performance and Complexity: [performance_complexity, performance_complexity_positive, performance_complexity_positive_max, performance_complexity_scaled]
#   - Performance only: [performance, performance_delta, performance_delta_positive]
RL_ALG = args.alg  # Valid values: [ppo,a2c,random]


####
## RL algorithm hyperparameters
####

# Constants PPO; set value to None to use SB3 default
# Official Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

NET_ARCH_CONFIGS = [[64, 64], [128, 128], [256, 256]]

PPO_HYPERPARAMETERS = {
    "learning_rate": args.lr,  # Learning rate for the policy network
    "n_steps": args.st,  # Number of steps to run per environment per update; larger values lead to lower variance in the policy gradient estimate
    "batch_size": args.bs,  # Number of samples per batch; smaller batch sizes may lead to faster convergence but less stability
    "n_epochs": 10,  # Number of epochs when optimizing the surrogate objective; larger values lead to more stable training but slower convergence
    "gamma": args.ga,  # Discount factor; larger values lead to more emphasis on future rewards
    "gae_lambda": 0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator; larger values lead to lower bias but higher varianc
    "clip_range": args.cr,  # Clipping parameter for the surrogate objective; larger values lead to more stable training but slower convergence
    "clip_range_vf": None,  # Clipping parameter for the value function; larger values lead to more stable training but slower convergence
    "normalize_advantage": True,  # Normalize advantage function in PPO
    "ent_coef": args.ec,  # Entropy coefficient for the loss calculation; increase it slightly if the agent is overfitting to improve exploration
    "vf_coef": args.vf,  # Value function coefficient for the loss calculation
    "max_grad_norm": 0.5,  # Maximum gradient norm for clipping
    "use_sde": None,  # Use standard deviation for action sampling
    "sde_sample_freq": None,  # Sample frequency for state dependent exploration
    "rollout_buffer_class": None,  # Rollout buffer class
    "rollout_buffer_kwargs": None,  # Rollout buffer kwargs
    "target_kl": None,  # Target KL divergence for the PPO algorithm
    "stats_window_size": None,  # Size of the window for computing the running average of the stats
    "policy_kwargs": dict(
        net_arch=NET_ARCH_CONFIGS[args.na]
    ),  # Policy and value function network architecture; 'net_arch=dict(pi=[256, 128], vf=[256, 128])'
    "verbose": 1,  # Verbosity level
    "seed": SEED,  # Seed for the pseudo-random generators; set for reproducibility
    "device": "auto",  # Device (cpu, cuda, ...) on which the code should be run; Setting it to auto, the code will be run on the GPU if possible
    "init_setup_model": None,  # Whether or not to build the network at the creation of the instance
}

# TODO: Fix data logging bug for A2C 'policy_kwargs' parameter
# Constants A2C; set value to None to use SB3 default
# Official Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
A2C_HYPERPARAMETERS = {
    "learning_rate": args.lr,  # Learning rate for the policy network
    "n_steps": args.st,  # Number of steps to run per environment per update; larger values lead to lower variance in the policy gradient estimate
    "gamma": 0.99,  # Discount factor; larger values lead to more emphasis on future rewards
    "gae_lambda": None,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator; larger values lead to lower bias but higher varianc
    "ent_coef": args.ec,  # Entropy coefficient for the loss calculation; increase it slightly if the agent is overfitting to improve exploration
    "vf_coef": args.vf,  # Value function coefficient for the loss calculation
    "max_grad_norm": 0.5,  # Maximum gradient norm for clipping
    "rms_prop_eps": None,  # RMSProp epsilon. It stabilizes square root computation in denominator of RMSProp update
    "use_rms_prop": None,  # Whether to use RMSprop (default) or Adam as optimizer
    "use_sd": None,  # Use standard deviation for action sampling
    "sde_sample_freq": None,  # Sample frequency for state dependent exploration
    "rollout_buffer_class": None,  # Rollout buffer class
    "rollout_buffer_kwargs": None,  # Rollout buffer kwargs
    "normalize_advantage": None,  # Whether to normalize or not the advantage
    "stats_window_size": None,  # Size of the window for computing the running average of the stats
    "policy_kwargs": None,  # Policy network architecture; 'net_arch=dict(pi=[128, 128], vf=[256, 256])'
    "verbose": 1,  # Verbosity level
    "seed": SEED,  # Seed for the pseudo-random generators; None for no / random seed; [0, 42, 123, 999, 2024] for reproducibility
    "device": "auto",  # Device (cpu, cuda, ...) on which the code should be run; Setting it to auto, the code will be run on the GPU if possible
    "init_setup_model": None,  # Whether or not to build the network at the creation of the instance
}


####
## Constants quantum circuit
####
CIRCUIT_MEASUREMENT_MODE = (
    args.mm
)  # Measurement mode for the quantum circuit. Valid values: [minimum, all]
CIRCUIT_DEPTH_MAX = args.cd  # Maximum allowed depth of the quantum circuit
CIRCUIT_DEPTH_DYNAMIC = False  # Increase allowed circuit depth incrementally with training progress; CIRCUIT_DEPTH_MAX is the maximum depth
CIRCUIT_DEPTH_DYNAMIC_INIT = 3  # Initial circuit depth for dynamic circuit depth
CIRCUIT_DEPTH_DYNAMIC_INCR = 1  # Increment for dynamic circuit depth
CIRCUIT_DEPTH_STAGNATION_THRESHOLD = (
    0.15  # Threshold for stagnation of circuit depth increase
)
CIRCUIT_DEPTH_WINDOW_SIZE = 500  # Number of episodes to monitor for stagnation

GATE_PARAM_MODE = (
    args.gpm
)  # Mode for parameterized quantum gate initialization. Valid values: [random, static]
GATE_PARAM_DEFAULT = args.gpv  # Default parameter for parameterized quantum gate initialization (if GATE_PARAM_MODE = "static")
GATE_PARAM_VALUE_RANGE = args.gpv  # Range for random parameterized quantum gate initialization (if GATE_PARAM_MODE = "random"); [-GATE_PARAM_VALUE_RANGE, GATE_PARAM_VALUE_RANGE]; for pi: 3.1415
GATE_PARAM_SEED_COUNT = (
    args.gps
)  # Number of seeds / runs for parameterized quantum gate initialization

####
## Parameter optimization
####
OPT_JAX = True  # Use JAX for parameter optimization and JIT compilation
OPT = (
    args.opt
)  # Optimizer for weight optimization. Valid values: [adam,nadam,sgd,adagrad,rmsprop]
OPT_DYNAMIC_LEARNING_RATE = False  # Set learning rate dynamically based on batch size according to Linear Scaling Rule
OPT_LEARNING_RATE = (
    args.olr
)  # Static learning rate for optimization if OPT_DYNAMIC_LEARNING_RATE = False
OPT_EPOCHS = args.oep  # Number of epochs for weight optimization
OPT_BATCH_MODE = args.bm  # Batch mode for weight optimization. Valid values: [fixed,fixed_random,window,list,list_random]
# In case of 'fixed_random': set OPT_BATCH_SIZE_WINDOW for min and max batch size for random sampling
# In case of 'list_random': set OPT_BATCH_SIZE_LIST_RANDOM_LEN and OPT_BATCH_SIZE_WINDOW
OPT_BATCH_SIZE = args.obs_fix  # Batch mode 'fixed'; Batch size <= X_train
OPT_BATCH_SIZE_WINDOW = args.obs_win  # Batch mode 'window'; Batch size <= X_train
OPT_BATCH_SIZE_LIST_CONFIGS = [
    [
        15,
        16,
        17,
        18,
        19,
        20,
        26,
        29,
        33,
        34,
        37,
        43,
        44,
        47,
        48,
        83,
        95,
        96,
        97,
        98,
        99,
        100,
    ],
    [26, 29, 33, 34, 37, 43, 44],
    [14, 16, 17, 83, 95, 96, 97, 98, 99, 100],
    [33, 34, 43, 44],
    [95, 96, 97, 98, 99],
    [11, 29, 31, 32],
    [31, 32, 43, 44],
    [31, 32],
]
# Iris batch [26,29,33,34,37,43,44,47,48]; [14,16,17,26,29,33,34,37,43,44,47,48,83,95,96,97]
# MNIST 2 batch [33,34,37,43,47,48,49,53]
# MNIST batch [47,48,53,54,63,64]
OPT_BATCH_SIZE_LIST_IDX = (
    args.obs_list
)  # Index for batch list for parameter optimization
OPT_BATCH_SIZE_LIST = OPT_BATCH_SIZE_LIST_CONFIGS[
    OPT_BATCH_SIZE_LIST_IDX
]  # Batch mode 'list'; Batch sie <= X_train
OPT_BATCH_SIZE_LIST_RANDOM_LEN = (
    4  # Number of random Batch size values for "list_random" mode
)
OPT_USE_VALIDATION_BATCH = (
    False  # Use validation batch for calculating accuracy during optimization
)
OPT_VALIDATION_SET = 0.3  # Fraction of training data set for validating optimized parameters (= calculating accuracy)
OPT_CONV_TOL = (
    1e-6  # Convergence threshold to stop weight optimization; not implemented yet
)


####
## Constants Machine Learning Task
####
ML_TASK = "classification"  # Valid options: [classification, rl]

# Constants classification task
CLASS_TASK = args.ct  # Valid values: [iris,iris_2,mnist,mnist_[2,3,4,5,6,7,8,9]]


####
### Paths
####

#### For Windows OS


### For Linux OS
# PATH_MAIN = "/home/salfers/ba/data"
# PATH_MAIN = "/home/salfers/ba/data/tests"  # ! Path for testing purposes
# PATH_CLASS_DATA = f"/home/salfers/ba/data/classification/{CLASS_TASK}/{CLASS_TASK}_data.pkl"

### Works OS independently
PATH_RUN = (
    f"{PATH_MAIN}/{ML_TASK}/{CLASS_TASK}/rl_alg/{RL_ALG}/config_"
    + str(CONFIG_NR)
    + "/depth_"
    + str(CIRCUIT_DEPTH_MAX)
)
PATH_CACHE = f"{PATH_MAIN}/{ML_TASK}/{CLASS_TASK}/cache"

if GATE_PARAM_MODE == "static":
    path_suffix = str(GATE_PARAM_DEFAULT)
    if CIRCUIT_MEASUREMENT_MODE == "all":
        path_suffix += "_all"
elif GATE_PARAM_MODE == "random":
    path_suffix = str(GATE_PARAM_VALUE_RANGE)
    if CIRCUIT_MEASUREMENT_MODE == "all":
        path_suffix += "_all"

if OPT_BATCH_MODE == "window":
    PATH_RUN += (
        "/batch_win_"
        + str(OPT_BATCH_SIZE_WINDOW[0])
        + "_"
        + str(OPT_BATCH_SIZE_WINDOW[1])
        + "/run_"
        + str(RUN_NR)
    )
    PATH_CACHE += (
        "/cache_batch_win_"
        + str(OPT_BATCH_SIZE_WINDOW[0])
        + "_"
        + str(OPT_BATCH_SIZE_WINDOW[1])
        + "_"
        + OPT
        + "_"
        + str(OPT_LEARNING_RATE)
        + "_"
        + GATE_PARAM_MODE
        + "_"
        + path_suffix
    )
elif OPT_BATCH_MODE == "list":
    PATH_RUN += (
        "/batch_list_idx_" + str(OPT_BATCH_SIZE_LIST_IDX) + "/run_" + str(RUN_NR)
    )
    PATH_CACHE += (
        "/cache_batch_list_idx_"
        + str(OPT_BATCH_SIZE_LIST_IDX)
        + "_"
        + OPT
        + "_"
        + str(OPT_LEARNING_RATE)
        + "_"
        + GATE_PARAM_MODE
        + "_"
        + path_suffix
    )
elif OPT_BATCH_MODE == "fixed":
    PATH_RUN += "/batch_fix_" + str(OPT_BATCH_SIZE) + "/run_" + str(RUN_NR)
    PATH_CACHE += (
        "/cache_batch_fix_"
        + str(OPT_BATCH_SIZE)
        + "_"
        + OPT
        + "_"
        + str(OPT_LEARNING_RATE)
        + "_"
        + GATE_PARAM_MODE
        + "_"
        + path_suffix
    )

CACHE_THRESHOLD = 50  # Number of newly cached circuits before cache is written to disk

# Directory for testing purposes "_" + str(OPT_BATCH_SIZE_WINDOW[1])


PATH_MODEL = "/models/"
PATH_LOG = "/logs/"
PATH_DATA = "/data/"
PATH_PLOT = "/plots/"

PERFORMANCE_TARGET = 1.0  # Episode terminated if reached
BONUS_REWARD = args.br == "true"  # Reward bonus if performance is above threshold
BONUS_REWARD_THRESHOLD = (
    args.brt
)  # Threshold for bonus reward if performance is above threshold
TEST_SIZE = 0.3  # Size of test data set in percent for splitting data into training and test set
TEST_RANDOM = SEED  # Seed for splitting data into training and test set


####
## Subdirectory structure
####
# - classification
#   - iris
#        - cache
#        - cost_landscape
#        - rl_alg
#           - 00_plots
#           - ppo
#               - 00_plots
#               - config_1
#                   - 00_plots
#                       - depth
#                           - batch size
#                               - run_1
#                                   - models                    # RL models
#                                   - logs                      # Tensorboard logs
#                                   - data                      # Data logs
#                                   - plots                     # Plots
#                               - run_2
#                               - run_3
#                               - ...
#                               - run_x
#               - config_2
#               - config_3
#               - ...
#               - config_x
#           - a2c
#   - iris_2
# 	- mnist
# 	- mnist_2
# 	- mnist_3
# 	- ...
# - rl
####
