# QAS4ML: Quantum Architecture Search for Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.40.0-orange.svg)](https://pennylane.ai)
[![JAX](https://img.shields.io/badge/JAX-0.4.34-red.svg)](https://jax.readthedocs.io)

A reinforcement learning framework for automatically discovering optimal quantum circuit architectures for machine learning classification tasks. This project uses deep reinforcement learning to search the space of variational quantum circuits (VQCs) and find architectures that maximize classification performance.

## 🌟 Features

- **Automated Quantum Circuit Design**: Uses RL agents (PPO, A2C) to automatically construct quantum circuits
- **Multiple Classification Datasets**: Support for Iris, Wine, and MNIST digit classification
- **Flexible Quantum Gates**: Utilizes RX, RY, RZ rotation gates and CNOT entangling gates
- **Performance Optimization**: JAX-based optimization with multiple optimizers (Adam, SGD, Adagrad, etc.)
- **Comprehensive Analysis**: Built-in tools for circuit analysis, visualization, and benchmarking
- **Caching System**: Efficient caching to avoid redundant circuit evaluations
- **Extensive Logging**: Detailed logging and data collection for experiment tracking

## 🏗️ Architecture

The framework consists of several key components:

### Core Components

- **`rl_env_cl.py`**: Custom Gymnasium environment for quantum circuit construction
- **`main.py`**: Entry point for training RL agents or generating random baselines
- **`run.py`**: Training orchestration and model management
- **`config.py`**: Centralized configuration with command-line argument parsing

### Data Processing & Analysis

- **`ml_classification.py`**: Dataset preprocessing for Iris, Wine, and MNIST
- **`analyse.py`**: Circuit analysis, benchmarking, and visualization tools
- **`plot.py`**: Plotting utilities for training metrics and circuit properties

### Utilities

- **`helpers.py`**: Helper functions for circuit parsing, caching, and file operations
- **`log.py`**: Logging infrastructure for experiments and data collection

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

Train a PPO agent to find optimal quantum circuits for Iris classification:

```bash
python src/main.py --ct iris --cn 0 --rn 1 --cd 4 --rm 0 --ts 150000 --ep 2 --st 1024 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 5 --mm minimum --alg ppo
```

Generate a random baseline for comparison:

```bash
python src/main.py --ct iris --cn 0 --rn 1 --cd 4 --rm 0 --ts 150000 --ep 2 --st 1024 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 5 --mm minimum --alg random
```

## 📊 Supported Datasets

| Dataset         | Classes | Features          | Qubits Required |
| --------------- | ------- | ----------------- | --------------- |
| **Iris**  | 2 or 3  | 4                 | 2-3             |
| **Wine**  | 2 or 3  | 13 (padded to 16) | 4               |
| **MNIST** | 2-10    | 64 (PCA reduced)  | Variable        |

## ⚙️ Configuration

Key configuration parameters:

### Reinforcement Learning

- `--alg`: RL algorithm (`ppo`, `a2c`, `random`)
- `--ts`: Timesteps per episode
- `--ep`: Number of episodes
- `--lr`: Learning rate
- `--ga`: Discount factor (gamma)

### Quantum Circuit

- `--cd`: Maximum circuit depth
- `--gpm`: Gate parameter initialization (`random`, `static`)
- `--mm`: Measurement mode (`minimum`, `all`)

### Optimization

- `--opt`: Optimizer (`adam`, `sgd`, `adagrad`, `rmsprop`, `nadam`)
- `--olr`: Optimization learning rate
- `--oep`: Optimization epochs
- `--bm`: Batch mode (`fixed`, `window`, `list`)

## 📈 Reward Functions

The framework supports multiple reward functions to guide the RL agent:

- **Performance-based**: Rewards based solely on classification accuracy
- **Performance + Complexity**: Balances accuracy with circuit complexity
- **Delta-based**: Rewards improvements over previous performance
