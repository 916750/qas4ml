"""Helper functions"""

import json
import os
import random
from typing import Literal

import config as c
import log as l
import pennylane as qml
import rl_env_cl as rl_cl
from filelock import FileLock


def create_directory(path: str):
    """
    Creates a directory if it does not exist.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        l.console_log(f"Directory '{directory}' was created.")
    else:
        l.console_log(f"Directory '{directory}' already exists. No creation required.")


def parse_circuit(json_file_path, key_name: Literal["episode", "circuit", "architecture"], number, weights_mode: Literal["random", "static", "json", "argument"], weights_arg=None):
    """
    Parses a JSON file to extract a quantum circuit and its parameters.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if key_name == "episode":
            if "episodes" not in data:
                raise ValueError("The JSON file does not contain the 'episodes' key.")

            episode_key = f"episode_{number}"

            if episode_key not in data["episodes"]:
                raise ValueError(f"Episode {number} not found in the JSON file.")

            episode_data = data["episodes"][episode_key]

            if "summary" not in episode_data:
                raise ValueError(f"Episode {number} does not contain 'summary' key.")

            summary = episode_data["summary"]

            if "circuit" not in summary:
                raise ValueError(f"The 'summary' object in episode {number} does not contain 'circuit' key.")

            circuit = summary["circuit"]
            weights_json = summary["weights"]

        elif key_name == "circuit":
            circuit_key = f"circuit_{number}"

            if circuit_key not in data:
                raise ValueError(f"Circuit {number} not found in the JSON file.")

            circuit = data[circuit_key]["circuit"]
            weights_json = data[circuit_key]["weights"]
            
        elif key_name == "architecture": 
            circuit_key = f"circuit_{number}"
            
            if circuit_key not in data["circuits"]:
                raise ValueError(f"Circuit {number} not found in the JSON file.")
            circuit = data["circuits"][circuit_key]["architecture"]
            weights_json = data["circuits"][circuit_key]["optimized_weights"]
            

        # Initialize weights based on the weights_mode
        if weights_mode == "json":
            weights = weights_json
        elif weights_mode == "random":
            num_params = sum(1 for gate in circuit if gate.split(" ")[0] in ["RX", "RY", "RZ"])
            weights = [random.uniform(-qml.math.pi, qml.math.pi) for _ in range(num_params)]
        elif weights_mode == "static":
            num_params = sum(1 for gate in circuit if gate.split(" ")[0] in ["RX", "RY", "RZ"])
            weights = [0.01] * num_params
        elif weights_mode == "argument":
            if weights_arg is None:
                raise ValueError("weights must be provided when weights_mode is 'argument'")
            num_params = sum(1 for gate in circuit if gate.split(" ")[0] in ["RX", "RY", "RZ"])
            if len(weights_arg) != num_params:
                raise ValueError(f"Expected {num_params} weights, but got {len(weights_arg)}")
            weights = weights_arg
        else:
            raise ValueError(f"Invalid weights_mode: {weights_mode}")

        # Define the quantum circuit as a PennyLane function
        def quantum_circuit(params):
            param_idx = 0
            for gate in circuit:
                if "(" in gate and ")" in gate:
                    gate_name = gate.split(" ")[0]
                    qubits = gate[gate.index("(") + 1:gate.index(")")]
                    qubit_indices = [int(q) for q in qubits.split(",")]

                    # Map gate names to PennyLane operations
                    if gate_name == "CNOT":
                        qml.CNOT(wires=qubit_indices)
                    elif gate_name == "RX":
                        qml.RX(params[param_idx], wires=qubit_indices[0])
                        param_idx += 1
                    elif gate_name == "RY":
                        qml.RY(params[param_idx], wires=qubit_indices[0])
                        param_idx += 1
                    elif gate_name == "RZ":
                        qml.RZ(params[param_idx], wires=qubit_indices[0])
                        param_idx += 1
                    else:
                        raise ValueError(f"Unsupported gate: {gate_name}")

        return quantum_circuit(weights)

    except FileNotFoundError:
        print(f"Error: File not found at path {json_file_path}.")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Ensure it is properly formatted.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def save_dictionary_as_json(dictionary, json_file_path):
    """
    Saves a dictionary as a JSON file.
    """
    try:
        with open(json_file_path, 'w') as file:
            json.dump(dictionary, file, indent=4)
        l.console_log(f"Dictionary saved as JSON file at path: {json_file_path}")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while saving dictionary as JSON file: {e}")


def get_rl_hyperparameters():
    """
    Returns the hyperparameters for the reinforcement learning algorithm set in config.py.
    """
    if c.RL_ALG == "ppo":
        return {k: v for k, v in c.PPO_HYPERPARAMETERS.items() if v is not None}
    elif c.RL_ALG == "a2c":
        return {k: v for k, v in c.A2C_HYPERPARAMETERS.items() if v is not None}
    else:
        raise ValueError(f"Invalid RL algorithm: {c.RL_ALG}")
    
    
def get_opt_learning_rate(batch_size: int, base_lr: float = 0.0005, base_batch_size: int = 16):
    """
    Calculates the learning rate based on the batch size according to the Lineare Scaling Rule
    """
    return base_lr * (batch_size / base_batch_size)


class Cache: 
    
    def __init__(self):
        self.file_path_cache = c.PATH_CACHE + ".json"
        self.file_path_lock = c.PATH_CACHE + ".lock"
        self.cache = {}
        self.new_values_count = 0
        self.load_cache()
                
    def load_cache(self):
        with FileLock(self.file_path_lock):
            if os.path.exists(self.file_path_cache):
                with open(self.file_path_cache, 'r') as f:
                    try:
                        self.cache = json.load(f)
                        l.console_log(f"Cache successfully loaded from {self.file_path_cache}")
                    except json.JSONDecodeError:
                        l.console_log(f"Failed to load cache from {self.file_path_cache}. Creating new cache.")
                        self.cache = {}
                        
    def save_cache(self): 
        with FileLock(self.file_path_lock):
            if os.path.exists(self.file_path_cache):
                # Merge with the latest version to avoid overwriting
                with open(self.file_path_cache, 'r') as f:
                    current_data = json.load(f)
                current_data.update(self.cache)
                self.cache = current_data
                l.console_log(f"Cache {self.file_path_cache} successfully updated.")
            else: 
                l.console_log(f"No existing cache found. Creating new cache {self.file_path_cache}.")
                
            with open(self.file_path_cache, 'w') as f:
                    json.dump(self.cache, f, indent=4)
            
        # Reset the new values count
        self.new_values_count = 0
                
    def set(self, key, value):
        if key not in self.cache:
            self.cache[key] = value
            self.new_values_count += 1  # Increment for new values only
            # Check if synchronization is needed
            if self.new_values_count >= c.CACHE_THRESHOLD:
                self.synchronize()
    
    def get(self, key):
        return self.cache.get(key)
    
    def synchronize(self): 
        l.console_log(f"Synchronizing cache {self.file_path_cache}...")
        self.save_cache()
        self.load_cache()
        l.console_log(f"Cache {self.file_path_cache} synchronized.")
        

def stronglyEntanglingLayers(weights, num_layers, num_qubits):
    """
    Custom version of qml.StronglyEntanglingLayers; uses list of weights instead of a tensor
    
    Official PennyLane implementation: https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html
    """
    assert qml.math.shape(weights)[0] == num_layers * num_qubits * 3, \
        "The length of weights must be num_layers * num_qubits * 3."

    weight_index = 0  # To track the current index in the weights list

    for layer in range(num_layers):
        # Apply parameterized RX, RY, and RZ gates to each qubit
        for qubit in range(num_qubits):
            qml.RX(weights[weight_index], wires=qubit)
            weight_index += 1
            qml.RY(weights[weight_index], wires=qubit)
            weight_index += 1
            qml.RZ(weights[weight_index], wires=qubit)
            weight_index += 1
        
        # Apply CNOT gates for entanglement
        for qubit in range(num_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % num_qubits])  # Connect each qubit to the next in a ring topology
            
            
def create_circuit_tensor(env, data):
    """
    Parases JSON circuit representation and converts it into corresponding observation (3D binary tensor)
    """
    
    for gate_string in data["circuit"]:
        gate_name = gate_string.split(" ")[0]
        gate_idx = env.gate_set_names.index(gate_name)
        gate = env.gate_set[gate_idx]
        wires = [int(w) for w in gate_string.split("(")[1].strip(")").split(",")]
        
        if gate == qml.CNOT:
            max_qubit_depth_idx = max(env.qubit_depth_idx[wires[0]], env.qubit_depth_idx[wires[1]])
            env.observation[max_qubit_depth_idx, wires[0], gate_idx + wires[0]] = 1
            env.observation[max_qubit_depth_idx, wires[1], gate_idx + wires[0]] = 1
            env.qubit_depth_idx[wires[0]] = max_qubit_depth_idx + 1
            env.qubit_depth_idx[wires[1]] = max_qubit_depth_idx + 1
        else: 
            env.observation[env.qubit_depth_idx[wires[0]], wires[0], gate_idx] = 1
            env.qubit_depth_idx[wires[0]] += 1


def get_circuit_hash(env, data):
    """
    Returns the hash value of the current circuit. The hash value is a unique identifier for the circuit, which can be used to check if the circuit has already been seen.
    The hash value is generated by hashing the observation tensor of the environment object which is updated based on the JSON circuit representation before.
    """
    
    create_circuit_tensor(env, data)        # Update observation tensor of env object with current circuit data 
    circuit_hash = env.hash_observation()   # Get hash value of the current circuit
    env.reset()                             # Reset env object for next circuit
    return circuit_hash   
