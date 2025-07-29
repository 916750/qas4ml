"Analysis and benchmarking of the best VQCs developed by the RL Agent"

import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from typing import Literal

import config as c
import helpers as h
import jax
import jax.numpy as jnp
import log as l
import matplotlib
import matplotlib.pyplot as plt
import ml_classification as ml
import numpy as np
import optax
import pennylane as qml
import plotly.graph_objects as go
import rl_env_cl as rl_cl
from jax.experimental import io_callback
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from qutip import Bloch, Qobj, sigmax, sigmay, sigmaz

random.seed(c.SEED)
jax.random.PRNGKey(c.SEED)


def prepare_data_log():
    """
    Prepares the data log for the optimization loop.
    """
    data_log = {
        "circuit": {},
        "config": {
            "optimizer": c.OPT,
            "learning_rate": c.OPT_LEARNING_RATE,
            "epochs": c.OPT_EPOCHS,
            "opt_batch_mode": c.OPT_BATCH_MODE,
            "measurement_mode": c.CIRCUIT_MEASUREMENT_MODE,
            "seed": c.SEED,
        },
        "optimization": {},
    }

    if c.OPT_BATCH_MODE == "fixed":
        data_log["config"]["batch_size"] = c.OPT_BATCH_SIZE
    elif c.OPT_BATCH_MODE == "window":
        data_log["config"]["batch_window"] = c.OPT_BATCH_SIZE_WINDOW
    elif c.OPT_BATCH_MODE == "list":
        data_log["config"]["batch_list"] = c.OPT_BATCH_SIZE_LIST

    return data_log


def map_measurements_to_classes(env, measurements: jnp.ndarray):
    """
    Map probabilities of quantum states to class probabilities by summing over relevant states.

    Args:
        env: RL QAS environment object
        measurements: jnp.ndarray of measurements (probabilities of quantum states)

    Returns:
        jnp.ndarray of class probabilities
    """
    num_states = len(measurements)
    states_per_class = num_states // env.classes_num

    # Initialize probabilities for each class
    class_probs = jnp.zeros(env.classes_num)

    # Sum the probabilities of the measurements for each class
    for class_idx in range(env.classes_num):
        start_idx = class_idx * states_per_class
        end_idx = start_idx + states_per_class
        class_probs = class_probs.at[class_idx].set(
            jnp.sum(measurements[start_idx:end_idx])
        )

    return class_probs


# Method copied from rl_env_cl.py and adapted for analysis purposes
def optimization_loop_jax(env, weights, batch_size, data_log, seed) -> float:
    """
    JAX compatible inner loop of the RL Agent where parameters of the Quantum Circuit are optimized
    """

    def update_data_log(args):
        """
        Updates the data log with the current optimization step data.
        """
        (
            step_idx,
            loss_val,
            grads,
            params,
            acc_train,
            acc_test,
            acc_total,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        ) = args

        if acc_test > best_acc_test or (
            acc_test == best_acc_test and acc_train > best_acc_train
        ):
            best_acc_train = acc_train
            best_acc_test = acc_test
            best_acc_total = acc_total
            best_params = params

            data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"][
                "best_weights"
            ] = {
                "best_step": step_idx.item(),
                "best_params": [p.item() for p in best_params],
                "best_acc_train": best_acc_train.item(),
                "best_acc_test": best_acc_test.item(),
                "best_acc_total": best_acc_total.item(),
            }

        data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"][
            f"step_{step_idx.item()}"
        ] = {
            "loss": loss_val.item(),
            # "grads": grads_serializable,
            "params": [p.item() for p in best_params],
            "acc_train": best_acc_train.item(),
            "acc_test": best_acc_test.item(),
            "acc_total": best_acc_total.item(),
        }

        return (
            step_idx,
            loss_val,
            grads,
            params,
            acc_train,
            acc_test,
            acc_total,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        )

    @jax.jit
    def cost_jax(params, data, targets):
        def single_example_loss(params, x, y):
            probs = env.full_circuit(params, x)
            probs = map_measurements_to_classes(env, probs)
            return -jnp.sum(y * jnp.log(probs))

        loss = jax.vmap(single_example_loss, in_axes=(None, 0, 0))(
            params, data, targets
        )
        return jnp.mean(loss)

    @jax.jit
    def cost_batch_jax(params, X, Y):
        idx = jax.random.choice(
            jax.random.PRNGKey(0), len(X), (batch_size,), replace=False
        )
        X_batch = X[idx]
        Y_batch = Y[idx]

        return cost_jax(params, X_batch, Y_batch)

    @jax.jit
    def update_step(i, args):
        (
            params,
            opt_state,
            data,
            targets,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        ) = args

        loss_val, grads = jax.value_and_grad(cost_batch_jax)(params, data, targets)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        acc_train = calculate_performance_metric(env, params, mode="train")
        acc_test = calculate_performance_metric(env, params, mode="test")
        acc_total = (acc_train + acc_test) / 2

        data_log_args = (
            i,
            loss_val,
            grads,
            params,
            acc_train,
            acc_test,
            acc_total,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        )
        data_log_return = io_callback(update_data_log, data_log_args, data_log_args)
        (
            i,
            loss_val,
            grads,
            params,
            acc_train,
            acc_test,
            acc_total,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        ) = data_log_return

        # jax.debug.print("Step: {i} | Loss: {loss_val} | Params: {params} | Accuracy train: {acc_train} | Accuracy test: {acc_test} | Accuracy total: {acc_total}",
        #                 i=i, loss_val=loss_val, params=params, acc_train=acc_train, acc_test=acc_test, acc_total=acc_total)

        def print_best_results():
            jax.debug.print(
                "Best params: {best_params} | Best accuracy train: {best_acc_train} | Best accuracy test: {best_acc_test} | Best accuracy total: {best_acc_total}",
                best_params=best_params,
                best_acc_train=best_acc_train,
                best_acc_test=best_acc_test,
                best_acc_total=best_acc_total,
            )

        # jax.lax.cond((jnp.mod(i, c.OPT_EPOCHS-1) == 0), print_best_results, lambda: None)
        jax.lax.cond((i == c.OPT_EPOCHS - 1), print_best_results, lambda: None)

        return (
            params,
            opt_state,
            data,
            targets,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        )

    @jax.jit
    def optimization_jit(args):
        results = jax.lax.fori_loop(0, c.OPT_EPOCHS, update_step, args)
        (
            params,
            opt_state,
            data,
            targets,
            best_params,
            best_acc_train,
            best_acc_test,
            best_acc_total,
        ) = results

        return params, best_params, best_acc_train, best_acc_test, best_acc_total

    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,
        "adagrad": optax.adagrad,
        "rmsprop": optax.rmsprop,
        "nadam": optax.nadam,
    }

    if c.OPT_DYNAMIC_LEARNING_RATE:
        opt = optimizers.get(c.OPT, optax.adam)(h.get_opt_learning_rate(batch_size))
    else:
        opt = optimizers.get(c.OPT, optax.adam)(c.OPT_LEARNING_RATE)

    data = env.X_train_jax
    targets = env.y_train_jax
    params_jax = jnp.array(weights)
    best_params = jnp.array(weights)
    opt_state = opt.init(params_jax)
    best_acc_train = float(0)
    best_acc_test = float(0)
    best_acc_total = float(0)
    args = (
        params_jax,
        opt_state,
        data,
        targets,
        best_params,
        best_acc_train,
        best_acc_test,
        best_acc_total,
    )

    params, best_params, best_acc_train, best_acc_test, best_acc_total = (
        optimization_jit(args)
    )
    params_final = [p.item() for p in params]

    return params_final


# Method copied and adapted from rl_env_cl.py
def calculate_performance_metric(
    env, weights, mode: Literal["train", "test", "average"]
) -> float:
    @jax.jit
    def calculate_performance_metric_jit(weights):
        def calculate_accuracy(params, X, y):
            measurements = jax.vmap(env.full_circuit, in_axes=(None, 0))(params, X)

            @jax.jit
            def measurements_to_classes_jit(measurements: jnp.ndarray) -> jnp.ndarray:
                relevant_elements = map_measurements_to_classes(env, measurements)
                class_pred = jnp.zeros_like(relevant_elements)
                class_pred = class_pred.at[jnp.argmax(relevant_elements)].set(1)
                return class_pred

            predictions = jax.vmap(measurements_to_classes_jit)(measurements)

            @jax.jit
            def accuracy_jit(y_true, y_pred):
                return jnp.mean(jnp.all(y_true == y_pred, axis=1))

            return accuracy_jit(y, predictions)

        params = jnp.array(weights)

        if mode == "test":
            acc = calculate_accuracy(params, env.X_test_jax, env.y_test_jax)
            return acc
        elif mode == "train":
            acc = calculate_accuracy(params, env.X_train_jax, env.y_train_jax)
            return acc
        elif mode == "average":
            acc_train = calculate_accuracy(params, env.X_train_jax, env.y_train_jax)
            acc_test = calculate_accuracy(params, env.X_test_jax, env.y_test_jax)
            acc_average = (acc_train + acc_test) / 2
            return acc_average

    acc = calculate_performance_metric_jit(weights)
    return acc.astype(float)


def run_optimization_loop(
    target_path: str,
    param_range_bound: float,
    number_seeds: int = 5,
    json_path_circuit: str = None,
    circuit_number: int = None,
):
    """
    Runs the optimization loop for the given environment and weights.
    """
    data_log = prepare_data_log()
    env = rl_cl.QuantumCircuit()
    seeds = [x for x in range(1, number_seeds + 1)]

    if json_path_circuit:
        with open(json_path_circuit, "r") as file:
            data = json.load(file)

        num_weights = data["circuits"][f"circuit_{circuit_number}"]["num_weights"]
    else:
        # ! Set number of weights if circuit hard coded below

        ####
        # For Strongly Entangling Layer VQCs for Benchmark Purposes
        # num_layers = 3
        # num_weights = num_layers * env.qubit_count * 3
        ####

        num_weights = 6

    batch_sizes = {
        "window": range(c.OPT_BATCH_SIZE_WINDOW[0], c.OPT_BATCH_SIZE_WINDOW[1] + 1),
        "list": c.OPT_BATCH_SIZE_LIST,
        "fixed": [c.OPT_BATCH_SIZE],
        "list_random": [
            random.randint(c.OPT_BATCH_SIZE_WINDOW[0], c.OPT_BATCH_SIZE_WINDOW[1])
            for _ in range(c.OPT_BATCH_SIZE_LIST_RANDOM_LEN)
        ],
        "fixed_random": [
            random.randint(c.OPT_BATCH_SIZE_WINDOW[0], c.OPT_BATCH_SIZE_WINDOW[1])
        ],
    }.get(c.OPT_BATCH_MODE)

    if json_path_circuit:
        data_log["circuit"] = data["circuits"][f"circuit_{circuit_number}"][
            "architecture"
        ]
    else:
        ####
        # 1 Strongly Entangling Layer for Iris with 2 qubits
        # "RX (0, 1)",
        # "RX (1, 0)",
        # "RY (0, 1)",
        # "RY (1, 0)",
        # "RZ (0, 1)",
        # "RZ (1, 0)",
        # "CNOT (0, 1)",
        # "CNOT (1, 0)"
        ####

        ####
        # 1 Strongly Entangling Layer for MNIST 2 with 5 qubits
        # "RX (0, 1)",
        # "RX (1, 0)",
        # "RX (2, 0)",
        # "RX (3, 0)",
        # "RX (4, 0)",
        # "RY (0, 1)",
        # "RY (1, 0)",
        # "RY (2, 0)",
        # "RY (3, 0)",
        # "RY (4, 0)",
        # "RZ (0, 1)",
        # "RZ (1, 0)",
        # "RZ (2, 0)",
        # "RZ (3, 0)",
        # "RZ (4, 0)",
        # "CNOT (0, 1)",
        # "CNOT (1, 2)",
        # "CNOT (2, 3)",
        # "CNOT (3, 4)",
        # "CNOT (4, 0)"
        ####

        #! Set circuit architecture as list of string elements according to hard coded circuit below
        data_log["circuit"] = [
            "CNOT (1, 3)",
            "RY (3, 2)",
            "CNOT (0, 1)",
            "CNOT (4, 3)",
            "CNOT (2, 4)",
            "RX (0, 1)",
            "RX (1, 0)",
            "RY (3, 1)",
            "CNOT (3, 0)",
            "CNOT (2, 1)",
            "CNOT (0, 3)",
            "RZ (1, 0)",
            "CNOT (4, 3)",
            "RY (0, 2)",
        ]

    def quantum_circuit(params, x):
        qml.AmplitudeEmbedding(
            features=x,
            wires=range(env.qubit_count),
            normalize=False,
            validate_norm=True,
        )

        #! Insert your circuit here by parsing or hardcoding it
        # h.parse_circuit(json_file_path=json_path_circuit,
        #                 key_name="episode",
        #                 number=circuit_number,
        #                 weights_mode="argument",
        #                 weights_arg=params)

        qml.CNOT(wires=[1, 3])
        qml.RY(params[0], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[4, 3])
        qml.CNOT(wires=[2, 4])
        qml.RX(params[1], wires=0)
        qml.RX(params[2], wires=1)
        qml.RY(params[3], wires=3)
        qml.CNOT(wires=[3, 0])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[0, 3])
        qml.RZ(params[4], wires=1)
        qml.CNOT(wires=[4, 3])
        qml.RY(params[5], wires=0)

        ####
        # For Strongly Entangling Layer VQCs for Benchmark Purposes
        # h.stronglyEntanglingLayers(weights=params, num_layers=num_layers, num_qubits=env.qubit_count)
        ####

        return qml.probs(wires=range(env.measurement_qubits))

    env.full_circuit = qml.QNode(quantum_circuit, env.device, interface="jax")

    params_draw = [0.1] * num_weights
    single_data_point = env.X_train_batch[0]
    print(qml.draw(quantum_circuit)(params_draw, single_data_point))

    for batch_size in batch_sizes:
        l.console_log(f"Running optimization loop for batch size {batch_size}")
        data_log["optimization"][f"batch_size_{batch_size}"] = {
            "start_time": datetime.now().strftime("%d.%m.%Y:%H:%M:%S")
        }

        for seed in seeds:
            l.console_log(f"Running optimization loop for seed {seed}")
            data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"] = {
                "start_time": datetime.now().strftime("%d.%m.%Y:%H:%M:%S")
            }
            random.seed(seed)
            if param_range_bound == 3.1415:
                weights_init = [
                    random.uniform(-jnp.pi, jnp.pi) for _ in range(num_weights)
                ]
            else:
                weights_init = [
                    random.uniform(-param_range_bound, param_range_bound)
                    for _ in range(num_weights)
                ]

            # Reset seeds to the run seed
            random.seed(c.SEED)
            jax.random.PRNGKey(c.SEED)

            data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"][
                "best_weights"
            ] = {}
            data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"][
                "weights_init"
            ] = weights_init
            weights_final = optimization_loop_jax(
                env, weights_init, batch_size, data_log, seed
            )
            data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"][
                "weights_final"
            ] = weights_final
            data_log["optimization"][f"batch_size_{batch_size}"][f"seed_{seed}"][
                "end_time"
            ] = datetime.now().strftime("%d.%m.%Y:%H:%M:%S")
            l.console_log(f"Optimization loop for seed {seed} finished")
        data_log["optimization"][f"batch_size_{batch_size}"]["end_time"] = (
            datetime.now().strftime("%d.%m.%Y:%H:%M:%S")
        )

        jax.clear_caches()

        l.console_log(f"Optimization loop for batch size {batch_size} finished")

    h.save_dictionary_as_json(data_log, target_path)
    l.console_log(f"Optimizatiion loop finished. Data log saved to {target_path}")


def get_unique_circuits(
    json_files: list[str],
    target_path: str,
    mode: Literal["random", "performance", "all"],
    number_circuits: int = None,
    performance_threshold: float = None,
):
    """
    Extracts unique circuits from one or more JSON files and saves them to a target JSON file.
    """

    if mode not in ["performance", "random", "all"]:
        raise ValueError("Invalid mode. Must be 'performance', 'random', or 'all'.")
    if mode == "random" and number_circuits is None:
        raise ValueError("Number of circuits must be provided for 'random' mode.")
    if mode == "performance" and performance_threshold is None:
        raise ValueError(
            "Performance threshold must be provided for 'performance' mode."
        )
    if mode == "all" and (
        number_circuits is not None or performance_threshold is not None
    ):
        raise ValueError(
            "Neither 'number_circuits' nor 'performance_threshold' should be provided for 'all' mode."
        )

    env = (
        rl_cl.QuantumCircuit()
    )  # Create Env Object for hash value calculation of circuits
    unique_circuits = {}
    circuit_set = set()  # To track unique circuits
    circuit_count = 1

    try:
        for json_file_path in json_files:
            l.console_log(f"Processing file: {json_file_path}")

            with open(json_file_path, "r") as file:
                data = json.load(file)

            if "episodes" not in data:
                raise ValueError(
                    f"The JSON file {json_file_path} does not contain the 'episodes' key."
                )

            episodes = data["episodes"]

            for episode_key, episode_data in episodes.items():
                if "summary" in episode_data:
                    summary = episode_data["summary"]

                    # Check if all required keys are present
                    required_keys = [
                        "classification_accuracy_last",
                        "circuit",
                        "weights",
                        "steps",
                        "circuit_depth",
                        "episode_reward",
                        "truncated",
                    ]
                    if all(key in summary for key in required_keys):
                        accuracy = summary["classification_accuracy_last"]

                        # Filter circuits based on mode
                        if mode == "performance" and accuracy < performance_threshold:
                            continue

                        circuit_hash = h.get_circuit_hash(
                            env, summary
                        )  # Get hash value of the current circuit

                        if circuit_hash not in circuit_set:
                            circuit_set.add(circuit_hash)
                            gates_value = (
                                summary["steps"] - 1
                                if summary["truncated"]
                                else summary["steps"]
                            )
                            unique_circuits[f"circuit_{circuit_count}"] = {
                                "circuit": summary["circuit"],
                                "weights": summary["weights"],
                                "gates": gates_value,
                                "circuit_depth": summary["circuit_depth"],
                                "reward": summary["episode_reward"],
                                "accuracy": accuracy,
                                "hash": circuit_hash,
                            }
                            circuit_count += 1

        # If mode is "random", randomly select circuits
        if mode == "random":
            unique_circuits = dict(
                random.sample(
                    unique_circuits.items(), min(number_circuits, len(unique_circuits))
                )
            )

        # If mode is "performance", limit the number of circuits if number_circuits is provided
        elif mode == "performance" and number_circuits is not None:
            unique_circuits = dict(list(unique_circuits.items())[:number_circuits])

        # Log the number of unique circuits found
        l.console_log(
            f"Found {len(unique_circuits)} unique circuits in {len(json_files)} files | Mode: {mode} | Number of requested circuits: {number_circuits} | Performance threshold: {performance_threshold}"
        )

        # Save the unique circuits dictionary as a JSON file
        with open(target_path, "w") as file:
            json.dump(unique_circuits, file, indent=4)
        l.console_log(f"Unique circuits saved as JSON file at path: {target_path}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")
    except json.JSONDecodeError:
        raise ValueError(
            "Failed to decode the JSON file. Ensure it is properly formatted."
        )
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def plot_acc_and_loss_opt_epochs(
    figure_path: str,
    qas_json: str = None,
    benchmark_json: str = None,
    compare: bool = False,
    mode: Literal["loss", "train", "test", "all"] = "all",
    epochs: int = 1000,
    batch_size: int = None,
):
    """
    Plots the mean and standard deviation of loss, train accuracy, and test accuracy over optimization epochs
    for a specific VQC. Supports comparison between RL-QAS VQC and SEL VQC.

    Args:
        figure_path (str): Path to save the generated plot as an SVG file.
        qas_json (str): Path to the JSON file containing RL-QAS VQC optimization data.
        benchmark_json (str): Path to the JSON file containing SEL VQC optimization data.
        compare (bool): If True, compares RL-QAS VQC and SEL VQC in the same plot. Defaults to False.
        mode (str): The mode of the plot. Can be "loss", "train", "test", or "all".
        epochs (int): Number of epochs to include in the plot. Default is 1000.
        batch_size (int): The batch size to filter and plot. If None, the first available batch size is used.
    """
    if compare:
        if not qas_json or not benchmark_json:
            raise ValueError(
                "Both 'qas_json' and 'benchmark_json' must be provided when 'compare' is True."
            )
    else:
        if not qas_json and not benchmark_json:
            raise ValueError(
                "Either 'qas_json' or 'benchmark_json' must be provided when 'compare' is False."
            )
        if qas_json and benchmark_json:
            raise ValueError(
                "Only one of 'qas_json' or 'benchmark_json' must be provided when 'compare' is False."
            )

    def process_json(json_path: str, label: str):
        # Load the JSON file
        with open(json_path, "r") as file:
            data = json.load(file)

        # Determine the batch size key
        if batch_size is None:
            batch_size_key = next(iter(data["optimization"].keys()))
        else:
            batch_size_key = f"batch_size_{batch_size}"
            if batch_size_key not in data["optimization"]:
                raise ValueError(f"Batch size {batch_size} not found in the data.")

        seeds = [
            key
            for key in data["optimization"][batch_size_key].keys()
            if key.startswith("seed_")
        ]

        # Initialize containers for the data
        steps = None
        all_loss = []
        all_acc_train = []
        all_acc_test = []

        # Iterate over seeds and collect data
        for seed in seeds:
            seed_data = data["optimization"][batch_size_key][seed]
            step_numbers = sorted(
                [
                    int(step.split("_")[1])
                    for step in seed_data.keys()
                    if step.startswith("step_")
                ]
            )

            # Limit the steps to the specified number of epochs
            step_numbers = step_numbers[:epochs]

            if steps is None:
                steps = step_numbers

            loss = [seed_data[f"step_{step}"]["loss"] for step in step_numbers]
            acc_train = [
                seed_data[f"step_{step}"]["acc_train"] for step in step_numbers
            ]
            acc_test = [seed_data[f"step_{step}"]["acc_test"] for step in step_numbers]

            all_loss.append(loss)
            all_acc_train.append(acc_train)
            all_acc_test.append(acc_test)

        # Convert to numpy arrays for easier manipulation
        all_loss = np.array(all_loss)
        all_acc_train = np.array(all_acc_train)
        all_acc_test = np.array(all_acc_test)

        # Calculate mean and standard deviation
        mean_loss = np.mean(all_loss, axis=0)
        std_loss = np.std(all_loss, axis=0)
        mean_acc_train = np.mean(all_acc_train, axis=0)
        std_acc_train = np.std(all_acc_train, axis=0)
        mean_acc_test = np.mean(all_acc_test, axis=0)
        std_acc_test = np.std(all_acc_test, axis=0)

        # Smooth the data using a moving average
        def moving_average(data, window_size=10):
            return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

        mean_loss_smooth = moving_average(mean_loss)
        mean_acc_train_smooth = moving_average(mean_acc_train)
        mean_acc_test_smooth = moving_average(mean_acc_test)
        std_loss_smooth = moving_average(std_loss)
        std_acc_train_smooth = moving_average(std_acc_train)
        std_acc_test_smooth = moving_average(std_acc_test)
        steps_smooth = steps[: len(mean_loss_smooth)]

        return {
            "steps": steps_smooth,
            "mean_loss": mean_loss_smooth,
            "std_loss": std_loss_smooth,
            "mean_acc_train": mean_acc_train_smooth,
            "std_acc_train": std_acc_train_smooth,
            "mean_acc_test": mean_acc_test_smooth,
            "std_acc_test": std_acc_test_smooth,
            "label": label,
        }

    # Process the JSON files
    plots = []
    if qas_json:
        plots.append(process_json(qas_json, label="RL-QAS VQC"))
    if benchmark_json:
        plots.append(process_json(benchmark_json, label="SEL VQC"))

    # Define colors for each graph
    colors = ["blue", "green", "red", "orange", "purple"]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Add a second y-axis for loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    for idx, plot_data in enumerate(plots):
        steps = plot_data["steps"]
        color = colors[idx % len(colors)]  # Cycle through colors

        if mode == "loss" or mode == "all":
            ax2.plot(
                steps,
                plot_data["mean_loss"],
                label=f"{plot_data['label']} Loss",
                linestyle="--",
                color=color,
                zorder=1,
            )
            ax2.fill_between(
                steps,
                plot_data["mean_loss"] - plot_data["std_loss"],
                plot_data["mean_loss"] + plot_data["std_loss"],
                color=color,
                alpha=0.2,
            )

        if mode == "train" or mode == "all":
            ax1.plot(
                steps,
                plot_data["mean_acc_train"],
                label=f"{plot_data['label']} Train Accuracy",
                color=color,
                zorder=1,
            )
            ax1.fill_between(
                steps,
                np.maximum(
                    plot_data["mean_acc_train"] - plot_data["std_acc_train"], 0.0
                ),
                np.minimum(
                    plot_data["mean_acc_train"] + plot_data["std_acc_train"], 1.0
                ),
                color=color,
                alpha=0.2,
            )

        if mode == "test" or mode == "all":
            ax1.plot(
                steps,
                plot_data["mean_acc_test"],
                label=f"{plot_data['label']} Test Accuracy",
                linestyle=":",
                color=color,
                zorder=1,
            )
            ax1.fill_between(
                steps,
                np.maximum(plot_data["mean_acc_test"] - plot_data["std_acc_test"], 0.0),
                np.minimum(plot_data["mean_acc_test"] + plot_data["std_acc_test"], 1.0),
                color=color,
                alpha=0.2,
            )

    # Add labels, title, and legend
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Combine handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine and sort handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Sort the labels so that "RL-QAS" labels come first, followed by "SEL"
    sorted_indices = sorted(
        range(len(labels)),
        key=lambda i: (not labels[i].startswith("RL-QAS"), labels[i]),
    )
    handles = [handles[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # # Add the legend with the sorted labels
    legend = ax2.legend(
        handles,
        labels,
        loc="lower right",  # Position in der unteren rechten Ecke
        title="VQC",
        ncol=1,  # Eine Spalte
        frameon=True,  # Rahmen um die Legende aktivieren
    )

    # Adapt properties of the legend
    legend.get_frame().set_alpha(
        1.0
    )  # Deckkraft der Legende (1.0 = vollständig sichtbar)
    legend.get_frame().set_facecolor("white")  # Hintergrundfarbe der Legende
    legend.get_frame().set_edgecolor("black")  # Rahmenfarbe der Legende
    legend.set_zorder(10)

    # Ensure the grid and layout are properly adjusted
    ax1.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    plt.tight_layout()

    # Save the plot
    # plt.show()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, format="svg")
    plt.close()
    l.console_log(f"Plot saved to {figure_path}")


def plot_acc_opt_batch_size(file_path_circuit: str, figure_file_path: str):
    """
    Plots the mean and standard deviation of accuracies for different batch sizes from a JSON file.
    """

    with open(file_path_circuit, "r") as file:
        data = json.load(file)

    if "optimization" not in data:
        raise KeyError("Der Key 'optimization' fehlt in der JSON-Datei.")

    optimization_data = data["optimization"]

    batch_sizes = []
    mean_accuracies = []
    std_accuracies = []

    for batch_size_key, seeds in optimization_data.items():
        if not batch_size_key.startswith("batch_size_"):
            continue

        batch_size = int(batch_size_key.split("_")[-1])
        accuracies = []

        for seed_key, seed_data in seeds.items():
            if not seed_key.startswith("seed_"):
                continue

            best_acc_test = seed_data.get("best_weights", {}).get("best_acc_test")
            if best_acc_test is not None:
                accuracies.append(best_acc_test)

        if accuracies:
            batch_sizes.append(batch_size)
            mean_accuracies.append(np.mean(accuracies))
            std_accuracies.append(np.std(accuracies))

    sorted_indices = np.argsort(batch_sizes)
    batch_sizes = np.array(batch_sizes)[sorted_indices]
    mean_accuracies = np.array(mean_accuracies)[sorted_indices]
    std_accuracies = np.array(std_accuracies)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, mean_accuracies, color="blue")
    plt.fill_between(
        batch_sizes,
        np.maximum(mean_accuracies - std_accuracies, 0.0),
        np.minimum(mean_accuracies + std_accuracies, 1.0),
        color="blue",
        alpha=0.2,
        label="Std Deviation",
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
    plt.savefig(figure_file_path, format="svg")
    l.console_log(f"Plot saved to {figure_file_path}")
    plt.close()


def get_best_opt_batch_sizes(json_file_path: str, performance_threshold: float) -> dict:
    """
    Filters batch sizes and seeds from a JSON file where the best_acc_test is greater than or equal to the performance threshold.
    """

    batch_sizes = {}

    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)

        if "optimization" not in data:
            raise KeyError("The key 'optimization' is missing in the JSON file.")

        for batch_size_key, seeds in data["optimization"].items():
            if not batch_size_key.startswith("batch_size_"):
                continue

            batch_size_nr = int(batch_size_key.split("_")[-1])

            valid_seeds = {}

            for seed_key, seed_data in seeds.items():
                if not seed_key.startswith("seed_"):
                    continue

                seed_nr = int(seed_key.split("_")[-1])

                best_weights = seed_data.get("best_weights", {})
                best_acc_test = best_weights.get("best_acc_test")

                if best_acc_test is not None and best_acc_test >= performance_threshold:
                    valid_seeds[seed_nr] = best_acc_test

            if valid_seeds:
                batch_sizes[batch_size_nr] = valid_seeds

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Ensure it is properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return batch_sizes


def plot_run(
    json_file_path: str,
    metric: Literal[
        "episode_reward", "classification_accuracy_last", "steps", "circuit_depth"
    ],
    target_path: str,
):
    """
    Parses a JSON file containing RL training metrics and creates a plot of the average episode_reward over episodes vs episode number.
    """
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)

        if "episodes" not in data:
            raise ValueError(
                "Error: The JSON file does not contain the 'episodes' key."
            )

        episodes = data["episodes"]

        episode_numbers = []
        episode_metrics = []

        for episode_key, episode_data in episodes.items():
            if "summary" in episode_data and f"{metric}" in episode_data["summary"]:
                episode_number = int(episode_key.split("_")[1])
                episode_metric = episode_data["summary"][f"{metric}"]

                episode_numbers.append(episode_number)
                episode_metrics.append(episode_metric)
            else:
                raise ValueError(
                    f"Error: 'summary' or metric '{metric}' not found in episode {episode_key}."
                )

        sorted_indices = sorted(
            range(len(episode_numbers)), key=lambda i: episode_numbers[i]
        )
        episode_numbers = [episode_numbers[i] for i in sorted_indices]
        episode_metrics = [episode_metrics[i] for i in sorted_indices]

        # Decrease / increase window size for less / more smoothing
        window_size = 200
        averaged_metrics = [
            np.mean(episode_metrics[i : i + window_size])
            for i in range(0, len(episode_metrics) - window_size + 1, window_size)
        ]
        averaged_episode_numbers = [
            episode_numbers[i + window_size // 2]
            for i in range(0, len(episode_numbers) - window_size + 1, window_size)
        ]

        metric_labels = {
            "episode_reward": "Reward",
            "classification_accuracy_last": "Accuracy on Test",
            "steps": "Quantum Gates",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric)

        plt.figure(figsize=(10, 6))
        plt.plot(averaged_episode_numbers, averaged_metrics)
        plt.xlabel("Episodes")
        plt.ylabel(f"{metric_label}")
        plt.grid(True)
        plt.tight_layout()
        # figure_file_path = c.PATH_RUN + c.PATH_PLOT + f"{metric}_per_episode.svg"
        figure_file_path = target_path + f"/{metric}_per_episode.svg"
        plt.savefig(figure_file_path, format="svg")
        plt.close()
        l.console_log(f"Plot saved to {figure_file_path}")
        # plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at path {json_file_path}.")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Ensure it is properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_runs(
    metric: Literal[
        "episode_reward", "classification_accuracy_last", "steps", "circuit_depth"
    ],
    target_file_path: str,
    json_file_paths: list[str],
):
    """
    Parses multiple JSON files containing RL training metrics and creates a plot of the mean and standard deviation
    of the specified metric over episodes across all provided JSON files, smoothed over a window of x episodes.
    The parsed data is also saved to a JSON file.
    """
    try:
        all_episode_numbers = []
        all_episode_metrics = []

        # Extract the "config" object from the first JSON file
        with open(json_file_paths[0], "r") as file:
            first_data = json.load(file)
        config_data = first_data.get("config", {})

        for json_file_path in json_file_paths:
            with open(json_file_path, "r") as file:
                data = json.load(file)

            if "episodes" not in data:
                raise ValueError(
                    f"Error: The JSON file {json_file_path} does not contain the 'episodes' key."
                )

            episodes = data["episodes"]

            episode_numbers = []
            episode_metrics = []

            for episode_key, episode_data in episodes.items():
                if "summary" in episode_data and f"{metric}" in episode_data["summary"]:
                    episode_number = int(episode_key.split("_")[1])
                    episode_metric = episode_data["summary"][f"{metric}"]

                    episode_numbers.append(episode_number)
                    episode_metrics.append(episode_metric)
                else:
                    raise ValueError(
                        f"Error: 'summary' or metric '{metric}' not found in episode {episode_key} in file {json_file_path}."
                    )

            sorted_indices = sorted(
                range(len(episode_numbers)), key=lambda i: episode_numbers[i]
            )
            episode_numbers = [episode_numbers[i] for i in sorted_indices]
            episode_metrics = [episode_metrics[i] for i in sorted_indices]

            all_episode_numbers.append(episode_numbers)
            all_episode_metrics.append(episode_metrics)

        # Ensure all episode lists are of the same length
        min_length = min(len(episodes) for episodes in all_episode_numbers)
        all_episode_numbers = [
            episodes[:min_length] for episodes in all_episode_numbers
        ]
        all_episode_metrics = [metrics[:min_length] for metrics in all_episode_metrics]

        # Calculate mean and standard deviation
        mean_metrics = np.mean(all_episode_metrics, axis=0)
        std_metrics = np.std(all_episode_metrics, axis=0)
        episode_numbers = all_episode_numbers[0]

        # Decrease / increase window size for less / more smoothing
        window_size = 200
        smoothed_mean_metrics = [
            np.mean(mean_metrics[i : i + window_size])
            for i in range(0, len(mean_metrics) - window_size + 1, window_size)
        ]
        smoothed_std_metrics = [
            np.mean(std_metrics[i : i + window_size])
            for i in range(0, len(std_metrics) - window_size + 1, window_size)
        ]
        smoothed_episode_numbers = [
            episode_numbers[i + window_size // 2]
            for i in range(0, len(episode_numbers) - window_size + 1, window_size)
        ]

        metric_labels = {
            "episode_reward": "Reward",
            "classification_accuracy_last": "Accuracy on Test",
            "steps": "Quantum Gates",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_episode_numbers, smoothed_mean_metrics)
        if metric == "classification_accuracy_last":
            plt.fill_between(
                smoothed_episode_numbers,
                np.maximum(
                    np.array(smoothed_mean_metrics) - np.array(smoothed_std_metrics),
                    0.0,
                ),
                np.minimum(
                    np.array(smoothed_mean_metrics) + np.array(smoothed_std_metrics),
                    1.0,
                ),
                alpha=0.2,
                label=f"Standard Deviation",
            )
        else:
            plt.fill_between(
                smoothed_episode_numbers,
                np.array(smoothed_mean_metrics) - np.array(smoothed_std_metrics),
                np.array(smoothed_mean_metrics) + np.array(smoothed_std_metrics),
                alpha=0.2,
                label=f"Standard Deviation",
            )
        plt.xlabel("Episodes")
        plt.ylabel(f"{metric_label}")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(target_file_path + "/00_plots", exist_ok=True)

        figure_file_path = (
            target_file_path + f"/00_plots/{metric}_per_episode_mean_std.svg"
        )
        plt.savefig(figure_file_path, format="svg")
        plt.close()
        l.console_log(f"Plot saved to {figure_file_path}")

        parsed_data = {
            "header": config_data,
            "episode_numbers": smoothed_episode_numbers,
            "mean_metrics": smoothed_mean_metrics,
            "std_metrics": smoothed_std_metrics,
        }
        json_target_file = (
            target_file_path + f"/00_plots/{metric}_per_episode_mean_std.json"
        )
        with open(json_target_file, "w") as json_file:
            json.dump(parsed_data, json_file, indent=4)
        l.console_log(f"Parsed data saved to {json_target_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Ensure it is properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_runs_compare_ppo_and_random_agent_episode_based(
    ppo_jsons: list[str], random_jsons: list[str], metric: str, target_path: str
):
    """
    Compares the performance of a PPO Agent and a Random Baseline by plotting the specified metric
    over episodes from multiple JSON files. The results are saved as a combined SVG plot and a JSON file.
    """

    def process_json_files(json_files):
        all_episode_numbers = []
        all_episode_metrics = []

        for json_file_path in json_files:
            with open(json_file_path, "r") as file:
                data = json.load(file)

            if "episodes" not in data:
                raise ValueError(
                    f"Error: The JSON file {json_file_path} does not contain the 'episodes' key."
                )

            episodes = data["episodes"]

            episode_numbers = []
            episode_metrics = []

            for episode_key, episode_data in episodes.items():
                if "summary" in episode_data and metric in episode_data["summary"]:
                    episode_number = int(episode_key.split("_")[1])
                    episode_metric = episode_data["summary"][metric]

                    episode_numbers.append(episode_number)
                    episode_metrics.append(episode_metric)
                else:
                    raise ValueError(
                        f"Error: 'summary' or metric '{metric}' not found in episode {episode_key} in file {json_file_path}."
                    )

            sorted_indices = sorted(
                range(len(episode_numbers)), key=lambda i: episode_numbers[i]
            )
            episode_numbers = [episode_numbers[i] for i in sorted_indices]
            episode_metrics = [episode_metrics[i] for i in sorted_indices]

            all_episode_numbers.append(episode_numbers)
            all_episode_metrics.append(episode_metrics)

        # Ensure all episode lists are of the same length
        min_length = min(len(episodes) for episodes in all_episode_numbers)
        all_episode_numbers = [
            episodes[:min_length] for episodes in all_episode_numbers
        ]
        all_episode_metrics = [metrics[:min_length] for metrics in all_episode_metrics]

        # Calculate mean and standard deviation
        mean_metrics = np.mean(all_episode_metrics, axis=0)
        std_metrics = np.std(all_episode_metrics, axis=0)
        episode_numbers = all_episode_numbers[0]

        # Smooth the data using a moving average
        window_size = 200
        smoothed_mean_metrics = [
            np.mean(mean_metrics[i : i + window_size])
            for i in range(0, len(mean_metrics) - window_size + 1, window_size)
        ]
        smoothed_std_metrics = [
            np.mean(std_metrics[i : i + window_size])
            for i in range(0, len(std_metrics) - window_size + 1, window_size)
        ]
        smoothed_episode_numbers = [
            episode_numbers[i + window_size // 2]
            for i in range(0, len(episode_numbers) - window_size + 1, window_size)
        ]

        return smoothed_episode_numbers, smoothed_mean_metrics, smoothed_std_metrics

    try:
        # Process PPO JSON files
        ppo_episode_numbers, ppo_mean_metrics, ppo_std_metrics = process_json_files(
            ppo_jsons
        )

        # Process Random Baseline JSON files
        random_episode_numbers, random_mean_metrics, random_std_metrics = (
            process_json_files(random_jsons)
        )

        # Align the episode ranges to make the graphs equally long
        min_episodes = min(len(ppo_episode_numbers), len(random_episode_numbers))
        ppo_episode_numbers = ppo_episode_numbers[:min_episodes]
        ppo_mean_metrics = ppo_mean_metrics[:min_episodes]
        ppo_std_metrics = ppo_std_metrics[:min_episodes]
        random_episode_numbers = random_episode_numbers[:min_episodes]
        random_mean_metrics = random_mean_metrics[:min_episodes]
        random_std_metrics = random_std_metrics[:min_episodes]

        # Metric labels for the plot
        metric_labels = {
            "episode_reward": "Reward",
            "classification_accuracy_last": "Accuracy",
            "steps": "Quantum Gates",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric, metric)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot PPO Agent data
        plt.plot(ppo_episode_numbers, ppo_mean_metrics, label="PPO", color="blue")
        plt.fill_between(
            ppo_episode_numbers,
            np.maximum(np.array(ppo_mean_metrics) - np.array(ppo_std_metrics), 0.0),
            np.minimum(np.array(ppo_mean_metrics) + np.array(ppo_std_metrics), 1.0)
            if metric == "classification_accuracy_last"
            else np.array(ppo_mean_metrics) + np.array(ppo_std_metrics),
            color="blue",
            alpha=0.2,
        )

        # Plot Random Baseline data
        plt.plot(
            random_episode_numbers, random_mean_metrics, label="Random", color="orange"
        )
        plt.fill_between(
            random_episode_numbers,
            np.maximum(
                np.array(random_mean_metrics) - np.array(random_std_metrics), 0.0
            ),
            np.minimum(
                np.array(random_mean_metrics) + np.array(random_std_metrics), 1.0
            )
            if metric == "classification_accuracy_last"
            else np.array(random_mean_metrics) + np.array(random_std_metrics),
            color="orange",
            alpha=0.2,
        )

        # Add labels, title, and legend
        plt.xlabel("Episodes")
        plt.ylabel(metric_label)
        plt.legend(loc="lower right", title="Agent")
        plt.grid(True)
        plt.tight_layout()

        # Ensure the target directory exists
        os.makedirs(target_path, exist_ok=True)

        figure_file_path = os.path.join(target_path, f"{metric}_ppo_vs_random.svg")
        plt.savefig(figure_file_path, format="svg")
        plt.close()
        print(f"Plot saved to {figure_file_path}")

        parsed_data = {
            "ppo": {
                "episode_numbers": ppo_episode_numbers,
                "mean_metrics": ppo_mean_metrics,
                "std_metrics": ppo_std_metrics,
            },
            "random": {
                "episode_numbers": random_episode_numbers,
                "mean_metrics": random_mean_metrics,
                "std_metrics": random_std_metrics,
            },
        }
        json_file_path = os.path.join(target_path, f"{metric}_ppo_vs_random.json")
        with open(json_file_path, "w") as json_file:
            json.dump(parsed_data, json_file, indent=4)
        print(f"Parsed data saved to {json_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_runs_compare_ppo_and_random_agent_step_based(
    ppo_jsons: list[str], random_jsons: list[str], metric: str, target_path: str
):
    """
    Compares the performance of a PPO Agent and a Random Baseline by plotting the specified metric
    over steps from multiple JSON files. The results are saved as a combined SVG plot and a JSON file.
    """

    def process_json_files(json_files):
        all_step_numbers = []
        all_step_metrics = []

        for json_file_path in json_files:
            with open(json_file_path, "r") as file:
                data = json.load(file)

            if "episodes" not in data:
                raise ValueError(
                    f"Error: The JSON file {json_file_path} does not contain the 'episodes' key."
                )

            episodes = data["episodes"]

            step_numbers = []
            step_metrics = []
            total_steps = 0

            for episode_key, episode_data in episodes.items():
                if "summary" in episode_data and metric in episode_data["summary"]:
                    episode_steps = episode_data["summary"].get("steps", 0)
                    total_steps += episode_steps
                    step_metric = episode_data["summary"][metric]

                    step_numbers.append(total_steps)
                    step_metrics.append(step_metric)
                else:
                    raise ValueError(
                        f"Error: 'summary' or metric '{metric}' not found in episode {episode_key} in file {json_file_path}."
                    )

            all_step_numbers.append(step_numbers)
            all_step_metrics.append(step_metrics)

        # Ensure all step lists are of the same length
        min_length = min(len(steps) for steps in all_step_numbers)
        all_step_numbers = [steps[:min_length] for steps in all_step_numbers]
        all_step_metrics = [metrics[:min_length] for metrics in all_step_metrics]

        # Calculate mean and standard deviation
        mean_metrics = np.mean(all_step_metrics, axis=0)
        std_metrics = np.std(all_step_metrics, axis=0)
        step_numbers = all_step_numbers[0]

        # Smooth the data using a moving average
        window_size = 200
        smoothed_mean_metrics = [
            np.mean(mean_metrics[i : i + window_size])
            for i in range(0, len(mean_metrics) - window_size + 1, window_size)
        ]
        smoothed_std_metrics = [
            np.mean(std_metrics[i : i + window_size])
            for i in range(0, len(std_metrics) - window_size + 1, window_size)
        ]
        smoothed_step_numbers = [
            step_numbers[i + window_size // 2]
            for i in range(0, len(step_numbers) - window_size + 1, window_size)
        ]

        return smoothed_step_numbers, smoothed_mean_metrics, smoothed_std_metrics

    try:
        # Process PPO JSON files
        ppo_step_numbers, ppo_mean_metrics, ppo_std_metrics = process_json_files(
            ppo_jsons
        )

        # Process Random Baseline JSON files
        random_step_numbers, random_mean_metrics, random_std_metrics = (
            process_json_files(random_jsons)
        )

        # Align the step ranges to make the graphs equally long
        max_steps = min(max(ppo_step_numbers), max(random_step_numbers))
        ppo_indices = [
            i for i, step in enumerate(ppo_step_numbers) if step <= max_steps
        ]
        random_indices = [
            i for i, step in enumerate(random_step_numbers) if step <= max_steps
        ]

        ppo_step_numbers = [ppo_step_numbers[i] for i in ppo_indices]
        ppo_mean_metrics = [ppo_mean_metrics[i] for i in ppo_indices]
        ppo_std_metrics = [ppo_std_metrics[i] for i in ppo_indices]

        random_step_numbers = [random_step_numbers[i] for i in random_indices]
        random_mean_metrics = [random_mean_metrics[i] for i in random_indices]
        random_std_metrics = [random_std_metrics[i] for i in random_indices]

        # Metric labels for the plot
        metric_labels = {
            "episode_reward": "Reward",
            "classification_accuracy_last": "Accuracy",
            "steps": "Quantum Gates",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric, metric)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot PPO Agent data
        plt.plot(ppo_step_numbers, ppo_mean_metrics, label="PPO", color="blue")
        plt.fill_between(
            ppo_step_numbers,
            np.maximum(np.array(ppo_mean_metrics) - np.array(ppo_std_metrics), 0.0),
            np.minimum(np.array(ppo_mean_metrics) + np.array(ppo_std_metrics), 1.0)
            if metric == "classification_accuracy_last"
            else np.array(ppo_mean_metrics) + np.array(ppo_std_metrics),
            color="blue",
            alpha=0.2,
        )

        # Plot Random Baseline data
        plt.plot(
            random_step_numbers, random_mean_metrics, label="Random", color="orange"
        )
        plt.fill_between(
            random_step_numbers,
            np.maximum(
                np.array(random_mean_metrics) - np.array(random_std_metrics), 0.0
            ),
            np.minimum(
                np.array(random_mean_metrics) + np.array(random_std_metrics), 1.0
            )
            if metric == "classification_accuracy_last"
            else np.array(random_mean_metrics) + np.array(random_std_metrics),
            color="orange",
            alpha=0.2,
        )

        # Add labels, title, and legend
        plt.xlabel("Steps")
        plt.ylabel(metric_label)
        plt.legend(loc="lower right", title="Agent")
        plt.grid(True)
        plt.tight_layout()

        # Ensure the target directory exists
        os.makedirs(target_path, exist_ok=True)

        # Save the plot as an SVG file
        figure_file_path = os.path.join(
            target_path, f"{metric}_ppo_vs_random_steps.svg"
        )
        plt.savefig(figure_file_path, format="svg")
        plt.close()
        print(f"Plot saved to {figure_file_path}")

        # Save the parsed data to a JSON file
        parsed_data = {
            "ppo": {
                "step_numbers": ppo_step_numbers,
                "mean_metrics": ppo_mean_metrics,
                "std_metrics": ppo_std_metrics,
            },
            "random": {
                "step_numbers": random_step_numbers,
                "mean_metrics": random_mean_metrics,
                "std_metrics": random_std_metrics,
            },
        }
        json_file_path = os.path.join(target_path, f"{metric}_ppo_vs_random_steps.json")
        with open(json_file_path, "w") as json_file:
            json.dump(parsed_data, json_file, indent=4)
        print(f"Parsed data saved to {json_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_runs_compare_depth(
    ppo_file_lists: list[list[str]],
    random_file_lists: list[list[str]],
    metric: Literal[
        "episode_reward", "classification_accuracy_last", "steps", "circuit_depth"
    ],
    target_path: str,
):
    """
    Compares the specified metric across multiple sets of JSON files for PPO and Random agents, each representing a different circuit depth.
    Each set of JSON files is plotted as a separate graph in a combined plot, using steps as the x-axis.
    """
    try:
        # Metric labels for the plot
        metric_labels = {
            "episode_reward": "Reward",
            "classification_accuracy_last": "Accuracy",
            "steps": "Quantum Gates",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric, metric)

        plt.figure(figsize=(10, 6))  # Initialize the plot

        all_smoothed_step_numbers = []
        all_smoothed_mean_metrics = []
        all_smoothed_std_metrics = []
        circuit_depth_labels = []

        def process_file_lists(file_lists, agent_label):
            for json_files in file_lists:
                all_step_numbers = []
                all_step_metrics = []
                circuit_depth_label = None

                for json_file_path in json_files:
                    with open(json_file_path, "r") as file:
                        data = json.load(file)

                    # Extract the circuit depth from the first JSON file in the list
                    if circuit_depth_label is None:
                        circuit_depth_label = data.get("config", {}).get(
                            "CIRCUIT_DEPTH_MAX", "Unknown Depth"
                        )

                    if "episodes" not in data:
                        raise ValueError(
                            f"Error: The JSON file {json_file_path} does not contain the 'episodes' key."
                        )

                    episodes = data["episodes"]

                    step_numbers = []
                    step_metrics = []
                    total_steps = 0

                    for episode_key, episode_data in episodes.items():
                        if (
                            "summary" in episode_data
                            and metric in episode_data["summary"]
                        ):
                            episode_steps = episode_data["summary"].get("steps", 0)
                            total_steps += episode_steps
                            step_metric = episode_data["summary"][metric]

                            step_numbers.append(total_steps)
                            step_metrics.append(step_metric)
                        else:
                            raise ValueError(
                                f"Error: 'summary' or metric '{metric}' not found in episode {episode_key} in file {json_file_path}."
                            )

                    all_step_numbers.append(step_numbers)
                    all_step_metrics.append(step_metrics)

                # Ensure all step lists are of the same length
                min_length = min(len(steps) for steps in all_step_numbers)
                all_step_numbers = [steps[:min_length] for steps in all_step_numbers]
                all_step_metrics = [
                    metrics[:min_length] for metrics in all_step_metrics
                ]

                # Calculate mean and standard deviation
                mean_metrics = np.mean(all_step_metrics, axis=0)
                std_metrics = np.std(all_step_metrics, axis=0)
                step_numbers = all_step_numbers[0]

                # Smooth the data using a moving average
                window_size = 200
                smoothed_mean_metrics = [
                    np.mean(mean_metrics[i : i + window_size])
                    for i in range(0, len(mean_metrics) - window_size + 1, window_size)
                ]
                smoothed_std_metrics = [
                    np.mean(std_metrics[i : i + window_size])
                    for i in range(0, len(std_metrics) - window_size + 1, window_size)
                ]
                smoothed_step_numbers = [
                    step_numbers[i + window_size // 2]
                    for i in range(0, len(step_numbers) - window_size + 1, window_size)
                ]

                all_smoothed_step_numbers.append(smoothed_step_numbers)
                all_smoothed_mean_metrics.append(smoothed_mean_metrics)
                all_smoothed_std_metrics.append(smoothed_std_metrics)
                circuit_depth_labels.append(f"{circuit_depth_label} ({agent_label})")

        # Process PPO file lists
        process_file_lists(ppo_file_lists, "PPO")

        # Process Random file lists
        process_file_lists(random_file_lists, "Random")

        # Align all graphs to the smallest common step range
        min_steps = min(max(steps) for steps in all_smoothed_step_numbers)
        aligned_step_numbers = []
        aligned_mean_metrics = []
        aligned_std_metrics = []

        for i in range(len(all_smoothed_step_numbers)):
            indices = [
                j
                for j, step in enumerate(all_smoothed_step_numbers[i])
                if step <= min_steps
            ]
            aligned_step_numbers.append(
                [all_smoothed_step_numbers[i][j] for j in indices]
            )
            aligned_mean_metrics.append(
                [all_smoothed_mean_metrics[i][j] for j in indices]
            )
            aligned_std_metrics.append(
                [all_smoothed_std_metrics[i][j] for j in indices]
            )

        # Plot the aligned data
        for i in range(len(aligned_step_numbers)):
            plt.plot(
                aligned_step_numbers[i],
                aligned_mean_metrics[i],
                label=f"{circuit_depth_labels[i]}",
            )
            if metric == "classification_accuracy_last":
                plt.fill_between(
                    aligned_step_numbers[i],
                    np.maximum(
                        np.array(aligned_mean_metrics[i])
                        - np.array(aligned_std_metrics[i]),
                        0.0,
                    ),
                    np.minimum(
                        np.array(aligned_mean_metrics[i])
                        + np.array(aligned_std_metrics[i]),
                        1.0,
                    ),
                    alpha=0.2,
                )
            else:
                plt.fill_between(
                    aligned_step_numbers[i],
                    np.array(aligned_mean_metrics[i])
                    - np.array(aligned_std_metrics[i]),
                    np.array(aligned_mean_metrics[i])
                    + np.array(aligned_std_metrics[i]),
                    alpha=0.2,
                )

        # Add labels, title, and legend
        plt.xlabel("Steps")
        plt.ylabel(metric_label)
        plt.legend(loc="lower right", title="Max Circuit Depth")
        plt.grid(True, zorder=0)
        # plt.set_axisbelow(True)
        plt.tight_layout()

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Save the plot as an SVG file
        plt.savefig(target_path, format="svg")
        plt.close()
        l.console_log(f"Plot saved to {target_path}")

    except Exception as e:
        l.console_log(f"An error occurred: {e}")


def get_unique_circuits_number(json_file):
    """
    Get number of unique circuits based on JSON file "unique_circuits_config.json" created with method "get_unique_circuits"
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    max_number = max(
        int(key.split("_")[1]) for key in data.keys() if key.startswith("circuit_")
    )

    return max_number


def get_training_meta_data(json_files, target_path):
    """
    Parses multiple JSON files, extracts metadata, and saves a new JSON file with aggregated data.
    """
    result = {"summary": {}}
    runs = []
    all_summaries = []

    for file_path in json_files:
        l.console_log(f"Parsing {file_path}")
        with open(file_path, "r") as file:
            data = json.load(file)

        run_nr = f"run_{data['config']['RUN_NR']}"

        start_time = datetime.strptime(data["start_time"], "%d.%m.%Y:%H:%M:%S")
        end_time = datetime.strptime(data["end_time"], "%d.%m.%Y:%H:%M:%S")
        training_time = (end_time - start_time).total_seconds()

        run_data = {
            "start_time": data["start_time"],
            "end_time": data["end_time"],
            "training_time": training_time,
            **data["summary"],
        }
        result[run_nr] = run_data
        runs.append((start_time, end_time))
        all_summaries.append(data["summary"])

    if runs:
        overall_start_time = min(runs, key=lambda x: x[0])[0]
        overall_end_time = max(runs, key=lambda x: x[1])[1]
        result["summary"]["run_time"] = (
            overall_end_time - overall_start_time
        ).total_seconds()

        summary_keys = all_summaries[0].keys()
        for key in summary_keys:
            if isinstance(all_summaries[0][key], (int, float)):
                result["summary"][key] = sum(
                    summary[key] for summary in all_summaries
                ) / len(all_summaries)

    with open(target_path, "w") as target_file:
        json.dump(result, target_file, indent=4)
    l.console_log(f"Meta data saved to {target_path}")


def plot_unique_circuits_accuracy_bar_chart(json_file, target_path):
    """
    Generates a grouped bar chart for displaying number of unique circuits found per accuracy bin;
    matplotlib: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    accuracies = []
    for key, circuit in data.items():
        if "accuracy" in circuit:
            accuracies.append(circuit["accuracy"])

    # Define Accuracy-Bins (e.g., 0.0-0.1, 0.1-0.2, ..., 1.0-1.1)
    bins = np.arange(0.0, 1.1, 0.1)
    counts, bin_edges = np.histogram(accuracies, bins=bins)

    # Create labels for bins
    labels = [
        f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)
    ]

    # x-Positions for bars
    x = np.arange(len(counts))

    plt.figure(figsize=(10, 6))
    plt.bar(x, counts, width=0.6, align="center")
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Accuracy")
    plt.ylabel("Unique Circuits")
    plt.tight_layout()

    plt.savefig(target_path, format="svg")
    l.console_log(f"Plot saved to {target_path}")
    plt.close()


def plot_unique_circuits_accuracy_per_max_depth_bar_chart(json_files, target_path):
    # Define Accuracy-Bins
    bins = np.arange(0.0, 1.1, 0.1)
    nbins = len(bins) - 1

    all_counts = []
    labels = []

    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)

        accuracies = [
            circuit["accuracy"]
            for circuit in data.values()
            if isinstance(circuit, dict) and "accuracy" in circuit
        ]

        counts, _ = np.histogram(accuracies, bins=bins)
        all_counts.append(counts)

        match = re.search(r"cd(/d+)", fpath)
        if match:
            depth_label = match.group(1)
        else:
            depth_label = fpath
        labels.append(depth_label)

    nfiles = len(json_files)
    x = np.arange(nbins)
    total_bar_width = 0.8
    bar_width = total_bar_width / nfiles

    colors = [cm.Blues(0.3 + 0.7 * i / (nfiles - 1)) for i in range(nfiles)]

    plt.figure(figsize=(10, 6))
    for i, counts in enumerate(all_counts):
        offset = (i - nfiles / 2) * bar_width + bar_width / 2
        plt.bar(
            x + offset,
            counts,
            width=bar_width,
            label=labels[i],
            color=colors[i],
            zorder=2,
        )

    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(nbins)]
    plt.xticks(x, bin_labels, rotation=45)
    plt.xlabel("Accuracy")
    plt.ylabel("Unique Circuits")
    plt.grid(axis="y", alpha=0.7, zorder=-1)
    plt.legend(title="Max Circuit Depth", loc="upper left")
    plt.tight_layout()
    plt.savefig(target_path, format="svg")
    l.console_log(f"Plot saved to {target_path}")
    plt.close()


def plot_acc_gates_depth_3d_scatter(json_file, target_path):
    """
    Generates a 3D scatter plot of accuracy, gates, and circuit depth from a JSON file.
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    gates = []
    depths = []
    accuracies = []

    for circuit in data.values():
        if (
            isinstance(circuit, dict)
            and "gates" in circuit
            and "circuit_depth" in circuit
            and "accuracy" in circuit
        ):
            gates.append(circuit["gates"])
            depths.append(circuit["circuit_depth"])
            accuracies.append(circuit["accuracy"])

    scale_factor = 100  # adapt if necessary
    marker_sizes = [acc * scale_factor for acc in accuracies]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        gates,
        depths,
        accuracies,
        s=marker_sizes,
        alpha=0.6,
        c=accuracies,
        cmap="viridis",
        edgecolors="k",
    )

    ax.set_xlabel("Gates")
    ax.set_ylabel("Circuit Depth")
    ax.set_zlabel("Accuracy")
    fig.colorbar(scatter, ax=ax, label="Accuracy")

    plt.tight_layout()
    plt.savefig(target_path, format="svg")
    l.console_log(f"Plot saved to {target_path}")
    plt.close()


def plot_gate_type_frequency_acc_parallel_coordinates(json_file, target_path):
    with open(json_file, "r") as f:
        data = json.load(f)

    dimensions = ["CNOT", "RX", "RY", "RZ", "accuracy"]

    rows = []
    for circuit in data.values():
        if isinstance(circuit, dict) and "circuit" in circuit and "accuracy" in circuit:
            gate_list = circuit["circuit"]
            count_cnot = sum(
                1 for gate in gate_list if gate.split()[0].upper() == "CNOT"
            )
            count_rx = sum(1 for gate in gate_list if gate.split()[0].upper() == "RX")
            count_ry = sum(1 for gate in gate_list if gate.split()[0].upper() == "RY")
            count_rz = sum(1 for gate in gate_list if gate.split()[0].upper() == "RZ")
            accuracy = circuit["accuracy"]
            rows.append(
                {
                    "CNOT": count_cnot,
                    "RX": count_rx,
                    "RY": count_ry,
                    "RZ": count_rz,
                    "accuracy": accuracy,
                }
            )

    if not rows:
        print("Keine validen Circuit-Daten gefunden.")
        return

    all_accuracies = [row["accuracy"] for row in rows]
    vmin, vmax = min(all_accuracies), max(all_accuracies)
    norm = plt.Normalize(vmin, vmax)
    colormap = matplotlib.colormaps["viridis"]

    x = np.arange(len(dimensions))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    n = len(rows)
    if n > 100:
        sampled_rows = random.sample(rows, 300)
    else:
        sampled_rows = rows

    for row in sampled_rows:
        y = [row[dim] for dim in dimensions]
        color = colormap(norm(row["accuracy"]))
        ax.plot(x, y, color=color, alpha=0.3)

    avg_values = [np.mean([row[dim] for row in rows]) for dim in dimensions]
    avg_accuracy = np.mean(all_accuracies)
    agg_color = colormap(norm(avg_accuracy))
    ax.plot(
        x, avg_values, color=agg_color, linewidth=3, marker="o", label="Aggregated Mean"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.set_xlabel("Quantum Gate")
    ax.set_ylabel("Count")

    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Accuracy")

    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(target_path, format="svg")
    l.console_log(f"Plot saved to {target_path}")
    plt.close()


def plot_gate_type_frequency_acc_scatter(json_file, target_path):
    """
    Plots the accuracy of quantum circuits against the frequency of different gate types.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    gate_types = ["CNOT", "RX", "RY", "RZ"]

    results = {gate: {"count": [], "accuracy": []} for gate in gate_types}

    for circuit in data.values():
        if isinstance(circuit, dict) and "circuit" in circuit and "accuracy" in circuit:
            gate_list = circuit["circuit"]
            accuracy = circuit["accuracy"]

            counts = {gate: 0 for gate in gate_types}
            for gate_str in gate_list:
                prefix = gate_str.split()[0].upper()
                if prefix in counts:
                    counts[prefix] += 1

            for gate in gate_types:
                results[gate]["count"].append(counts[gate])
                results[gate]["accuracy"].append(accuracy)

    for gate in gate_types:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            results[gate]["count"], results[gate]["accuracy"], marker="o", alpha=0.7
        )
        plt.xlabel(f"{gate} Gates")
        plt.ylabel("Accuracy")
        plt.grid(True)

        out_file = os.path.join(
            target_path, f"acc_gate_type_frequency_{gate}_scatter.svg"
        )
        plt.tight_layout()
        plt.savefig(out_file, format="svg")
        l.console_log(f"Plot saved to {out_file}")
        plt.close()


def plot_gate_distribution_heatmap(json_file, target_path):
    """
    Generates a heatmap showing the distribution of quantum gates across different circuit depths.
    """
    try:
        with open(json_file, "r") as file:
            data = json.load(file)

        env = rl_cl.QuantumCircuit()

        max_depth = max(data[circuit_key]["circuit_depth"] for circuit_key in data)

        gate_types = ["CNOT", "RX", "RY", "RZ"]
        gate_counts = np.zeros((max_depth, len(gate_types)))

        for circuit_key in data:
            circuit_data = data[circuit_key]
            h.create_circuit_tensor(env, circuit_data)  # Update env.observation tensor

            for depth_idx in range(max_depth):
                for gate_idx, gate_name in enumerate(gate_types):
                    gate_counts[depth_idx, gate_idx] += np.sum(
                        env.observation[
                            depth_idx, :, env.gate_set_names.index(gate_name)
                        ]
                    )

            env.reset()

        gate_probabilities = gate_counts / np.sum(gate_counts)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            gate_probabilities.T, aspect="auto", cmap="Blues", origin="lower"
        )

        ax.set_xlabel("Circuit Depth")
        ax.set_ylabel("Quantum Gate")
        ax.set_xticks(range(max_depth))
        ax.set_xticklabels(range(1, max_depth + 1))
        ax.set_yticks(range(len(gate_types)))
        ax.set_yticklabels(gate_types)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Probability")

        plt.tight_layout()
        plt.savefig(target_path, format="svg")
        plt.close()
        l.console_log(f"Plot saved to {target_path}")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file}")
    except json.JSONDecodeError:
        raise ValueError(
            "Failed to decode the JSON file. Ensure it is properly formatted."
        )
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def plot_cost_landscape(
    first_param_idx: int,
    second_param_idx: int,
    param_ranges,
    fixed_params,
    batch_size: int,
    target_path: str,
    resolution=50,
):
    """
    Plots the cost landscape of a quantum circuit using the actual cost function.
    """
    env = rl_cl.QuantumCircuit()

    @jax.jit
    def cost_jax(params, data, targets):
        def single_example_loss(params, x, y):
            probs = env.full_circuit(params, x)
            probs = map_measurements_to_classes(env, probs)
            return -jnp.sum(y * jnp.log(probs))

        loss = jax.vmap(single_example_loss, in_axes=(None, 0, 0))(
            params, data, targets
        )
        return jnp.mean(loss)

    @jax.jit
    def cost_batch_jax(params, X, Y):
        idx = jax.random.choice(
            jax.random.PRNGKey(0), len(X), (batch_size,), replace=False
        )
        X_batch = X[idx]
        Y_batch = Y[idx]

        return cost_jax(params, X_batch, Y_batch)

    def quantum_circuit(params, x):
        qml.AmplitudeEmbedding(
            features=x,
            wires=range(env.qubit_count),
            normalize=False,
            validate_norm=True,
        )

        #! Insert your circuit here
        # h.parse_circuit(json_file_path=json_path_circuit,
        #                 key_name="episode",
        #                 number=circuit_number,
        #                 weights_mode="argument",
        #                 weights_arg=params)

        qml.CNOT(wires=[0, 1])
        qml.RZ(params[0], wires=1)
        qml.RY(params[1], wires=1)
        qml.RX(params[2], wires=0)

        return qml.probs(wires=range(env.measurement_qubits))

    env.full_circuit = qml.QNode(quantum_circuit, env.device, interface="jax")

    # Create a grid of parameter values
    param1_range = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
    param2_range = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
    param1, param2 = np.meshgrid(param1_range, param2_range)

    # Initialize the cost values
    cost_values = np.zeros_like(param1)

    # Iterate over the grid and calculate the cost
    for i in range(resolution):
        for j in range(resolution):
            params = fixed_params.copy()
            params[first_param_idx] = param1[i, j]
            params[second_param_idx] = param2[i, j]
            # Calculate the cost using the actual cost function
            cost_values[i, j] = cost_batch_jax(
                jnp.array(params), env.X_train_jax, env.y_train_jax
            )

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        param1, param2, cost_values, cmap="viridis", edgecolor="none"
    )
    ax.set_xlabel(r"$/theta_1$")
    ax.set_ylabel(r"$/theta_2$")
    ax.set_zlabel("Cost")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    # plt.show()
    plt.tight_layout()
    plt.savefig(target_path, format="svg")
    plt.close()
    l.console_log(f"Plot saved to {target_path}")


# TODO: Doesn't work yet
def generate_quantikz_code(vqc, params=None):
    """
    Generates LaTeX code using quantikz for a given Variational Quantum Circuit (VQC).
    """
    num_qubits = (
        max(
            max(map(int, elem.split("(")[1].replace(")", "").split(",")))
            for elem in vqc
        )
        + 1
    )

    latex_code = [
        "//documentclass[tikz,border=2mm]{standalone}",
        "//usepackage{quantikz}",
        "//begin{document}",
        "//begin{quantikz}",
    ]

    rows = [f"//lstick{{$//ket{{q_{{{i}}}}}$}} & " for i in range(num_qubits)]
    params_idx = 0

    for i, gate in enumerate(vqc):
        parts = gate.split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid gate format: {gate}")
        gate_type, qubits_str = parts
        qubits = list(map(int, qubits_str.strip("() ").split(",")))

        if gate_type == "CNOT":
            control, target = qubits
            rows[control] += f"//ctrl{{{target - control}}} & "
            rows[target] += f"//targ{{}} & "
        elif gate_type in {"RX", "RY", "RZ"}:
            qubit = qubits[0]
            param = (
                f"//theta_{{{i + 1}}}"
                if params is None
                else f"{params[params_idx]:.2f}"
            )
            rows[qubit] += f"//gate{{{gate_type}({param})}} & "
            params_idx += 1
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

        for j in range(num_qubits):
            if j not in qubits:
                rows[j] += "//qw & "

    rows = [row.rstrip(" & ") + " //" for row in rows]

    latex_code.extend(rows)

    latex_code.append("//end{quantikz}")
    latex_code.append("//end{document}")

    print("/n".join(latex_code))


def plot_gate_frequencies_bar(json_file, target_path):
    """
    Plots the frequency of different quantum gate types in a bar chart.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    gate_counts = Counter()
    qubit_counts = {
        "CNOT": Counter(),
        "RX": Counter(),
        "RY": Counter(),
        "RZ": Counter(),
    }

    for circuit in data.values():
        if "circuit" in circuit:
            for gate in circuit["circuit"]:
                gate_type, qubits = gate.split(" ", 1)
                qubits = qubits.strip("()").split(",")
                gate_counts[gate_type] += 1
                if gate_type == "CNOT":
                    qubit_counts[gate_type][tuple(map(int, qubits))] += 1
                else:
                    qubit_counts[gate_type][int(qubits[0])] += 1

    colors = ["#08306b", "#2171b5", "#6baed6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = ["CNOT", "RX", "RY", "RZ"]
    x_positions = np.arange(len(x_labels))
    bar_width = 0.2

    for i, gate_type in enumerate(x_labels):
        # "Total"-Balken
        ax.bar(
            x_positions[i] - bar_width,
            gate_counts[gate_type],
            width=bar_width,
            label=f"{gate_type} Total",
            color=colors[0],
            zorder=2,
        )

        qubit_labels = []
        qubit_values = []
        if gate_type == "CNOT":
            qubit_labels = ["(0, 1)", "(1, 0)"]
            qubit_values = [
                qubit_counts[gate_type].get((0, 1), 0),
                qubit_counts[gate_type].get((1, 0), 0),
            ]
        else:
            qubit_labels = ["0", "1"]
            qubit_values = [
                qubit_counts[gate_type].get(0, 0),
                qubit_counts[gate_type].get(1, 0),
            ]

        ax.bar(
            x_positions[i],
            qubit_values[0],
            width=bar_width,
            label=f"{gate_type} {qubit_labels[0]}",
            color=colors[1],
            zorder=2,
        )
        ax.bar(
            x_positions[i] + bar_width,
            qubit_values[1],
            width=bar_width,
            label=f"{gate_type} {qubit_labels[1]}",
            color=colors[2],
            zorder=2,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Gate")
    ax.set_ylabel("Frequency")
    ax.legend(title="Gate-Qubit Combination", loc="upper left")

    plt.tight_layout()
    plt.grid(axis="y", alpha=0.7, zorder=-1)
    plt.savefig(target_path, format="svg")
    l.console_log(f"Plot saved to {target_path}")
    plt.close()


def plot_gate_sequence_transition_heatmap(json_file, target_path):
    """
    Generates a heatmap showing the transition probabilities between different quantum gate types.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    gate_sequences = []
    for circuit in data.values():
        if "circuit" in circuit:
            gate_sequences.extend(circuit["circuit"])

    gate_qubit_pairs = []
    for gate in gate_sequences:
        try:
            gate_type, qubits = gate.split(" ", 1)
            qubits = tuple(map(int, qubits.strip("()").split(",")))
            if gate_type in {"RX", "RY", "RZ"}:
                qubits = (qubits[0],)
            gate_qubit_pairs.append((gate_type, qubits))
        except (ValueError, IndexError):
            continue

    gate_types = ["CNOT", "RX", "RY", "RZ"]
    gate_index = {gate: idx for idx, gate in enumerate(gate_types)}

    transition_matrix = np.zeros((len(gate_types), len(gate_types)))

    for circuit in data.values():
        if "circuit" in circuit:
            gates = []
            for gate in circuit["circuit"]:
                try:
                    gate_type, qubits = gate.split(" ", 1)
                    qubits = tuple(map(int, qubits.strip("()").split(",")))
                    if gate_type in {"RX", "RY", "RZ"}:
                        qubits = (qubits[0],)
                    gates.append((gate_type, qubits))
                except (ValueError, IndexError):
                    continue

            for i in range(len(gates) - 1):
                from_gate_type = gates[i][0]
                to_gate_type = gates[i + 1][0]
                if gates[i][1][-1] == gates[i + 1][1][0]:
                    transition_matrix[
                        gate_index[from_gate_type], gate_index[to_gate_type]
                    ] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(transition_matrix, cmap="Blues", origin="upper")

    ax.set_xticks(np.arange(len(gate_types)))
    ax.set_yticks(np.arange(len(gate_types)))
    ax.set_xticklabels(gate_types, rotation=45, ha="right")
    ax.set_yticklabels(gate_types)
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Probability")

    plt.tight_layout()
    plt.savefig(target_path, format="svg")
    l.console_log(f"Plot saved to {target_path}")
    plt.close()


def extract_frequent_sequences(json_file, top_n=10):
    """
    Extracts the most frequent gate sequences from a JSON file containing quantum circuit data.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    sequences = []
    for circuit in data.values():
        if "circuit" in circuit:
            gates = [gate.split(" ")[0] for gate in circuit["circuit"]]
            sequences.extend(zip(gates, gates[1:]))
            sequences.extend(zip(gates, gates[1:], gates[2:]))

    sequence_counts = Counter(sequences)
    return sequence_counts.most_common(top_n)


def extract_sequences(json_file, sequence_lengths, top_n, target_path):
    """
    Extracts the most frequent gate sequences of specified lengths from a JSON file containing quantum circuit data.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    results = {}

    for sequence_length in sequence_lengths:
        sequences = []

        for circuit in data.values():
            if "circuit" in circuit:
                gates = []
                for gate in circuit["circuit"]:
                    try:
                        gate_type, qubits = gate.split(" ", 1)
                        qubits = tuple(map(int, qubits.strip("()").split(",")))
                        if gate_type in {"RX", "RY", "RZ"}:
                            qubits = (qubits[0],)
                        gates.append((gate_type, qubits))
                    except (ValueError, IndexError):
                        continue

                for i in range(len(gates) - sequence_length + 1):
                    if all(
                        gates[j][1][-1] == gates[j + 1][1][0]
                        for j in range(i, i + sequence_length - 1)
                    ):
                        sequences.append(
                            tuple(
                                f"{gate[0]} ({', '.join(map(str, gate[1]))})"
                                for gate in gates[i : i + sequence_length]
                            )
                        )

        sequence_counts = Counter(sequences)
        total_sequences = sum(sequence_counts.values())
        most_common_sequences = sequence_counts.most_common(top_n)

        results[f"sequence_length_{sequence_length}"] = {
            f"sequence_{idx + 1}": {
                "pattern": list(sequence),
                "frequency_absolut": count,
                "frequency_relative": round(count / total_sequences, 4),
            }
            for idx, (sequence, count) in enumerate(most_common_sequences)
        }

    with open(target_path, "w") as output_file:
        json.dump(results, output_file, indent=4)
    l.console_log(f"JSOn saved to {target_path}")


if __name__ == "__main__":
    # python ./analyse.py --ct 'iris' --cn 0 --rn 1 --cd 5 --rm 0 --ts 150000 --ep 2 --st 1024 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm 'fixed' --obs_win 1 105 --obs_list 2 --obs_fix 20 --br 'false' --brt 0.95 --olr 0.01 --opt 'adam' --oep 1000 --gpm 'random' --gpv 1.0 --gps 5 --mm 'minimum' --alg 'ppo'

    ########################
    # Run opt loop
    ########
    # param_range = 1.0
    # number_seeds = 3
    # ml_task = "mnist_2"
    # epochs_plot = 250
    # compare_benchmark_layers = 2
    # circuit_number = "2L"
    # batch_size = 20

    # if c.CIRCUIT_MEASUREMENT_MODE == "minimum":
    #     subpath = "mm_minimum"
    # elif c.CIRCUIT_MEASUREMENT_MODE == "all":
    #     subpath = "mm_all"

    ####
    # Run opt loop
    ####

    # run_optimization_loop(target_path=target, param_range_bound=param_range, number_seeds=number_seeds)
    ####

    ####
    # Plot Acc x Opt Batch Sizes
    ####

    # plot_acc_opt_batch_size(file_path_circuit=target, figure_file_path=figure_path)
    ####

    ####
    # Plot Acc x Opt Epochs
    # Valid modes: [loss, acc_train, acc_test, all]
    ####
    #
    ## For compare = False

    # plot_acc_and_loss_opt_epochs(figure_path=figure_path, qas_json=qas_json, compare=False, mode="all", epochs=epochs_plot, batch_size=batch_size)
    #
    # For compare = True

    # plot_acc_and_loss_opt_epochs(figure_path=figure_path, qas_json=qas_json, benchmark_json=benchmark_json, compare=True, mode="all", epochs=epochs_plot, batch_size=batch_size)
    ########################

    ########################
    # Get batch sizes for best acc test
    ########
    # batch_sizes = get_best_opt_batch_sizes(path, 0.9)
    # print(json.dumps(batch_sizes, indent=4))
    ########################

    ########################
    # Get unique circuits of one or several runs
    ########
    # ml_task = "mnist_2"
    # config_nr = 2
    # batch = 20
    # steps = 400000
    # performance_threshold = 0.9
    # # #
    # json_files_ppo = [

    # ]

    # json_files_random = [

    # ]

    # get_unique_circuits(json_files=json_files_ppo, target_path=target_path, mode="performance", performance_threshold=performance_threshold)
    ########################

    ########################
    # Plot run
    ########
    # alg = "ppo"
    # ml_task = "mnist_2"
    # config_nr = 2
    # batch = 20
    # steps = 400000
    # depth = 7
    # metrics = ["episode_reward", "classification_accuracy_last", "steps", "circuit_depth"]

    # json_files = [

    # ]

    # for i in range(3):

    #     os.makedirs(target_path, exist_ok=True)
    #     for metric in metrics:
    #         plot_run(json_file_path=json_files[i], metric=metric, target_path=target_path)
    ########################

    ########################
    # Plot runs
    ########
    # json_files = [

    # ]

    # json_files = [

    # ]

    # metrics = ["episode_reward", "classification_accuracy_last", "steps", "circuit_depth"]

    # for metric in metrics:
    #     plot_runs(metric=metric, target_file_path=target_path, json_file_paths=json_files)
    ########################

    ########################
    # Plot comparison of runs for PPO Agent and Random Baseline
    ########
    # json_files_ppo = [

    # ]
    #
    # json_files_random = [

    # ]
    #
    # for metric in metrics:

    #     # plot_runs_compare_ppo_and_random_agent_episode_based(ppo_jsons=json_files_ppo, random_jsons=json_files_random, metric=metric, target_path=target_path)
    #     plot_runs_compare_ppo_and_random_agent_step_based(ppo_jsons=json_files_ppo, random_jsons=json_files_random, metric=metric, target_path=target_path)
    ########################

    ########################
    # Plot comparison of runs with different max circuit depth within a config
    #########
    # metrics = ["episode_reward", "classification_accuracy_last", "steps", "circuit_depth"]
    # ml_task = "mnist_2"
    # config_nr = 17
    # batch = 20
    # steps = 400000

    # ppo_files_cd4 = [

    # ]

    # ppo_files_cd5 = [

    # ]

    # ppo_files_cd6 = [

    # ]

    # ppo_files_cd7 = [

    # ]
    # #
    # ppo_files_lists = [ppo_files_cd4, ppo_files_cd5, ppo_files_cd6]
    #
    # random_files_cd4 = [

    # ]
    #
    # random_files_cd5 = [

    # ]
    #
    # random_files_cd6 = [

    # ]
    #
    # random_files_lists = [random_files_cd4, random_files_cd5, random_files_cd6]
    # random_files_lists = [random_files_cd6]
    # random_files_lists = []
    #
    # for metric in metrics:

    #     plot_runs_compare_depth(ppo_file_lists=ppo_files_lists, random_file_lists=random_files_lists, metric=metric, target_path=target_path)

    ########################

    ########################
    # Get number of unique circuits based on JSON file created with method "get_unique_circuits"
    ########

    # print(get_unique_circuits_number(json_path))
    ########################

    ########################
    # Get meta data for one training (= 3 runs)
    #########
    # ml_task = "mnist_2"
    # config_nr = 2
    # batch = 20
    # steps = 400000
    # depths = [4,5,6,7]

    # for depth in depths:
    #     json_files = [

    #     ]

    #     get_training_meta_data(json_files, target_path)
    ########################

    ########################
    # Plot grouped bar chart for Accuracy x unique circuits
    ########
    # ml_task = "iris"
    # config_nr = 1
    #

    #
    # plot_unique_circuits_accuracy_bar_chart(json_file, target_path)
    ########################

    ########################
    # Plot grouped bar chart for Accuracy x unique circuits per max circuit depth (for each max circuit depth own bar)
    ########
    # ml_task = "iris"
    # config_nr = 1
    #
    # json_files = [

    # ]
    #

    #
    # plot_unique_circuits_accuracy_per_max_depth_bar_chart(json_files, target_path)
    ########################

    ########################
    # Plot 3D Scatter Plot for Accuracy x Gates x Depth
    ########
    # ml_task = "iris"
    # config_nr = 1
    #

    #
    # plot_acc_gates_depth_3d_scatter(json_file, target_path)
    ########################

    ########################
    # Plot gate type frequency over accuracy with parallel coordinates
    ########
    # ml_task = "iris"
    # config_nr = 1
    #

    #
    # plot_gate_type_frequency_acc_parallel_coordinates(json_file, target_path)
    ########################

    ########################
    # Create Scatter plot for each Quantum Gate Type that displays the count of the gate type and over accuracy
    ########
    # ml_task = "iris"
    # config_nr = 1
    #

    #

    #
    # plot_gate_type_frequency_acc_scatter(json_file, target_path)
    ########################

    ########################
    # Create gate distribution heatmap plot that displays probability of quantum gate types over circuit depth
    ########
    # ml_task = "iris"
    # config_nr = 1
    #

    #
    # plot_gate_distribution_heatmap(json_file, target_path)
    ########################

    ########################
    # Plot cost landscape
    ########
    # ml_task = "iris"
    # circuit_number = 1
    # first_param_index = 1
    # second_param_index = 2
    # param_range = "1.0"
    # # param_ranges = ((-np.pi, np.pi), (-np.pi, np.pi))  # Range for the first two parameters
    # param_ranges = ((-1.0, 1.0), (-1.0, 1.0))  # Range for the first two parameters
    # #! Set optimized params for circuit
    # fixed_params = [-1.7410495673015194,
    #                 1.247916362510123,
    #                 -1.0043564973024153
    #                 ]  # Parameters; one Parameter has to be fixed, the others are varied
    # batch_size = 20
    #

    # plot_cost_landscape(first_param_index, second_param_index, param_ranges, fixed_params, batch_size=batch_size, target_path=target_path, resolution=100)
    ########################

    # TODO: Doesn't work yet
    ########################
    # Generate LaTeX code for visualizing VQC via Quantikz
    ########
    # vqc = ["CNOT (0, 1)",
    #        "RZ (1, 0)",
    #        "RY (1, 0)",
    #        "RX (0, 1)"
    #         ]
    #
    # params = [-1.7410495673015194,
    #           1.247916362510123,
    #           -1.0043564973024153
    #         ]
    #
    # generate_quantikz_code(vqc, params)
    ########################

    ########################
    # Plot gate frequencies for a collected set of unique circuits
    ########
    # ml_task = "iris"
    # config_nr = 1
    # performance_threshold = 1.0

    # plot_gate_frequencies_bar(json_path, target_path)
    ########################

    ########################
    # Plot gate sequences for a collected set of unique circuits as heatmap
    ########
    # ml_task = "iris"
    # config_nr = 1
    # performance_threshold = 0.9

    # plot_gate_sequence_transition_heatmap(json_path, target_path)
    ########################

    ########################
    # Plot gate sequences for a collected set of unique circuits as network
    ########
    # ml_task = "iris"
    # config_nr = 1
    # performance_threshold = 1.0
    # sequence_lengths = [2,3,4]
    # sequence_lengths_string = ",".join(map(str, sequence_lengths))
    # top_n = 20

    # extract_sequences(json_path, sequence_lengths, top_n, target_path)
    ########################

    ########################
    # Plot gate sequences freuqncy for a collected set of unique circuits as sankey plot
    ########
    # ml_task = "iris"
    # config_nr = 1
    # performance_threshold = 1.0
    # sequence_length = 3
    # top_n = 10

    # plot_gate_sequences_sankey(json_path, sequence_length, top_n, target_path)
    ########################

    pass
