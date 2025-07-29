"""Reinforcement Learning Custom Environment for Quantum Architecture Search (ML Task Classification)"""

import collections
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations
from typing import Any, Literal

import config as c
import helpers as h
import jax
import jax.numpy as jnp
import log as l
import ml_classification as ml
import numpy as np
import optax
import pennylane as qml
from gymnasium import Env
from gymnasium.envs.registration import register
from gymnasium.spaces import MultiBinary, MultiDiscrete
from numpy.typing import NDArray
from pennylane import numpy as pnp
from pennylane.operation import Operation

register(
    id="QuantumCircuitCl-v0",
    entry_point="rl_env_cl:QuantumCircuit",
)


jax.config.update("jax_enable_x64", True)


@dataclass
class QuantumGate:
    """
    Dataclass for Quantum Gates in the Quantum Circuit Environment
    """
    op: type[Operation]
    wires: NDArray[Any]


class QuantumCircuit(Env[NDArray[Any], NDArray[Any]]):
    """
    RL Gymnasium Environment for Building Quantum Circuits
    """
    metadata = {"render_modes": ["human"]}
    gate_set = [
        qml.RX,
        qml.RY,
        qml.RZ,
        qml.CNOT
    ]
    gate_set_names = ["RX", "RY", "RZ", "CNOT"]


    def __init__(
        self
    ) -> None:
        super().__init__()
        
        l.console_log("Initializing Quantum Circuit Environment")
        
        # Dictionaries for Data Logging
        self.data_log_run = l.initialize_data_log_run()
        self.data_log_episode = l.initialize_data_log_episode()
        
        # Quantum Circuit related Attributes
        self.qubit_count = ml.get_required_qubit_number(c.CLASS_TASK)
        self.circuit_depth_max = c.CIRCUIT_DEPTH_MAX
        self.circuit_depth_max_dynamic = c.CIRCUIT_DEPTH_DYNAMIC_INIT
        self.circuit_depth = 0
        self.gate_count_max = self.qubit_count * self.circuit_depth_max
        self.gate_count_remaining = self.gate_count_max
        self.wires_permutations = self.get_wire_permutations(self.gate_set)
        self.gates_applied: list[QuantumGate] = []
        self.gates_params: list[float] = []
        
        # Reinforcement Learning related Attributes
        self.reward_cumulative = 0
        self.episode_idx = 0                                # Set to 1 by reset method before first step is executed
        self.step_idx = 0                                   # Set to 1 by step method when first step is executed
        
        # Action Space: Two integers representing the gate and the wires the gate operates on
        self.action_space = MultiDiscrete([len(self.gate_set), len(self.wires_permutations)], seed=c.SEED)
        self.illegal_actions = []
        
        ####
        ## Observation Space: 3D tensor containing binary values; dimensions: Circuit Depth x Qubit Count x (Gate Count + Qubit Count - 1); (qubit_count-1) for CNOT Gate
        ####
        # Reminder: Circuit depth dimension of tensor set to static value higher than circuit_depth_max for caching purposes; 
        # In this way, the same circuits receive the same hash value, even if the maximum circuit depth and therefore the observation may differ.
        # 
        # Reminder: Adapt reset method if obersevation_space or observation is changed
        #
        # Initial version without caching functionality:  
        # self.observation_space = MultiBinary([self.circuit_depth_max, self.qubit_count, len(self.gate_set)+(self.qubit_count-1)], seed=c.SEED)
        # self.observation = np.zeros((self.circuit_depth_max, self.qubit_count, len(self.gate_set)+(self.qubit_count-1)), dtype=np.int8)
        
        self.observation_space = MultiBinary([10, self.qubit_count, len(self.gate_set)+(self.qubit_count-1)], seed=c.SEED)
        self.observation = np.zeros((10, self.qubit_count, len(self.gate_set)+(self.qubit_count-1)), dtype=np.int8)
        self.qubit_depth_idx = [0] * self.qubit_count       # Index representing current depth for each qubit for the next gate to be applied
        
        # Quantum Device and Quantum Circuit
        self.device = qml.device("default.qubit", wires=self.qubit_count, shots=None)
        # self.device = qml.device("lightning.qubit", wires=self.qubit_count, shots=None)
        self.full_circuit = qml.QNode(self.quantum_circuit, self.device, interface='jax')
        
        ####
        ## Machine Learning Task related Attributes
        ####
        self.classes_num = ml.get_classes_number(c.CLASS_TASK)
        
        # Variables containing the whole data set 
        self.X_train, self.X_test, self.y_train, self.y_test = ml.load_preprocessed_data(split_train_test=False)
        self.X_train_jax = jnp.array(self.X_train)                          # Convert to JAX array for optimization with JAX
        self.y_train_jax = jnp.array(self.y_train)                          # Convert to JAX array for optimization with JAX
        self.X_test_jax = jnp.array(self.X_test)                            # Convert to JAX array for optimization with JAX
        self.y_test_jax = jnp.array(self.y_test)                            # Convert to JAX array for optimization with JAX
        
        # Variables containing only training data but again splitted into training and validation set
        # self.X_train_train, self.X_train_test, self.y_train_train, self.y_train_test = ml.load_preprocessed_data(split_train_test=True)
        # self.X_train_train_jax = jnp.array(self.X_train_train)            # Convert to JAX array for optimization with JAX
        # self.y_train_train_jax = jnp.array(self.y_train_train)            # Convert to JAX array for optimization with JAX
        # self.X_train_test_jax = jnp.array(self.X_train_test)              # Convert to JAX array for optimization with JAX
        # self.y_train_test_jax = jnp.array(self.y_train_test)              # Convert to JAX array for optimization with JAX
        
        # Variables containing only a small batch of the data set for faster VQC parameter optimization
        self.X_train_batch, self.y_train_batch = ml.get_data_batch(self.X_train, self.y_train, self.classes_num, c.OPT_VALIDATION_SET)
        self.X_test_batch, self.y_test_batch = ml.get_data_batch(self.X_test, self.y_test, self.classes_num, c.OPT_VALIDATION_SET)
        self.X_train_batch_jax = jnp.array(self.X_train_batch)              # Convert to JAX array for optimization with JAX
        self.y_train_batch_jax = jnp.array(self.y_train_batch)              # Convert to JAX array for optimization with JAX
        self.X_test_batch_jax = jnp.array(self.X_test_batch)                # Convert to JAX array for optimization with JAX
        self.y_test_batch_jax = jnp.array(self.y_test_batch)                # Convert to JAX array for optimization with JAX
        
        # Performance related attributes
        self.performance_target = c.PERFORMANCE_TARGET
        self.accuracy_prev = 0                                              # Accuracy on validation set (test) of previous step
        self.accuracy_average = 0                                           # Average accuracy on validation set (test) during episode
        self.accuracy_max = 0                                               # Max accuracy on validation set (test) during episode
        self.accuracy_history = collections.deque(maxlen=c.CIRCUIT_DEPTH_WINDOW_SIZE) # Store last n accuracies for dynamic circuit depth adjustment
        self.measurement_qubits = ml.get_required_measurement_qubits(self.classes_num, c.CIRCUIT_MEASUREMENT_MODE, c.CLASS_TASK)
        self.relevant_measurements = ml.get_relevant_measurements_number(self.classes_num, c.CIRCUIT_MEASUREMENT_MODE, c.CLASS_TASK)
        self.circuit_accuracy_cache = h.Cache()                             # Cache for storing accuracies on validation set (test) for each unique circuit
        self.clear_jax_cache = False
        
        ####
        ## Other
        ####
        random.seed(c.SEED)
        
        
    def step(self, action: NDArray[Any]) -> tuple[NDArray[Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the Environment
        """
                
        self.step_idx += 1
        
        data_log_step = l.initialize_data_log_step()
        
        valid_action = self.validate_action(action)
        
        # For h.update_data_log_step() method call
        gate_next = self.gate_set_names[action[0]] + " " + str(self.wires_permutations[int(action[1])]) 
        
        if not valid_action:
            accuracy_current = None
            reward = c.PENALTY_ILLEGAL_ACTION
            self.reward_cumulative += reward
            truncated = True
            terminated = False
        else:
            self.gate_count_remaining -= 1
            self.action_to_gate(action)
            self.update_illegal_actions(action)
            self.update_observation(action)
            self.circuit_depth = max(self.qubit_depth_idx)
            
            observation_hash = self.hash_observation()
            accuracy_current = self.get_or_calculate_accuracy(observation_hash) # Accuracy current = Accuracy test
                
            reward, terminated, truncated = self.generate_feedback(accuracy_current)
            self.reward_cumulative += reward
            
            self.accuracy_prev = accuracy_current   # Save current accuracy for calculating delta during next step
            self.accuracy_max = max(self.accuracy_max, accuracy_current) 
         
        l.update_data_log_step(
                data_log_step,
                action=[int(act) for act in action],
                gate=gate_next,
                illegal_actions=self.illegal_actions[:],
                reward=reward,
                cumulative_reward=self.reward_cumulative,
                classification_accuracy=accuracy_current,
                terminated=terminated,
                truncated=truncated,
                end_time = datetime.now().strftime("%d.%m.%Y:%H:%M:%S")
        )
        l.update_data_log_episode(
                self.data_log_episode,
                data_log_step,
                step_idx=self.step_idx
        )
        
        if terminated or truncated:
            l.update_data_log_episode_summary(
                self.data_log_episode,
                steps=self.step_idx,
                terminated=terminated,
                truncated=truncated,
                episode_reward=self.reward_cumulative,
                # observation = self.observation.tolist(),
                # circuit=[(str(gate.op), gate.wires) for gate in self.gates_applied],
                circuit_depth=max(self.qubit_depth_idx),
                weights=self.gates_params
            )
            l.update_data_log_run(
                self.data_log_run,
                self.data_log_episode,
                episode_idx=self.episode_idx
            )
            l.update_data_log_run_summary(
                self.data_log_run,
                steps=self.step_idx,
                reward=self.reward_cumulative,
                accuracy_episode_avg=self.data_log_episode["summary"]["classification_accuracy_avg"],
                accuracy_episode_max=self.data_log_episode["summary"]["classification_accuracy_max"],
                accuracy_episode_min=self.data_log_episode["summary"]["classification_accuracy_min"],
                circuit_depth=max(self.qubit_depth_idx),
            )
            
            # if self.episode_idx % 10 == 0:
            #     l.console_log(f"Episode {self.episode_idx} finished after {self.step_idx} steps | terminated: {terminated}, truncated: {truncated} | reward: {self.reward_cumulative} accuracy: {self.data_log_episode["summary"]["classification_accuracy_last"]} accuracy average: {self.data_log_episode["summary"]["classification_accuracy_avg"]}")
            # # self.render()
            
            l.console_log(f"Episode {self.episode_idx} finished after {self.step_idx} steps | terminated: {terminated}, truncated: {truncated} | reward: {self.reward_cumulative} accuracy: {self.data_log_episode["summary"]["classification_accuracy_last"]} accuracy average: {self.data_log_episode["summary"]["classification_accuracy_avg"]}")
            # self.render()
                
            if c.CIRCUIT_DEPTH_DYNAMIC:
                self.update_circuit_depth(self.accuracy_prev)
            
            if self.clear_jax_cache:
                jax.clear_caches()

        return self.observation, reward, terminated, truncated, {} # info
        
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> NDArray[Any]: 
        """
        Reset the Environment to the initial state
        """
        super().reset(seed=seed)
        self.data_log_episode = l.initialize_data_log_episode()
        self.circuit_depth = 0
        self.gate_count_remaining = self.gate_count_max
        self.gates_applied = []
        self.gates_params = []
        self.reward_cumulative = 0
        self.episode_idx += 1
        self.step_idx = 0
        self.illegal_actions = []
        self.observation = np.zeros((10, self.qubit_count, len(self.gate_set)+(self.qubit_count-1)), dtype=np.int8)
        self.qubit_depth_idx = [0] * self.qubit_count
        self.accuracy_prev = 0
        self.accuracy_average = 0
        self.accuracy_max = 0
        self.clear_jax_cache = False
        
        return self.observation, {}


    def render(self, mode: str = "human") -> None:
        """
        Render the current observation to the console
        """
        if mode == "human":
            print(qml.draw(self.full_circuit)(self.gates_params, self.X_train[0]))
           
        
    def get_wire_permutations(self, gate_set: list[Operation]) -> NDArray[Any]:
        """
        Get all possible wire permutations for the given gate set
        """
        wires = list(map(lambda gate: gate.num_wires, gate_set))
        wires_max = max(wires) if len(wires) > 0 else 0 
        wires_permutations = []
        for perm in list(permutations(range(self.qubit_count), wires_max)):
            wires_permutations.append(perm)
        return wires_permutations
    
    
    def quantum_circuit(self, params, x) -> float: 
        """
        Quantum Circuit executed by the QNode
        """
        gates_params_idx = 0
        qml.AmplitudeEmbedding(features=x, wires=range(self.qubit_count), normalize=False, validate_norm=True) # Features already normalized during preprocessing / loading data
        for gate in self.gates_applied:
            wires_num = gate.op.num_wires
            wires = gate.wires[:wires_num]
            if gate.op.num_params == 1:
                param = params[gates_params_idx]
                gate.op(param, wires=wires)
                gates_params_idx += 1
            else:
                gate.op(wires=wires)
        
        # Measure on all qubits to get full probability distribution over all states 
        # return qml.probs(wires=range(self.qubit_count))
        return qml.probs(wires=range(self.measurement_qubits))
        
        
    def action_to_gate(self, action: NDArray[Any]) -> QuantumGate:
        """
        Convert action of agent to Quantum Gate and append it to the gates_applied list
        """
        gate = self.gate_set[int(action[0])]
        wires = self.wires_permutations[int(action[1])]
        gate_config = QuantumGate(op=gate, wires=wires)
        self.gates_applied.append(gate_config)
        if gate.num_params == 1:
            self.gates_params.append(c.GATE_PARAM_DEFAULT)
        return gate_config
    
    
    def map_measurements_to_classes(self, measurements: jnp.ndarray):
        """
        Map probabilities of quantum states to class probabilities by summing over relevant states.
        """
        num_states = len(measurements)
        states_per_class = num_states // self.classes_num
        # Initialize class probabilities array
        class_probs = jnp.zeros(self.classes_num)

        # Sum probabilities for each class
        for class_idx in range(self.classes_num):
            start_idx = class_idx * states_per_class
            end_idx = start_idx + states_per_class
            class_probs = class_probs.at[class_idx].set(jnp.sum(measurements[start_idx:end_idx]))

        return class_probs

    
    def cost(self, params, X, Y):
        """
        Cost function for optimization of the Quantum Circuit parameters
        """
        loss = 0
        
        for x, y in zip(X, Y):
            probs = self.full_circuit(params, x)
            
            # Truncate probs to match the number of classes
            probs = probs[:self.relevant_measurements]  
            
            # Calculate the cross-entropy loss
            for y_i, p_i in zip(y, probs): 
                if y_i == 1:  # Avoid unnecessary computation for y_i == 0
                    loss = loss - (y_i * qml.math.log(p_i))
        
        return loss / len(X)
        
    
    def cost_batch(self, params, X, Y, batch_size: int = c.OPT_BATCH_SIZE): 
        """
        Cost function for optimization of the Quantum Circuit parameters with batch of data
        """
        idx = np.random.choice(len(X), size=batch_size, replace=False)
        X_batch = X[idx]
        Y_batch = Y[idx]
  
        return self.cost(params, X_batch, Y_batch)
    

    def inner_loop(self) -> float:
        """
        Inner loop of the RL Agent where parameters of the Quantum Circuit are optimized
        """
            
        max_params = []
        max_acc_train = 0
        max_acc_test = 0
        max_acc_total = 0
        
        batch_sizes = {
            "window": range(c.OPT_BATCH_SIZE_WINDOW[0], c.OPT_BATCH_SIZE_WINDOW[1] + 1),
            "list": c.OPT_BATCH_SIZE_LIST,
            "fixed": [c.OPT_BATCH_SIZE],
            "list_random": [random.randint(c.OPT_BATCH_SIZE_WINDOW[0], c.OPT_BATCH_SIZE_WINDOW[1]) for _ in range(c.OPT_BATCH_SIZE_LIST_RANDOM_LEN)],
            "fixed_random": [random.randint(c.OPT_BATCH_SIZE_WINDOW[0], c.OPT_BATCH_SIZE_WINDOW[1])]
        }.get(c.OPT_BATCH_MODE)
        
        seeds = list(range(1, c.GATE_PARAM_SEED_COUNT+1))   # List of seeds for random parameter initialization
        
        for batch_size in batch_sizes:
            # mode "static" for experimentation purposes > remove later 
            if c.GATE_PARAM_MODE == "static":
                params_init = [c.GATE_PARAM_DEFAULT for _ in range(len(self.gates_params))]
                best_params, best_acc_train, best_acc_test, best_acc_total = self.optimization_loop_jax(params_init, batch_size)
                
                if best_acc_test > max_acc_test or (best_acc_test == max_acc_test and best_acc_train > max_acc_train):
                    max_acc_train = best_acc_train
                    max_acc_test = best_acc_test
                    max_acc_total = best_acc_total
                    max_params = best_params
                
            elif c.GATE_PARAM_MODE == "random":
                for seed in seeds:
                    random.seed(seed)
                    if c.GATE_PARAM_VALUE_RANGE == 3.1415:
                        params_init = [random.uniform(-jnp.pi, jnp.pi) for _ in range(len(self.gates_params))]
                    else:
                        params_init = [random.uniform(-c.GATE_PARAM_VALUE_RANGE, c.GATE_PARAM_VALUE_RANGE) for _ in range(len(self.gates_params))]

                    best_params, best_acc_train, best_acc_test, best_acc_total = self.optimization_loop_jax(params_init, batch_size)
                    
                    # if c.OPT_USE_VALIDATION_BATCH:
                    #     performance_metric = self.calculate_performance_metric(weights=best_params, mode="validate")
                    # else: 
                    #     performance_metric = self.calculate_performance_metric(weights=best_params, mode="final")
                    # if performance_metric > performance_metric_max:
                    #     params_opt_max = best_params
                    #     performance_metric_max = performance_metric
                    
                    if best_acc_test > max_acc_test or (best_acc_test == max_acc_test and best_acc_train > max_acc_train):
                        max_acc_train = best_acc_train
                        max_acc_test = best_acc_test
                        max_acc_total = best_acc_total
                        max_params = best_params
        
        random.seed(c.SEED) # Reset seed to run seed for other random operations
        self.gates_params = max_params
        return max_params, max_acc_train, max_acc_test, max_acc_total


    def optimization_loop(self) -> float:  
        """
        Inner loop of the RL Agent where parameters of the Quantum Circuit are optimized
        """
        
        if self.gates_params:
            
            if c.OPT == "adam":
                opt = qml.AdamOptimizer(stepsize=0.01)
            elif c.OPT == "nesterov":
                opt = qml.NesterovMomentumOptimizer(stepsize=0.01)
            weights = pnp.array(self.gates_params, requires_grad=True)
            weights_best = weights.copy()
            cost_best = self.cost_batch(weights, self.X_train, self.y_train)
            
            for epoch in range(c.OPT_EPOCHS):
            
                weights, prev_cost = opt.step_and_cost(lambda v: self.cost_batch(v, self.X_train, self.y_train), weights)
                
                # Final optimization stage not always best, so save best weights
                if prev_cost > cost_best:
                    weights_best = weights.copy()
                    cost_best = prev_cost
                
                # TODO: Adapt and test; validate that this works as expected
                # cost.append(self.cost_batch(weights, self.X_train, self.y_train))
                # conv = np.abs(cost[-1] - prev_cost)
                # if conv <= c.OPT_CONV_TOL:
                #     break
                
                ####
                ## Uncomment code block and set flag of console_log to True for logging weight optimization progress to console
                ####
                # measurements = [self.full_circuit(weights, x) for x in self.X_train]
                # predictions = [self.measurements_to_classes(m) for m in measurements]
                # train_acc = ml.accuracy(self.y_train, predictions)
                #
                # if (epoch + 1) % 5 == 0:
                #     l.console_log(f"Epoch: {epoch+1}/{c.OPT_EPOCHS}, Loss: {prev_cost:.4f}, Train Accuracy: {train_acc:.2f}", flag=True)
                ####

            # cost_final_test = self.cost(weights, self.X_test, self.y_test)
            # measurements_test = [self.full_circuit(weights, x) for x in self.X_test]
            
            ####
            # Proof for correct implementation of cost function
            ##
            # measurements_test_check = [sublist[:self.relevant_measurements] for sublist in measurements_test]
            # cost_final_test_check = ml.cross_entropy_loss(self.y_test, measurements_test_check)
            # print(f"Final Test Loss: {cost_final_test:.4f}, Test Loss (Check): {cost_final_test_check:.4f}")
            ####
            
            self.gates_params = weights_best.tolist()


    def optimization_loop_jax(self, weights, batch_size) -> float:
        """
        JAX compatible inner loop of the RL Agent where parameters of the Quantum Circuit are optimized
        
        Parameters are optimized for training data (loss function uses training data only); best acc test and acc total tracked 
        > optimize parameters based on training data for best acc on test data 
        """
        # Just in time (JIT) compile performance relevant functions for execution speedup
        @jax.jit 
        def cost_jax(params, data, targets):
            def single_example_loss(params, x, y):
                probs = self.full_circuit(params, x)
                probs = self.map_measurements_to_classes(probs)
                return -jnp.sum(y * jnp.log(probs))

            loss = jax.vmap(single_example_loss, in_axes=(None, 0, 0))(params, data, targets)
            return jnp.mean(loss)
        
        @jax.jit
        def cost_batch_jax(params, X, Y):
            idx = jax.random.choice(jax.random.PRNGKey(0), len(X), (batch_size,), replace=False)
            X_batch = X[idx]
            Y_batch = Y[idx]

            return cost_jax(params, X_batch, Y_batch)
        
        @jax.jit 
        def update_step(i, args):
            
            ########
            ## Version without tracking and keeping best params for best acc test
            ####
            # params, opt_state, data, targets = args
            
            # loss_val, grads = jax.value_and_grad(cost_batch_jax)(params, data, targets)
            # updates, opt_state = opt.update(grads, opt_state)
            # params = optax.apply_updates(params, updates)
            
            # def print_fn():
            #     jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
        
            # jax.lax.cond((jnp.mod(i, 10) == 0), print_fn, lambda: None)
            
            # return (params, opt_state, data, targets)
            ########

            
            ########
            ## Adapted Version for keeping best parameters during optimization
            ## S. https://pennylane.ai/qml/demos/tutorial_state_preparation > search for "the final stage of optimization isn't always the best"
            ####
            params, opt_state, data, targets, best_params, best_acc_train, best_acc_test, best_acc_total = args
            
            loss_val, grads = jax.value_and_grad(cost_batch_jax)(params, data, targets)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            acc_train = self.calculate_performance_metric(params, mode="train")
            acc_test = self.calculate_performance_metric(params, mode="test")
            acc_total = (acc_train + acc_test) / 2
            
            def update_best_values(acc_train, acc_test, acc_total, params, best_acc_train, best_acc_test, best_acc_total, best_params):
                condition = (acc_test > best_acc_test) | ((acc_test == best_acc_test) & (acc_train > best_acc_train))
                return (
                    jax.lax.cond(condition, lambda: acc_train, lambda: best_acc_train),
                    jax.lax.cond(condition, lambda: acc_test, lambda: best_acc_test),
                    jax.lax.cond(condition, lambda: acc_total, lambda: best_acc_total),
                    jax.lax.cond(condition, lambda: params, lambda: best_params),
                )

            best_acc_train, best_acc_test, best_acc_total, best_params = update_best_values(
                acc_train, acc_test, acc_total, params, best_acc_train, best_acc_test, best_acc_total, best_params
            )
            
            ####
            # Print statements for debugging purposes
            ##
            # jax.debug.print("Step: {i} | Loss: {loss_val} | Params: {params} | Accuracy train: {acc_train} | Accuracy test: {acc_test} | Accuracy total: {acc_total}", 
            #                 i=i, loss_val=loss_val, params=params, acc_train=acc_train, acc_test=acc_test, acc_total=acc_total)
            
            # def print_best_results():
            #     jax.debug.print("Best params: {best_params} | Best accuracy train: {best_acc_train} | Best accuracy test: {best_acc_test} | Best accuracy total: {best_acc_total}",
            #                     best_params=best_params, best_acc_train=best_acc_train, best_acc_test=best_acc_test, best_acc_total=best_acc_total)
                
            # jax.lax.cond((i == c.OPT_EPOCHS-1), print_best_results, lambda: None)
            ####
            
            return (params, opt_state, data, targets, best_params, best_acc_train, best_acc_test, best_acc_total)
            ########
            
        @jax.jit
        def optimization_jit(args):
            results = jax.lax.fori_loop(0, c.OPT_EPOCHS, update_step, args)
            params, opt_state, data, targets, best_params, best_acc_train, best_acc_test, best_acc_total = results
            
            return params, best_params, best_acc_train, best_acc_test, best_acc_total
        
      
        optimizers = {
            "adam": optax.adam,
            "sgd": optax.sgd,
            "adagrad": optax.adagrad,
            "rmsprop": optax.rmsprop,
            "nadam": optax.nadam
        }
        
        ########
        ## Dynamic learning rate for optimizer - not yet tested
        ####
        if c.OPT_DYNAMIC_LEARNING_RATE:
            opt = optimizers.get(c.OPT, optax.adam)(h.get_opt_learning_rate(batch_size))
        else: 
            opt = optimizers.get(c.OPT, optax.adam)(c.OPT_LEARNING_RATE)
        ########
        
        data = self.X_train_jax
        targets = self.y_train_jax
        params_jax = jnp.array(weights)
        best_params = jnp.array(weights)
        opt_state = opt.init(params_jax)
        best_acc_train = float(0)
        best_acc_test = float(0)
        best_acc_total = float(0)
        args = (params_jax, opt_state, data, targets, best_params, best_acc_train, best_acc_test, best_acc_total)

        params, best_params, best_acc_train, best_acc_test, best_acc_total = optimization_jit(args)
        
        return [p.item() for p in best_params], best_acc_train.item(), best_acc_test.item(), best_acc_total.item()
            
            
    def calculate_performance_metric(self, weights, mode: Literal["train", "test", "average"]) -> float:
        
        @jax.jit
        def calculate_performance_metric_jit(weights):
            def calculate_accuracy(params, X, y):
                measurements = jax.vmap(self.full_circuit, in_axes=(None, 0))(params, X)
                
                @jax.jit
                def measurements_to_classes_jit(measurements: jnp.ndarray) -> jnp.ndarray:
                    # relevant_elements = measurements[:self.relevant_measurements]
                    relevant_elements = self.map_measurements_to_classes(measurements)
                    class_pred = jnp.zeros_like(relevant_elements)
                    class_pred = class_pred.at[jnp.argmax(relevant_elements)].set(1)
                    return class_pred
                
                predictions = jax.vmap(measurements_to_classes_jit)(measurements)
                
                @jax.jit
                def accuracy_jit(y_true, y_pred):
                    return jnp.mean(jnp.all(y_true == y_pred, axis=1))
                
                return accuracy_jit(y, predictions)
            
            params = jnp.array(weights)
            
            # if mode == "validate":
            #     train_acc = calculate_accuracy(params, self.X_train_batch_jax, self.y_train_batch_jax)
            #     test_acc = calculate_accuracy(params, self.X_test_batch_jax, self.y_test_batch_jax) 
            # elif mode == "final":
            #     train_acc = calculate_accuracy(params, self.X_train_jax, self.y_train_jax)
            #     test_acc = calculate_accuracy(params, self.X_test_jax, self.y_test_jax)
                    
            if mode == "test":
                acc = calculate_accuracy(params, self.X_test_jax, self.y_test_jax)
                return acc
            elif mode == "train":
                acc = calculate_accuracy(params, self.X_train_jax, self.y_train_jax)
                return acc
            elif mode == "average":
                acc_train = calculate_accuracy(params, self.X_train_jax, self.y_train_jax)
                acc_test = calculate_accuracy(params, self.X_test_jax, self.y_test_jax)
                acc_average = (acc_train + acc_test) / 2
                return acc_average
            
        if len(weights) == 0: 
            acc = calculate_performance_metric_jit(weights).astype(float)
            acc_float = acc.item()
            return acc_float
        else:
            return calculate_performance_metric_jit(weights).astype(float)
        

        ########
        ## Additional performance metrics for classification tasks - not yet integrated
        ####
        # test_precision = ml.precision(self.y_test, predictions_test)
        # test_recall = ml.recall(self.y_test, predictions_test)
        # test_f1 = ml.f1(self.y_test, predictions_test)
        #
        # test_precision_micro = ml.precision(self.y_test, predictions_test, average="micro")
        # test_recall_micro = ml.recall(self.y_test, predictions_test, average="micro")
        # test_f1_micro = ml.f1(self.y_test, predictions_test, average="micro")
        #
        # test_precision_macro = ml.precision(self.y_test, predictions_test, average="macro")
        # test_recall_macro = ml.recall(self.y_test, predictions_test, average="macro")
        # test_f1_macro = ml.f1(self.y_test, predictions_test, average="macro")
        #
        # test_precision_weighted = ml.precision(self.y_test, predictions_test, average="weighted")
        # test_recall_weighted = ml.recall(self.y_test, predictions_test, average="weighted")
        # test_f1_weighted = ml.f1(self.y_test, predictions_test, average="weighted")
        #
        # print(f"Final Test Accuracy: {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}")
        # print(f"Final Test Accuracy: {test_acc}, Precision (Micro): {test_precision_micro}, Recall (Micro): {test_recall_micro}, F1 Score (Micro): {test_f1_micro}")
        # print(f"Final Test Accuracy: {test_acc}, Precision (Macro): {test_precision_macro}, Recall (Macro): {test_recall_macro}, F1 Score (Macro): {test_f1_macro}")
        # print(f"Final Test Accuracy: {test_acc}, Precision (Weighted): {test_precision_weighted}, Recall (Weighted): {test_recall_weighted}, F1 Score (Weighted): {test_f1_weighted}")  
        ########

    
    def validate_action(self, action: NDArray[Any]) -> bool:
        """
        Validate action taken by the agent based on several pre-defined criteria
        """
        # Convert action to respective format of illegal action list entries 
        gate_idx = int(action[0])
        gate = self.gate_set[gate_idx]
        if gate.num_wires == 1:
            wires = int(self.wires_permutations[int(action[1])][0])
        elif gate.num_wires == 2:
            wires = [int(wire) for wire in self.wires_permutations[action[1]]]
        
        # If action is contained in illegal actions list, action is invalid
        if len(self.illegal_actions) != 0:
            if [gate_idx, wires] in self.illegal_actions:
                return False
        
        # If max circuit depth is exceeded through action, action is invalid
        if c.CIRCUIT_DEPTH_DYNAMIC: 
            if not self.validate_circuit_depth(gate, wires, self.circuit_depth_max_dynamic):
                return False
        else: 
            if not self.validate_circuit_depth(gate, wires, self.circuit_depth_max):
                return False
        
        # If no gates are remaining, action is invalid
        if self.gate_count_remaining < 0:
            return False
        
        return True
        
        
    def validate_circuit_depth(self, gate: QuantumGate, wires: int | list[int], circuit_depth_max: int) -> bool:
        """
        Method that validates that the maximum allowed circuit depth is not exceeded by the action taken by the agent
        """
        if gate.num_wires == 1:
            if self.qubit_depth_idx[wires] >= circuit_depth_max:
                return False
            else:
                return True
        elif gate.num_wires == 2:
            if any(self.qubit_depth_idx[wire] >= circuit_depth_max for wire in wires):
                return False
            else:
                return True
    

    def generate_feedback(self, accuracy: float):
        """
        Generate feedback for the agent based on the current classification accuracy of the quantum circuit
        """
        terminated = accuracy >= self.performance_target
        truncated = False
        reward = self.calculate_reward(accuracy, terminated)
        
        return reward, terminated, truncated
        
        
    def calculate_reward(self, accuracy: float, terminated: bool) -> float:
        """
        Calculate the reward
        """
        
        ####
        ## Reward shaping components
        ####
        # Further components one can use for experimenting with reward shaping 
        # 
        # if self.accuracy_prev == 0:
        #    accuracy_delta_scaled = accuracy
        #    self.accuracy_average = accuracy
        # else: 
        #    accuracy_delta_scaled = (accuracy_delta**2 * 100) if accuracy_delta > 0 else -(accuracy_delta**2 * 100)
        #    self.accuracy_average = (self.accuracy_average * (self.step_idx - 1) + accuracy) / self.step_idx
        #
        # circuit_depth_remaining_percent = 1 - circuit_depth_used_percent
        # gates_remaining_percent = 1 - gates_used_percent
        # 
        ####
        
        # Calculate required components for reward shaping
        if self.accuracy_prev == 0:
            accuracy_delta = accuracy
            accuracy_delta_positive = accuracy
            self.accuracy_average = accuracy
            self.accuracy_prev = accuracy
            accuracy_delta_compared_to_max = accuracy
        else: 
            accuracy_delta = accuracy - self.accuracy_prev
            accuracy_delta_positive = accuracy_delta if accuracy_delta > 0 else 0
            accuracy_delta_compared_to_max = accuracy - self.accuracy_max
            
        circuit_depth_remaining = self.circuit_depth_max - self.circuit_depth
        gates_remaining = self.gate_count_remaining
        complexity_remaining = (circuit_depth_remaining + gates_remaining) / 2
        horizon_extension_reward = self.circuit_depth_max * 10
        
        ####
        ## For scaled version of reward mode "performance_complexity"
        ####
        # circuit_depth_remaining_percent = circuit_depth_remaining / self.circuit_depth_max
        # gates_remaining_percent = gates_remaining / self.gate_count_max
        # complexity_remaining_percent = (circuit_depth_remaining_percent + gates_remaining_percent) / 2
        # horizon_extension_reward_percent = 1  
        ####
     
        ####
        ## Choose reward mode
        ####
        reward = 0
        if c.REWARD_MODE == "performance_complexity":
            ##
            # Balances performance and complexity; reward for accuracy_delta > 0; penalty for accuracy_delta < 0
            ##
            # reward += 0.5 * accuracy_delta + accuracy_delta * complexity_remaining
            reward += (0.5 * accuracy_delta + accuracy_delta * (complexity_remaining + horizon_extension_reward)) / 10
            
            # Scaled version; remember to uncommend variables above
            # reward += 0.5 * accuracy_delta + accuracy_delta * (complexity_remaining_percent + horizon_extension_reward_percent)
        elif c.REWARD_MODE == "performance_complexity_positive":
            ##
            # Balances performance and complexity; reward for accuracy_delta > 0; no penalty for accuracy_delta < 0
            ##
            reward += 0.5 * accuracy_delta_positive + accuracy_delta_positive * complexity_remaining
        elif c.REWARD_MODE == "performance_complexity_positive_max":
            ##
            # Balances performance and complexity; reward for accuracy_delta > 0, but only if accuracy is greater than maximum accuracy; no penalty for accuracy_delta <= 0
            ##
            if accuracy_delta_compared_to_max > 0:
                reward += 0.5 * accuracy_delta_compared_to_max + accuracy_delta_compared_to_max * complexity_remaining
        elif c.REWARD_MODE == "performance_complexity_scaled":
            ##
            # Balances performance and complexity in a non-linear way; only rewards, no penalties; discount factor based on step count
            ##
            if accuracy < 0.5:
                reward += accuracy
            else:
                circuit_depth_used_percent = self.circuit_depth / self.circuit_depth_max
                gates_used_percent = (self.gate_count_max - self.gate_count_remaining) / self.gate_count_max
                accuracy_component = (accuracy / 1.01 - accuracy)
                depth_component = (circuit_depth_used_percent / 1.01 - circuit_depth_used_percent)
                gate_component = (gates_used_percent / 1.01 - gates_used_percent)
                complexity_component = (depth_component + gate_component) / 2
                reward += (0.99 ** self.step_idx) * (accuracy_component / complexity_component)
        elif c.REWARD_MODE == "performance":
            ##
            # Only performance considered; complexity not taken into account
            ##
            reward += accuracy
        elif c.REWARD_MODE == "performance_delta":
            ##
            # Only performance considered; complexity not taken into account; reward for accuracy_delta > 0; penalty for accuracy_delta < 0
            ##
            reward += accuracy_delta
        elif c.REWARD_MODE == "performance_delta_positive":
            ##
            # Only performance considered; complexity not taken into account; reward for accuracy_delta > 0; no penalty for accuracy_delta <= 0
            ##
            reward += accuracy_delta_positive
        
        ####
        # Uncomment if you want to add a performance bonus for high accuracies
        ####    
        if c.BONUS_REWARD:
            if accuracy >= c.BONUS_REWARD_THRESHOLD:   
                reward += accuracy
        ####
     
        if terminated:
            reward += 100
        
        return reward

 
    def update_observation(self, action: NDArray[Any]) -> None:
        """
        Update the observation of the environment
        """
        gate_idx = action[0]
        gate = self.gate_set[gate_idx]
        wires = self.wires_permutations[int(action[1])]
        
        if gate == qml.CNOT:
            max_qubit_depth_idx = max(self.qubit_depth_idx[wires[0]], self.qubit_depth_idx[wires[1]])
            self.observation[max_qubit_depth_idx, wires[0], gate_idx + wires[0]] = 1
            self.observation[max_qubit_depth_idx, wires[1], gate_idx + wires[0]] = 1
            self.qubit_depth_idx[wires[0]] = max_qubit_depth_idx + 1
            self.qubit_depth_idx[wires[1]] = max_qubit_depth_idx + 1
        else: 
            self.observation[self.qubit_depth_idx[wires[0]], wires[0], gate_idx] = 1
            self.qubit_depth_idx[wires[0]] += 1

    
    def update_illegal_actions(self, action: NDArray[Any]) -> None:
        """
        Update the list of illegal actions based on the current action taken by the agent
        """        
        gate = self.gate_set[int(action[0])]
        if gate.num_wires == 1:
            wires = int(self.wires_permutations[int(action[1])][0])
        elif gate.num_wires == 2:
            wires = [int(wire) for wire in self.wires_permutations[int(action[1])]]
        
        
        # Define conditions for updating illegal actions
        def condition_1_qubit_gates(act):
            return (
                (self.gate_set[act[0]].num_wires == 1 and wires == act[1]) or
                (self.gate_set[act[0]].num_wires == 2 and wires in act[1])
            )
            
        def condition_2_qubit_gates(act):
            return (
                (self.gate_set[act[0]].num_wires == 1 and act[1] in wires) or
                (self.gate_set[act[0]].num_wires == 2 and any(x in act[1] for x in wires))
            )
        
        # If next gate is 1 Qubit gate, remove 1 Qubit and 2 Qubit gates on respective wire
        if gate.num_wires == 1:
            self.illegal_actions = [act for act in self.illegal_actions if not condition_1_qubit_gates(act)]
        # If next gate is 2 Qubit gate, remove 1 Qubit and 2 Qubit gates on both wires
        elif gate.num_wires == 2:
            self.illegal_actions = [act for act in self.illegal_actions if not condition_2_qubit_gates(act)]
          
        # Add current action to illegal actions for future steps
        if gate.num_wires == 1:
            self.illegal_actions.append([int(action[0]), wires])
        elif gate.num_wires == 2:
            self.illegal_actions.append([int(action[0]), wires])
            
            
    def update_circuit_depth(self, accuracy: int):
        """
        Update the allowed circuit depth (in case of dynamic circuit depth)
        """
        
        self.accuracy_history.append(accuracy)
        
        # If window size reached, calculate accuracy change
        if len(self.accuracy_history) >= c.CIRCUIT_DEPTH_WINDOW_SIZE:
            accuracy_max = max(self.accuracy_history)
            accuracy_min = min(self.accuracy_history)
            accuracy_change = accuracy_max - accuracy_min
            
            # Check if accuracy change is below threshold
            if accuracy_change < c.CIRCUIT_DEPTH_STAGNATION_THRESHOLD:
                # If final circuit depth max isn't reached yet, increase dynamic circuit depth max
                if self.circuit_depth_max_dynamic < self.circuit_depth_max:
                    self.circuit_depth_max_dynamic += c.CIRCUIT_DEPTH_DYNAMIC_INCR
                    l.console_log(f"################ Dynamic maximum circuit depth increased by {c.CIRCUIT_DEPTH_DYNAMIC_INCR}. New dynamic maximum circuit depth is {self.circuit_depth_max_dynamic} ################")
                    
                    if self.circuit_depth_max_dynamic == self.circuit_depth_max: 
                        l.console_log(f"################ Final maximum circuit depth of {self.circuit_depth_max_dynamic} reached ################")
                    
                    # Clear history    
                    self.accuracy_history = collections.deque(maxlen=c.CIRCUIT_DEPTH_WINDOW_SIZE)
                    
       
    def hash_observation(self):
        """
        Hash the observation of the environment to create a unique identifier for the current circuit
        """
        
        observation_bytes = np.packbits(self.observation.flatten())
        observation_hash = hashlib.sha256(observation_bytes).hexdigest()
        return observation_hash
    
    
    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of the quantum circuit
        """
        if self.gates_params:
            max_params, max_acc_train, max_acc_test, max_acc_total = self.inner_loop()
        else:
            max_acc_test = self.calculate_performance_metric(weights=self.gates_params, mode="test")
        return max_acc_test


    def get_or_calculate_accuracy(self, observation_hash: str):
        """
        Get the accuracy of the quantum circuit from the cache or calculate it if not available
        """
        
        if self.circuit_accuracy_cache.get(observation_hash):
            # l.console_log(f"Accuracy for observation {observation_hash} found in cache; stored accuracy {self.circuit_accuracy_cache[observation_hash]}")
            return self.circuit_accuracy_cache.get(observation_hash)
        else:
            accuracy = self.calculate_accuracy()
            self.circuit_accuracy_cache.set(observation_hash, accuracy)
            # l.console_log(f"Accuracy for observation {observation_hash} not found in cache; calculated accuracy {accuracy} stored in cache")
            self.clear_jax_cache = True
            return accuracy
            
    

if __name__=="__main__":
    
    pass
    
    ####
    ## Custom Env Conformity Check
    ####
    # gym.pprint_registry()
    # env = gym.make("QuantumCircuit-v0")
    # check_env(env, warn=True)
    ####
    