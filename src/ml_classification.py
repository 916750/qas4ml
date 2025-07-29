"""Methods related to classification tasks in machine learning"""

import math
import os
import pickle
import random
from typing import Literal

import config as c
import helpers as h
import jax
import jax.numpy as jnp
import log as l
import numpy as np
import pennylane as qml
import rl_env_cl as rl_cl
from sklearn import datasets, metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random.seed(c.SEED)


def preprocess_iris_data(
    classes: Literal[2, 3], filepath: str, classes_selected: tuple = None
):
    """
    Preprocesses Iris dataset for classification tasks and saves it to a file.
    Allows selection of specific classes when 'classes' is 2.
    """

    h.create_directory(filepath)

    if classes == 2 and (not classes_selected or len(classes_selected) != 2):
        raise ValueError(
            "For binary classification (classes=2), 'classes_selected' must be a tuple of two class indices."
        )

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    if classes == 2:
        selected_target_names = [target_names[i] for i in classes_selected]
        l.console_log(
            f"Selected target names for binary classification: {selected_target_names}"
        )
    else:
        l.console_log(
            f"Target names for multi-class classification: {list(target_names)}"
        )

    # Filter data based on selected classes
    if classes == 2:
        filter_mask = np.isin(y, classes_selected)
    else:
        filter_mask = np.isin(y, list(range(classes)))

    X = X[filter_mask]
    y = y[filter_mask]

    # Re-map labels to 0 and 1 for binary classification
    if classes == 2:
        y = np.where(y == classes_selected[0], 0, 1)

    # Feature scaling: Normalize data (L2-Norm)
    X_normalized = preprocessing.normalize(X, norm="l2")

    # One-Hot-Encoding for labels
    encoder = preprocessing.OneHotEncoder(sparse_output=False, dtype=int)
    y_one_hot_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized,
        y_one_hot_encoded,
        test_size=c.TEST_SIZE,
        random_state=c.TEST_RANDOM,
        shuffle=True,
        stratify=y_one_hot_encoded,
    )

    # Save preprocessed data
    with open(filepath, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    l.console_log(f"Preprocessed Iris data saved in '{filepath}'.")


def preprocess_mnist_data(classes: Literal[2, 3, 4, 5, 6, 7, 8, 9, 10], filepath: str):
    """
    Preprocesses MNIST dataset for classification tasks and saves it to a file.
    """
    h.create_directory(filepath)

    X, y = datasets.load_digits(return_X_y=True)

    filter_mask = np.isin(y, list(range(classes)))

    X = X[filter_mask]
    y = y[filter_mask]

    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    # 97.60% variance retained with 32 components
    pca = PCA(n_components=32)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    l.console_log(
        f"PCA applied: {pca.n_components_} components retained, explaining {explained_variance:.2%} of variance."
    )

    # Feature scaling: Normalize data (L2-Norm)
    X_normalized = preprocessing.normalize(X_pca, norm="l2")

    # Validate normalization
    norms = np.linalg.norm(X_normalized, axis=1)
    if np.allclose(norms, 1):
        l.console_log("L2 normalization successful: All rows have a norm of 1.")
    else:
        l.console_log("Warning: L2 normalization failed for some rows. Check the data.")

    # One-Hot-Encoding for labels
    encoder = preprocessing.OneHotEncoder(sparse_output=False, dtype=int)
    y_one_hot_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized,
        y_one_hot_encoded,
        test_size=c.TEST_SIZE,
        random_state=c.TEST_RANDOM,
        shuffle=True,
        stratify=y_one_hot_encoded,
    )

    # Save preprocessed data
    with open(filepath, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    l.console_log(f"Preprocessed MNIST data saved in '{filepath}'.")


def preprocess_wine_data(
    classes: Literal[2, 3], filepath: str, classes_selected: tuple = None
):
    """
    Preprocesses Wine dataset for classification tasks and saves it to a file.
    Ensures a balanced dataset by equalizing the number of samples per class.
    Pads features to 16 dimensions for compatibility with 4 qubits.
    Allows selection of specific classes when 'classes' is 2.
    """

    h.create_directory(filepath)

    if classes == 2 and (not classes_selected or len(classes_selected) != 2):
        raise ValueError(
            "For binary classification (classes=2), 'classes_selected' must be a tuple of two class indices."
        )

    # Load Wine dataset
    wine = datasets.load_wine()
    X, y = wine.data, wine.target
    target_names = wine.target_names

    # Log the target names of the selected classes
    if classes == 2:
        selected_target_names = [target_names[i] for i in classes_selected]
        l.console_log(
            f"Selected target names for binary classification: {selected_target_names}"
        )
    else:
        l.console_log(
            f"Target names for multi-class classification: {list(target_names)}"
        )

    # Filter data based on selected classes
    if classes == 2:
        filter_mask = np.isin(y, classes_selected)
    else:
        filter_mask = np.isin(y, list(range(classes)))

    X = X[filter_mask]
    y = y[filter_mask]

    # Re-map labels to 0 and 1 for binary classification
    if classes == 2:
        y = np.where(y == classes_selected[0], 0, 1)

    # Balance the dataset by equalizing the number of samples per class
    min_samples_per_class = min(np.bincount(y))
    balanced_indices = []
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        balanced_indices.extend(
            np.random.choice(class_indices, min_samples_per_class, replace=False)
        )

    X = X[balanced_indices]
    y = y[balanced_indices]

    # Pad features to 16 dimensions
    X_padded = np.pad(X, ((0, 0), (0, 16 - X.shape[1])), mode="constant")

    # Feature scaling: Normalize data (L2-Norm)
    X_normalized = preprocessing.normalize(X_padded, norm="l2")

    # One-Hot-Encoding for labels
    encoder = preprocessing.OneHotEncoder(sparse_output=False, dtype=int)
    y_one_hot_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized,
        y_one_hot_encoded,
        test_size=c.TEST_SIZE,
        random_state=c.TEST_RANDOM,
        shuffle=True,
        stratify=y_one_hot_encoded,
    )

    # Save preprocessed data
    with open(filepath, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    l.console_log(f"Preprocessed Wine data saved in '{filepath}'.")


def load_preprocessed_data(split_train_test: bool):
    """
    Loads preprocessed data from a file.
    """

    filepath = c.PATH_CLASS_DATA

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File '{filepath}' not found. Please preprocess data first or insert correct file path."
        )

    with open(filepath, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    l.console_log(f"Preprocessed data loaded from '{filepath}'.")

    if split_train_test:
        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
            X_train,
            y_train,
            test_size=c.TEST_SIZE,
            random_state=c.TEST_RANDOM,
            shuffle=True,
            stratify=y_train,
        )
        return X_train_train, X_train_test, y_train_train, y_train_test

    return X_train, X_test, y_train, y_test


def get_data_batch(
    data: np.ndarray, targets: np.ndarray, num_classes: int, data_set_fraction: float
):
    """
    Selects a balanced batch of data and targets from the given dataset.
    """
    if not (0 < data_set_fraction <= 1):
        raise ValueError("data_set_fraction must be between 0 and 1")

    total_samples = int(len(data) * data_set_fraction)
    samples_per_class = total_samples // num_classes

    data_batch = []
    targets_batch = []

    for class_idx in range(num_classes):
        class_mask = np.argmax(targets, axis=1) == class_idx
        class_data = data[class_mask]
        class_targets = targets[class_mask]

        if len(class_data) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_idx} to create a balanced batch"
            )

        selected_indices = np.random.choice(
            len(class_data), samples_per_class, replace=False
        )
        data_batch.append(class_data[selected_indices])
        targets_batch.append(class_targets[selected_indices])

    data_batch = np.concatenate(data_batch, axis=0)
    targets_batch = np.concatenate(targets_batch, axis=0)

    # Shuffle the batch
    indices = np.random.permutation(len(data_batch))
    data_batch = data_batch[indices]
    targets_batch = targets_batch[indices]

    return data_batch, targets_batch


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates Accuracy of classification (for validation of custom implementation)
    """
    return metrics.accuracy_score(labels, predictions)


def precision(
    labels: np.ndarray, predictions: np.ndarray, average: str = None, zero_division=1.0
) -> float:
    """
    Calculates Precision of classification (for validation of custom implementation)
    """
    return metrics.precision_score(
        labels, predictions, average=average, zero_division=zero_division
    )


def recall(
    labels: np.ndarray, predictions: np.ndarray, average: str = None, zero_division=1.0
) -> float:
    """
    Calculates Recall of classification (for validation of custom implementation)
    """
    return metrics.recall_score(
        labels, predictions, average=average, zero_division=zero_division
    )


def f1(
    labels: np.ndarray, predictions: np.ndarray, average: str = None, zero_division=1.0
) -> float:
    """
    Calculates F1 Score of classification (for validation of custom implementation)
    """
    return metrics.f1_score(
        labels, predictions, average=average, zero_division=zero_division
    )


def cross_entropy_loss(labels: np.ndarray, measurements: list) -> float:
    """
    Calculates Cross-Entropy Loss of classification (for validation of custom implementation)
    """
    return metrics.log_loss(labels, measurements)


def get_classes_number(classification_task) -> int:
    """
    Resolve the number of classes for the classification task
    """
    class_task_to_classes = {
        "iris": 3,
        "iris_2_0-1": 2,
        "iris_2_0-2": 2,
        "iris_2_1-2": 2,
        "mnist": 10,
        "mnist_2": 2,
        "mnist_3": 3,
        "mnist_4": 4,
        "mnist_5": 5,
        "mnist_6": 6,
        "mnist_7": 7,
        "mnist_8": 8,
        "mnist_9": 9,
        "wine": 3,
    }

    if classification_task not in class_task_to_classes:
        raise ValueError(f"Invalid classification task: '{classification_task}'.")

    return class_task_to_classes[classification_task]


def get_required_qubit_number(classification_task) -> int:
    """
    Resolve the number of qubits required for the classification task
    """
    if "iris" in classification_task:
        return 2
    elif "mnist" in classification_task:
        return 5
    elif "wine" in classification_task:
        return 4
    else:
        raise ValueError(f"Invalid classification task: '{classification_task}'.")


def get_required_measurement_qubits(
    classes_number: int,
    measurement_mode: Literal["minimum", "all"],
    classification_task: str,
) -> int:
    """
    Get the number of qubits required for measurements in the Quantum Circuit
    """
    try:
        if measurement_mode == "minimum":
            # Measure on the minimum number of qubits required to distinguish the classes
            return math.ceil(math.log2(classes_number))
        elif measurement_mode == "all":
            # Measure on all qubits
            return get_required_qubit_number(classification_task)
        else:
            raise ValueError(
                f"Invalid measurement_mode: '{measurement_mode}'. Expected 'minimum' or 'all'."
            )
    except (ValueError, TypeError) as e:
        l.console_log(f"Error in get_required_measurement_qubits: {e}")
        raise


def get_relevant_measurements_number(
    classes_number: int,
    measurement_mode: Literal["minimum", "all"],
    classification_task: str,
):
    """
    Get the number of relevant measurements for the Quantum Circuit
    """

    try:
        if measurement_mode == "minimum":
            return classes_number
        elif measurement_mode == "all":
            quibt_count = get_required_qubit_number(classification_task)
            measurement_states = 2**quibt_count
            return (measurement_states // classes_number) * classes_number
        else:
            raise ValueError(
                f"Invalid measurement_mode: '{measurement_mode}'. Expected 'minimum' or 'all'."
            )
    except (ValueError, TypeError) as e:
        l.console_log(f"Error in get_relevant_measurements_number: {e}")
        raise


def get_max_batch_size(classification_task: str) -> int:
    """
    Get the maximum batch size for the respective classification task
    """
    class_task_to_batch_size = {
        "iris": 105,
        "iris_2_0-1": 70,
        "iris_2_0-2": 70,
        "iris_2_1-2": 70,
        "mnist": 1257,
        "mnist_2": 252,
        "mnist_3": 375,
        "mnist_4": 504,
        "mnist_5": 630,
        "mnist_6": 758,
        "mnist_7": 884,
        "mnist_8": 1010,
        "mnist_9": 1131,
        "wine": 100,
    }

    if classification_task not in class_task_to_batch_size:
        raise ValueError(f"Invalid classification task: '{classification_task}'.")

    return class_task_to_batch_size[classification_task]


if __name__ == "__main__":
    # python ./ml_classification.py --ct 'wine' --cn 0 --rn 1 --cd 4 --rm 0 --ts 150000 --ep 2 --st 1024 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm 'window' --obs_win 1 252 --obs_list 2 --obs_fix 20 --br 'false' --brt 0.95 --olr 0.01 --opt 'adam' --oep 1000 --gpm 'random' --gpv 1.0 --gps 5

    ####
    # Preparing Data Sets
    ####

    ## Create data set files

    # preprocess_iris_data_2(2, path_1, (0,1))
    # preprocess_iris_data_2(2, path_2, (0,2))
    # preprocess_iris_data_2(2, path_3, (1,2))

    # preprocess_wine_data(3, filepath)

    # ## Validate proper creation of data set

    # env = rl_cl.QuantumCircuit()

    # print(f"X Train Shape: {env.X_train.shape}, Y Train Shape: {env.y_train.shape}")
    # print(f"Class distribution in Train: {env.y_train.sum(axis=0)}")
    # print(f"X Test Shape: {env.X_test.shape}, Y Test Shape: {env.y_test.shape}")
    # print(f"Class distribution in Test: {env.y_test.sum(axis=0)}")

    # print("############### Batch #################")
    # print(f"X Train Batch Shape: {env.X_train_batch.shape}, Y Train Batch Shape: {env.y_train_batch.shape}")
    # print(f"Class distribution in Train Batch: {env.y_train_batch.sum(axis=0)}")
    # print(f"X Test Batch Shape: {env.X_test_batch.shape}, Y Test Batch Shape: {env.y_test_batch.shape}")
    # print(f"Class distribution in Test Batch: {env.y_test_batch.sum(axis=0)}")

    # print("############### Single Data Point Example #################")
    # print(f"Single Data Point: {env.X_train_batch[0]}")
    # print(f"Single Label: {env.y_train_batch[0]}")

    pass
