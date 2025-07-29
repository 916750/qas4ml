import json
import os
from typing import Literal

import config as c
import log as l
import matplotlib.pyplot as plt
import numpy as np


def plot_episode(
    json_file_path,
    episode_number,
    metric: Literal["reward", "cumulative_reward", "classification_accuracy"],
):
    """
    Parses a JSON file containing RL training metrics and creates a plot of rewards vs step number for a specific episode.
    Hint: Step level data logging has to be enabled; otherwise JSON doesn't contain the required data
    """
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)

        if "episodes" not in data:
            print("Error: The JSON file does not contain the 'episodes' key.")
            return

        episode_key = f"episode_{episode_number}"

        if episode_key not in data["episodes"]:
            raise ValueError(
                f"Error: Episode {episode_number} not found in the JSON file."
            )

        episode_data = data["episodes"][episode_key]

        if "steps" not in episode_data:
            raise ValueError(
                f"Error: Episode {episode_number} does not contain 'steps' key."
            )

        steps = episode_data["steps"]

        step_numbers = []
        step_metrics = []

        for step_key, step_data in steps.items():
            if f"{metric}" in step_data:
                step_number = int(step_key.split("_")[1])
                step_metric = step_data[f"{metric}"]

                step_numbers.append(step_number)
                step_metrics.append(step_metric)
            else:
                raise ValueError(
                    f"Error: Metric '{metric}' not found in step {step_key}."
                )

        sorted_indices = sorted(range(len(step_numbers)), key=lambda i: step_numbers[i])
        step_numbers = [step_numbers[i] for i in sorted_indices]
        step_metrics = [step_metrics[i] for i in sorted_indices]

        metric_labels = {
            "reward": "Reward",
            "cumulative_reward": "Cumulative Reward",
            "classification_accuracy": "Classification Accuracy",
        }
        metric_label = metric_labels.get(metric)

        plt.figure(figsize=(10, 6))
        plt.plot(
            step_numbers,
            step_metrics,
            marker="o",
            label=f"{metric_label} for each Step in Episode {episode_number}",
        )
        plt.title(f"{metric_label} per Step for Episode {episode_number}")
        plt.xlabel("Step Number")
        plt.ylabel(f"{metric_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        figure_file_path = (
            c.PATH_RUN + c.PATH_PLOT + f"{metric}_episode_{episode_number}.svg"
        )
        plt.savefig(figure_file_path, format="svg")
        plt.close()
        l.console_log(f"Plot saved to {figure_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at path {json_file_path}.")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Ensure it is properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_run(
    json_file_path: str,
    metric: Literal[
        "episode_reward", "classification_accuracy_last", "steps", "circuit_depth"
    ],
):
    """
    Parses a JSON file containing RL training metrics and creates a plot of the average episode_reward over episodes vs episode number.
    Method used for automatic plot generation after a run has finished.
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
        window_size = 100
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
            "classification_accuracy_last": "Accuracy",
            "steps": "Quantum Gates used",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric)

        plt.figure(figsize=(10, 6))
        plt.plot(
            averaged_episode_numbers,
            averaged_metrics,
            label=f"Average Episode {metric_label} (75 episodes)",
        )
        plt.xlabel("Episodes")
        plt.ylabel(f"{metric_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        figure_file_path = c.PATH_RUN + c.PATH_PLOT + f"{metric}_per_episode.svg"

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
    *json_file_paths: str,
):
    """
    Parses multiple JSON files containing RL training metrics and creates a plot of the mean and standard deviation
    of the specified metric over episodes across all provided JSON files, smoothed over a window of x episodes.
    The parsed data is also saved to a JSON file.
    Method used for automatic plot generation after all runs have finished.
    """
    try:
        all_episode_numbers = []
        all_episode_metrics = []

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

        # Decrease / Increase window size for less / more smoothing
        window_size = 75
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
            "classification_accuracy_last": "Classification Accuracy",
            "steps": "Quantum Gates used",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            smoothed_episode_numbers,
            smoothed_mean_metrics,
            label=f"Mean Episode {metric_label} (75 episodes)",
        )
        plt.fill_between(
            smoothed_episode_numbers,
            np.array(smoothed_mean_metrics) - np.array(smoothed_std_metrics),
            np.array(smoothed_mean_metrics) + np.array(smoothed_std_metrics),
            alpha=0.2,
            label=f"Standard Deviation",
        )
        plt.title(f"{metric_label} per Episode")
        plt.xlabel("Episode Number")
        plt.ylabel(f"{metric_label} per Episode")
        plt.grid(True)
        plt.legend()
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


def plot_compare(
    metric: Literal[
        "episode_reward", "classification_accuracy_last", "steps", "circuit_depth"
    ],
    target_file_path: str,
    compare_level: Literal["batch_win", "depth", "config", "rl_alg"],
    *json_file_paths: str,
):
    """
    Compares the specified metric across multiple JSON files generated by plot_runs and creates a combined plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        labels = []

        for json_file_path in json_file_paths:
            with open(json_file_path, "r") as file:
                data = json.load(file)

            if (
                "episode_numbers" not in data
                or "mean_metrics" not in data
                or "std_metrics" not in data
            ):
                raise ValueError(
                    f"Error: The JSON file {json_file_path} does not contain the required keys."
                )

            episode_numbers = data["episode_numbers"]
            mean_metrics = data["mean_metrics"]
            std_metrics = data["std_metrics"]

            if compare_level == "batch_win":
                batch_size_window = data["header"]["OPT_BATCH_SIZE_WINDOW"]
                min_value = batch_size_window[0]
                max_value = batch_size_window[1]
                label = f"Batch Size {min_value} to {max_value}"
                labels.append(f"b{min_value}_b{max_value}")
            elif compare_level == "depth":
                max_depth = data["header"]["CIRCUIT_DEPTH_MAX"]
                batch_size_window = data["header"]["OPT_BATCH_SIZE_WINDOW"]
                min_value = batch_size_window[0]
                label = f"Max Circuit Depth {max_depth}"
                labels.append(f"d{max_depth}_b{min_value}")
            else:
                raise ValueError(f"Error: Unsupported compare_level '{compare_level}'.")

            plt.plot(episode_numbers, mean_metrics, label=label)
            plt.fill_between(
                episode_numbers,
                np.array(mean_metrics) - np.array(std_metrics),
                np.array(mean_metrics) + np.array(std_metrics),
                alpha=0.2,
            )

        metric_labels = {
            "episode_reward": "Reward",
            "classification_accuracy_last": "Classification Accuracy",
            "steps": "Quantum Gates used",
            "circuit_depth": "Circuit Depth",
        }
        metric_label = metric_labels.get(metric)

        plt.title(f"Comparison of {metric_label} per Episode")
        plt.xlabel("Episode Number")
        plt.ylabel(f"{metric_label} per Episode")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        os.makedirs(target_file_path + "/00_plots", exist_ok=True)
        if compare_level == "batch_win" and len(json_file_paths) == 11:
            labels_str = "all_batch_windows"
        else:
            labels_str = "_".join(labels)
        figure_file_path = (
            target_file_path
            + f"/00_plots/{metric}_per_episode_mean_std_{labels_str}.svg"
        )
        plt.savefig(figure_file_path, format="svg")
        plt.close()
        l.console_log(f"Comparison plot saved to {figure_file_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Ensure it is properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    ####
    ## For Windows OS
    ####
    metrics = [
        "episode_reward",
        "classification_accuracy_last",
        "steps",
        "circuit_depth",
    ]

    if c.OPT_BATCH_MODE == "window":
        batch_win = [
            "1_9",
            "10_19",
            "20_29",
            "30_39",
            "40_49",
            "50_59",
            "60_69",
            "70_79",
            "80_89",
            "90_99",
            "100_105",
        ]
        # Creating runs plot
        for metric in metrics:
            for bw in batch_win:
                plot_runs(metric, target_path, file1, file2, file3)

        # Creating compare plots

        for metric in metrics:
            plot_compare(
                metric,
                target_path,
                "batch_win",
                file1,
                file2,
                file3,
                file4,
                file5,
                file6,
                file7,
                file8,
                file9,
                file10,
                file11,
            )

    elif c.OPT_BATCH_MODE == "list":
        # Creating runs plot
        for metric in metrics:
            plot_runs(metric, target_path, file1, file2, file3)

    elif c.OPT_BATCH_MODE == "fixed":
        # Creating runs plot
        for metric in metrics:
            plot_runs(metric, target_path, file1, file2, file3)

    ####
    ## For Linux OS
    ####
    # metrics = ["episode_reward", "classification_accuracy_last", "steps", "circuit_depth"]

    # if c.OPT_BATCH_MODE == 'window':

    #     batch_win = ["1_9", "10_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_79", "80_89", "90_99", "100_105"]
    #     # Creating runs plot
    #     for metric in metrics:
    #         for bw in batch_win:
    #             target_path = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_{bw}"
    #             file1 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_{bw}/run_1/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #             file2 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_{bw}/run_2/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #             file3 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_{bw}/run_3/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #             plot_runs(metric, target_path, file1, file2, file3)

    #     # Creating compare plots

    #     for metric in metrics:
    #         target_path = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/"
    #         file1 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_1_9/00_plots/{metric}_per_episode_mean_std.json"
    #         file2 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_10_19/00_plots/{metric}_per_episode_mean_std.json"
    #         file3 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_20_29/00_plots/{metric}_per_episode_mean_std.json"
    #         file4 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_30_39/00_plots/{metric}_per_episode_mean_std.json"
    #         file5 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_40_49/00_plots/{metric}_per_episode_mean_std.json"
    #         file6 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_50_59/00_plots/{metric}_per_episode_mean_std.json"
    #         file7 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_60_69/00_plots/{metric}_per_episode_mean_std.json"
    #         file8 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_70_79/00_plots/{metric}_per_episode_mean_std.json"
    #         file9 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_80_89/00_plots/{metric}_per_episode_mean_std.json"
    #         file10 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_90_99/00_plots/{metric}_per_episode_mean_std.json"
    #         file11 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_win_100_105/00_plots/{metric}_per_episode_mean_std.json"
    #         plot_compare(metric, target_path, "batch_win", file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11)

    # elif c.OPT_BATCH_MODE == 'list':

    #     # Creating runs plot
    #     for metric in metrics:
    #         target_path = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_list_idx_{c.OPT_BATCH_SIZE_LIST_IDX}"
    #         file1 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_list_idx_{c.OPT_BATCH_SIZE_LIST_IDX}/run_1/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #         file2 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_list_idx_{c.OPT_BATCH_SIZE_LIST_IDX}/run_2/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #         file3 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_list_idx_{c.OPT_BATCH_SIZE_LIST_IDX}/run_3/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #         plot_runs(metric, target_path, file1, file2, file3)

    # elif c.OPT_BATCH_MODE == 'fixed':

    #     # Creating runs plot
    #     for metric in metrics:
    #         target_path = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_fix_{c.OPT_BATCH_SIZE}"
    #         file1 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_fix_{c.OPT_BATCH_SIZE}/run_1/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #         file2 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_fix_{c.OPT_BATCH_SIZE}/run_2/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #         file3 = f"/home/salfers/ba/data/classification/{c.CLASS_TASK}/rl_alg/{c.RL_ALG}/config_{c.CONFIG_NR}/depth_{c.CIRCUIT_DEPTH_MAX}/batch_fix_{c.OPT_BATCH_SIZE}/run_3/data/data_log_{c.EPISODES*c.TIMESTEPS}.json"
    #         plot_runs(metric, target_path, file1, file2, file3)
