import json
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.molgen.data.dataset import SelfiesDataset
from src.molgen.evaluation.metrics import (
    calculate_intdiv_snn_frag,
    calculate_novelty,
    calculate_uniqueness,
)
from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """
    This script trains the CTD model with the predefined hyperparameters in the config file.
    """
    # Define device, use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting the device
    torch.cuda.set_device(config.training.cuda_device)

    # Setting seed
    seed = config.general.seed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Load the dataset
    root_dir = get_root_directory()
    data_dir = os.path.join(root_dir, config.data.processed_data_path)

    # Load selfie alphabet
    symbol_to_index_path = os.path.join(root_dir, config.data.general_path, "symbol_to_index.json")
    index_to_symbol_path = os.path.join(root_dir, config.data.general_path, "index_to_symbol.json")

    with open(symbol_to_index_path) as f:
        symbol_to_index = json.load(f)

    with open(index_to_symbol_path) as f:
        index_to_symbol = json.load(f)

    # Load max selfie length
    max_selfie_length_path = os.path.join(root_dir, config.data.general_path, "max_selfie_length.txt")

    with open(max_selfie_length_path) as f:
        max_selfie_length = int(f.read())

    # Load dataset
    train_dataset = SelfiesDataset(
        data_dir, config.data.attribute_columns, index_to_symbol, symbol_to_index, max_selfie_length, is_train_set=True
    )
    val_dataset = SelfiesDataset(
        data_dir, config.data.attribute_columns, index_to_symbol, symbol_to_index, max_selfie_length, is_train_set=False
    )

    # Defining data scaler
    if config.ctd.attribute_scaler == "minmax":
        scaler = MinMaxScaler()
    elif config.ctd.attribute_scaler == "standard":
        scaler = StandardScaler()
    else:
        scaler = None

    # Fit scaler
    if scaler is not None:
        scaler.fit(train_dataset.data[config.data.attribute_columns])

        # Scale data
        train_dataset.data.loc[:, config.data.attribute_columns] = scaler.transform(
            train_dataset.data[config.data.attribute_columns]
        )
        val_dataset.data.loc[:, config.data.attribute_columns] = scaler.transform(
            val_dataset.data[config.data.attribute_columns]
        )

    # Number of conditions/attributes
    len(config.data.attribute_columns)

    # Load generated data
    generated_smiles_df_ctd = pd.read_csv(os.path.join(root_dir, config.ctd.generated_smiles_path + ".txt"), sep="\t")
    generated_smiles_df_gct = pd.read_csv(os.path.join(root_dir, config.gct.generated_smiles_path + ".txt"), sep="\t")
    generated_smiles_df_gct_nocondition = pd.read_csv(
        os.path.join(root_dir, "data/generated/gct_generated_nocondition" + ".txt"), sep="\t"
    )
    generated_smiles_df_styleclip = pd.read_csv(
        os.path.join(root_dir, config.styleclip.generated_smiles_path + ".txt"), sep="\t"
    )

    generated_smiles_dfs = [
        generated_smiles_df_ctd,
        generated_smiles_df_gct,
        generated_smiles_df_gct_nocondition,
        generated_smiles_df_styleclip,
    ]
    generated_smiles_dfs_names = ["ctd", "gct", "gct_nocondition", "styleclip"]
    # generated_smiles_dfs = [generated_smiles_df_gct]
    # generated_smiles_dfs_names = ["gct"]
    # all_metrics = {} # Store all metrics in a dictionary
    # for name in generated_smiles_dfs_names:
    #     all_metrics[name] = {}

    # Load Metrics
    with open(os.path.join(root_dir, "figures", "metrics.json")) as f:
        all_metrics = json.load(f)

    # # Calculate attribute accuracy
    # for attr in config.data.attribute_columns:
    #     plt.figure()
    #     for i, generated_smiles_df in enumerate(generated_smiles_dfs):
    #         attribute_difference = generated_smiles_df[attr].values - generated_smiles_df[attr + "_target"].values
    #         attribute_difference = attribute_difference[~np.isnan(attribute_difference)]
    #         attribute_difference_mean = np.mean(np.abs(attribute_difference))
    #         difference_count, difference_bins = np.histogram(attribute_difference, bins="auto", density=True)
    #         difference_bins = [(difference_bins[i + 1] + difference_bins[i]) / 2 for i in range(len(difference_bins) - 1)]
    #         plt.plot(difference_bins, difference_count, label=generated_smiles_dfs_names[i], color=generated_smiles_colors[i])
    #         #Fill from 0 to plot
    #         plt.fill_between(difference_bins, difference_count, color=generated_smiles_colors[i], alpha=0.2)
    #         all_metrics[generated_smiles_dfs_names[i]][f"{attr}_diff_mean"] = attribute_difference_mean
    #         all_metrics[generated_smiles_dfs_names[i]][f"{attr}_diff_std"] = np.std(np.abs(attribute_difference))
    #     plt.title("Histogram for difference - " + attr)
    #     plt.legend()
    #     plt.savefig(os.path.join(root_dir, "figures", "difference_hist", attr + "_all.png"))
    #     plt.close()

    # # Histogram of attribute values
    # for attr in config.data.attribute_columns:
    #     plt.figure()
    #     for i, generated_smiles_df in enumerate(generated_smiles_dfs):
    #         attribute_values = generated_smiles_df[attr].values
    #         attribute_values = attribute_values[~np.isnan(attribute_values)]
    #         attribute_count, attribute_bins = np.histogram(attribute_values, bins="auto", density=True)
    #         attribute_bins = [(attribute_bins[i + 1] + attribute_bins[i]) / 2 for i in range(len(attribute_bins) - 1)]
    #         plt.plot(attribute_bins, attribute_count, label=generated_smiles_dfs_names[i], color=generated_smiles_colors[i])
    #         #Fill from 0 to plot
    #         plt.fill_between(attribute_bins, attribute_count, color=generated_smiles_colors[i], alpha=0.2)
    #         all_metrics[generated_smiles_dfs_names[i]][f"{attr}_mean"] = np.mean(attribute_values)
    #         all_metrics[generated_smiles_dfs_names[i]][f"{attr}_std"] = np.std(attribute_values)
    #     plt.title("Histogram - " + attr)
    #     plt.legend()
    #     plt.savefig(os.path.join(root_dir, "figures", "attribute_hist", attr + "_all.png"))
    #     plt.close()

    # # Calculate Validity (Where all attributes where able to be calculated)
    # for i, generated_smiles_df in enumerate(generated_smiles_dfs):
    #     attributes = generated_smiles_df[config.data.attribute_columns].values
    #     non_nan_idx = np.sum(np.isnan(attributes), axis=1) == 0
    #     validity = sum(non_nan_idx) / len(non_nan_idx)
    #     all_metrics[generated_smiles_dfs_names[i]]["validity"] = validity
    #     print("Validity for " + generated_smiles_dfs_names[i] + ": " + str(validity))

    # Calculate Novelty
    for i, generated_smiles_df in enumerate(generated_smiles_dfs):
        novelty = calculate_novelty(generated_smiles_df["smiles"].values, train_dataset.data["smiles"].values)
        all_metrics[generated_smiles_dfs_names[i]]["novelty"] = novelty
        print("Novelty for " + generated_smiles_dfs_names[i] + ": " + str(novelty))

    # Save metrics
    with open(os.path.join(root_dir, "figures", "metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    # Calculate Uniqueness @1K, @10K
    for i, generated_smiles_df in enumerate(generated_smiles_dfs):
        uniqueness_1k = []
        for _ in range(config.evaluation.eval_confidence_interval_n):
            # Sample 1K
            generated_smiles_df_sample = generated_smiles_df["smiles"].sample(n=1000)
            # Calculate uniqueness
            unique = calculate_uniqueness(generated_smiles_df_sample)
            uniqueness_1k.append(unique)
        all_metrics[generated_smiles_dfs_names[i]]["uniqueness_1k_mu"] = np.mean(uniqueness_1k)
        all_metrics[generated_smiles_dfs_names[i]]["uniqueness_1k_ci"] = [
            np.percentile(uniqueness_1k, 2.5),
            np.percentile(uniqueness_1k, 97.5),
        ]
        uniqueness_10k = []
        for _ in range(config.evaluation.eval_confidence_interval_n):
            # Sample 10K
            generated_smiles_df_sample = generated_smiles_df["smiles"].sample(n=10000)
            # Calculate uniqueness
            unique = calculate_uniqueness(generated_smiles_df_sample)
            uniqueness_10k.append(unique)
        all_metrics[generated_smiles_dfs_names[i]]["uniqueness_10k_mu"] = np.mean(uniqueness_10k)
        all_metrics[generated_smiles_dfs_names[i]]["uniqueness_10k_ci"] = [
            np.percentile(uniqueness_10k, 2.5),
            np.percentile(uniqueness_10k, 97.5),
        ]

    # Calculate Fragment Similarity, SNN and Internal Diversity p=1,2
    # Calculate Internal Diversity p=2
    for i, generated_smiles_df in enumerate(generated_smiles_dfs):
        print("Calculating metrics for " + generated_smiles_dfs_names[i])
        fragment_similarity, snn_similarity, [int_p1_similarity, int_p2_similarity] = calculate_intdiv_snn_frag(
            generated_smiles_df["smiles"].values, train_dataset.data["smiles"].values, [1, 2]
        )
        all_metrics[generated_smiles_dfs_names[i]]["fragment_similarity"] = fragment_similarity
        all_metrics[generated_smiles_dfs_names[i]]["snn_similarity"] = snn_similarity
        all_metrics[generated_smiles_dfs_names[i]]["internal_diversity_p1"] = int_p1_similarity
        all_metrics[generated_smiles_dfs_names[i]]["internal_diversity_p2"] = int_p2_similarity
        with open(os.path.join(root_dir, "figures", "metrics.json"), "w") as f:
            json.dump(all_metrics, f)

    # Save metrics
    with open(os.path.join(root_dir, "figures", "metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    # # Load Metrics
    # with open(os.path.join(root_dir, "figures", "metrics.json"), "r") as f:
    #     all_metrics = json.load(f)


if __name__ == "__main__":  # pragma: no cover
    main(config)
