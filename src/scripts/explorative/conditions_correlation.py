import json
import os

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from src.molgen.data.dataset import SelfiesDataset
from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """
    This script makes a correlation matrix for the attributes in the dataset, and explores this with plots.
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

    # Getting the attribute names
    attribute_names = config.data.attribute_columns

    # Getting the attributes¨
    attributes = train_dataset.data[attribute_names]

    # Making the correlation matrix
    correlation_matrix = attributes.corr()

    # Plotting the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(correlation_matrix, cmap="seismic")

    # Add correlation values to the squares
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="w")

    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation Value")
    plt.savefig(os.path.join(root_dir, "figures", "correlation_matrix.png"), dpi=300)


if __name__ == "__main__":  # pragma: no cover
    main(config)
