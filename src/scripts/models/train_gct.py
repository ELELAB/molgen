import copy
import json
import os

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.molgen.data.dataset import SelfiesDataset
from src.molgen.models.gct import GCT
from src.molgen.models.kl_annealer import KLAnnealer
from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """
    This script trains the GCT model with the predefined hyperparameters in the config file.
    """
    # Define device, use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting seed
    seed = config.general.seed
    torch.manual_seed(seed)
    if device == "cuda":
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
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.is_train_set = False

    # Load dataloader
    DataLoader(train_dataset, batch_size=config.gct.batch_size, shuffle=True)
    DataLoader(val_dataset, batch_size=config.gct.batch_size, shuffle=True)

    # Defining the model
    model = GCT(
        max_selfie_len=max_selfie_length,
        n_attributes=len(config.data.attribute_columns),
        n_alphabet_elements=len(index_to_symbol.keys()),
        n_encoder_blocks=config.gct.n_encoder_blocks,
        n_decoder_blocks=config.gct.n_decoder_blocks,
        d_model=config.gct.d_model,
        d_ff=config.gct.d_ff,
        d_latent_space=config.gct.d_latent_space,
        n_mha_heads=config.gct.n_mha_heads,
        dropout=config.gct.dropout,
        normalizer_eps=config.gct.normalizer_eps,
        include_bias=config.gct.include_bias,
    )

    # Defining the optimizer
    optimizer = Adam(
        model.parameters(), lr=config.gct.initial_lr, betas=[config.gct.optimizer_beta_1, config.gct.optimizer_beta_2]
    )

    # Defining the learning rate scheduler
    ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.gct.lr_scheduler_factor,
        patience=config.gct.lr_scheduler_patience,
        verbose=True,
    )

    # for epoch in range(num_epochs):
    #     train()  # Perform training
    #     val_loss = validate()  # Calculate validation loss

    #     scheduler.step(val_loss)

    # Defining the kl annealer
    KLAnnealer(
        initial_beta=config.gct.kla_initial_beta,
        incriment_beta=config.gct.kla_increment_beta,
        beginning_epoch=config.gct.kla_beginning_epoch,
        max_beta=config.gct.kla_max_beta,
    )


if __name__ == "__main__":  # pragma: no cover
    main(config)
