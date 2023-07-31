import json
import os

import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from src.molgen.data.dataset import SelfiesDataset
from src.molgen.models.ctd import ConditionalTransformerDecoder
from src.molgen.models.losses import LossFunctionCTD
from src.molgen.training.utils import pretrain_model_ctd
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

    # Load dataloader
    trainloader = DataLoader(
        train_dataset, batch_size=config.ctd.batch_size, shuffle=True, num_workers=config.training.data_loader_workers
    )
    valloader = DataLoader(
        val_dataset, batch_size=config.ctd.batch_size, shuffle=True, num_workers=config.training.data_loader_workers
    )

    # Number of conditions/attributes
    n_conditions = len(config.data.attribute_columns)

    # Defining the model
    model = ConditionalTransformerDecoder(
        max_selfie_len=max_selfie_length,
        n_alphabet_elements=len(index_to_symbol.keys()),
        n_decoder_blocks=config.ctd.n_decoder_blocks,
        d_model=config.ctd.d_model,
        d_ff=config.ctd.d_ff,
        n_mha_heads_decoder=config.ctd.n_mha_heads_decoder,
        dropout_p=config.ctd.dropout,
        normalizer_eps=config.ctd.normalizer_eps,
        include_bias=config.ctd.include_bias,
        n_attributes=n_conditions,
    )
    model.to(device)

    # Print model parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load pretrained model if a new is not wanted, else start from scratch
    restart_training = config.ctd.restart_training
    highest_epoch = 0
    if not restart_training:
        # Find all models os.path.join(root_dir, config.ctd.model_save_path, f"pretrained")
        models_paths = os.listdir(os.path.join(root_dir, config.ctd.model_save_path))
        # Find all models related to the current model
        models_paths = [
            path
            for path in models_paths
            if path.startswith(f"pretrained_enc{config.ctd.n_encoder_blocks}_dec{config.ctd.n_decoder_blocks}")
        ]
        # Find all epochs
        models_epochs = [int(path.split("_")[-1].split(".")[0]) for path in models_paths]
        # Find the highest epoch
        highest_epoch = max(models_epochs)
        # Load the model
        if len(models_epochs) > 0:
            pretrained_model_path = os.path.join(
                root_dir,
                config.ctd.model_save_path,
                f"pretrained_enc{config.ctd.n_encoder_blocks}_dec{config.ctd.n_decoder_blocks}_{highest_epoch}.pt",
            )
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            highest_epoch = -1
        highest_epoch += 1

    # Load weights if wanted
    if config.ctd.preload_weights:
        model.load_state_dict(
            torch.load(os.path.join(root_dir, "models/ctd", config.ctd.preload_weights_name)),
            strict=config.ctd.preload_weights_strict,
        )

    # Defining the optimizer
    optimizer = Adam(
        model.parameters(), lr=config.ctd.initial_lr, betas=[config.ctd.optimizer_beta1, config.ctd.optimizer_beta2]
    )

    # Defining the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.ctd.lr_scheduler_factor,
        patience=config.ctd.lr_scheduler_patience,
        verbose=True,
        min_lr=config.ctd.lr_schedule_min_lr,
    )

    # Defining the loss function
    loss_function = LossFunctionCTD(
        padding_int=symbol_to_index["[nop]"],
        n_attributes=n_conditions,
    )

    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(program="pretrain_ctd.py", program_relpath="pretrain_ctd.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    wandb_run = wandb.init(
        project=config.name,
        name="pretrain_ctd",
        group="pretrain_ctd",
        reinit=True,
        resume=False,
        entity=config.wandb.entity,
    )

    # Training the model
    model, val_loss, val_accuracy = pretrain_model_ctd(
        config=config,
        model=model,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        n_conditions=n_conditions,
        n_epochs=config.ctd.num_epochs,
        symbol_to_index=symbol_to_index,
        wandb_run=wandb_run,
        highest_epoch=highest_epoch,
        save_model=True,
        save_name=f"pretrained_dec{config.ctd.n_decoder_blocks}",
    )

    # Saving final model
    torch.save(
        model.state_dict(),
        os.path.join(root_dir, config.ctd.model_save_path, f"pretrained_dec{config.ctd.n_decoder_blocks}_final.pt"),
    )


if __name__ == "__main__":  # pragma: no cover
    main(config)
