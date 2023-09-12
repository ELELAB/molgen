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
from src.molgen.models.gct import GCT, AttributePredictor
from src.molgen.models.losses import AttributeLoss
from src.molgen.training.utils import train_attribute_predictor
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
    if config.gct.attribute_scaler == "minmax":
        scaler = MinMaxScaler()
    elif config.gct.attribute_scaler == "standard":
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
        train_dataset, batch_size=config.gct.batch_size, shuffle=True, num_workers=config.training.data_loader_workers
    )
    valloader = DataLoader(
        val_dataset, batch_size=config.gct.batch_size, shuffle=True, num_workers=config.training.data_loader_workers
    )

    # Number of conditions/attributes
    n_conditions = len(config.data.attribute_columns)

    # Defining the model
    model = GCT(
        max_selfie_len=max_selfie_length,
        n_alphabet_elements=len(index_to_symbol.keys()),
        n_encoder_blocks=config.gct.n_encoder_blocks,
        n_decoder_blocks=config.gct.n_decoder_blocks,
        d_model=config.gct.d_model,
        d_ff=config.gct.d_ff,
        d_latent_space=config.gct.d_latent_space,
        n_mha_heads_encoder=config.gct.n_mha_heads_encoder,
        n_mha_heads_decoder=config.gct.n_mha_heads_decoder,
        dropout_p=config.gct.dropout,
        normalizer_eps=config.gct.normalizer_eps,
        include_bias=config.gct.include_bias,
        include_conditions_encoder=False,
        include_conditions_decoder=False,
        include_conditions_reparameterization=False,
        n_attributes=n_conditions,
    )
    model.to(device)

    models_paths = os.listdir(os.path.join(root_dir, config.gct.model_save_path))
    # Find all models related to the current model
    models_paths = [
        path
        for path in models_paths
        if path.startswith(f"pretrained_enc{config.gct.n_encoder_blocks}_dec{config.gct.n_decoder_blocks}_nocondition")
    ]
    # Find all epochs
    models_epochs = [int(path.split("_")[-1].split(".")[0]) for path in models_paths]
    # Find the highest epoch
    highest_epoch = max(models_epochs)
    # Load the model
    if len(models_epochs) > 0:
        pretrained_model_path = os.path.join(
            root_dir,
            config.gct.model_save_path,
            f"pretrained_enc{config.gct.n_encoder_blocks}_dec{config.gct.n_decoder_blocks}_nocondition_{highest_epoch}.pt",
        )
        model.load_state_dict(torch.load(pretrained_model_path))
    else:
        raise AssertionError("No pretrained model found")

    # Print model parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define attribute prediction model. This is a gct encoder with a linear layer on top.
    attribute_predictor = AttributePredictor(
        max_selfie_len=max_selfie_length,
        n_alphabet_elements=len(index_to_symbol.keys()),
        n_encoder_blocks=config.gct.n_encoder_blocks,
        d_model=config.gct.d_model,
        d_ff=config.gct.d_ff,
        n_mha_heads_encoder=config.gct.n_mha_heads_encoder,
        dropout_p=config.gct.dropout,
        normalizer_eps=config.gct.normalizer_eps,
        include_bias=config.gct.include_bias,
        n_attributes=n_conditions,
    )
    attribute_predictor.to(device)

    # Defining the optimizer
    optimizer = Adam(
        attribute_predictor.parameters(),
        lr=config.gct.initial_lr,
        betas=[config.gct.optimizer_beta1, config.gct.optimizer_beta2],
    )

    # Defining the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.gct.lr_scheduler_factor,
        patience=config.gct.lr_scheduler_patience,
        verbose=True,
        min_lr=config.gct.lr_schedule_min_lr,
    )

    # Defining the loss function
    loss_function = AttributeLoss(
        attribute_columns=config.data.attribute_columns,
        attribute_weights=config.attribute_predictor.attribute_weights,
        device=device,
    )

    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(
        program="train_attribute_predictor.py", program_relpath="train_attribute_predictor.py"
    )
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    wandb_run = wandb.init(
        project=config.name,
        name="train_attribute_predictor",
        group="train_attribute_predictor",
        reinit=True,
        resume=False,
        entity=config.wandb.entity,
    )

    # Training the model
    (attribute_predictor, val_acc, val_loss) = train_attribute_predictor(
        config=config,
        gct=model,
        attribute_predictor=attribute_predictor,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        n_epochs=config.attribute_predictor.n_epochs,
        symbol_to_index=symbol_to_index,
        wandb_run=wandb_run,
        highest_epoch=0,
        save_model=True,
        save_name=f"attribute_predictor_enc{config.gct.n_encoder_blocks}",
    )
    # Saving final attribute predictor model
    torch.save(
        attribute_predictor.state_dict(),
        os.path.join(
            root_dir,
            config.gct.model_save_path,
            f"attribute_predictor_enc{config.gct.n_encoder_blocks}_final.pt",
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    main(config)
