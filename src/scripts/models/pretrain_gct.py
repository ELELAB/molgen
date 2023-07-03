import copy
import json
import os

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.molgen.data.dataset import SelfiesDataset
from src.molgen.models.gct import GCT
from src.molgen.models.kl_annealer import KLAnnealer
from src.molgen.models.losses import LossFunction
from src.molgen.models.utils import make_nopeak_mask, make_padding_mask
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
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.is_train_set = False

    # Load dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.gct.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=config.gct.batch_size, shuffle=True)

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
        n_mha_heads=config.gct.n_mha_heads,
        dropout_p=config.gct.dropout,
        normalizer_eps=config.gct.normalizer_eps,
        include_bias=config.gct.include_bias,
        include_conditions_encoder=config.gct.include_conditions_encoder,
        include_conditions_decoder=config.gct.include_conditions_decoder,
        include_conditions_reparameterization=config.gct.include_conditions_reparameterization,
    )
    model.to(device)

    # Load pretrained model if a new is not wanted, else start from scratch
    restart_training = config.gct.restart_training
    highest_epoch = 0
    if not restart_training:
        # Find all models os.path.join(root_dir, config.gct.model_save_path, f"pretrained")
        models_paths = os.listdir(os.path.join(root_dir, config.gct.model_save_path))
        models_epochs = [int(path.split("_")[-1].split(".")[0]) for path in models_paths]
        # Find the highest epoch
        highest_epoch = max(models_epochs)
        # Load the model
        pretrained_model_path = os.path.join(root_dir, config.gct.model_save_path, f"pretrained_{highest_epoch}.pt")
        model.load_state_dict(torch.load(pretrained_model_path))
    highest_epoch += 1

    # Defining the optimizer
    optimizer = Adam(
        model.parameters(), lr=config.gct.initial_lr, betas=[config.gct.optimizer_beta1, config.gct.optimizer_beta2]
    )

    # Defining the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.gct.lr_scheduler_factor,
        patience=config.gct.lr_scheduler_patience,
        verbose=True,
    )

    # Defining the kl annealer
    klannealer = KLAnnealer(
        initial_beta=config.gct.kla_initial_beta,
        incriment_beta=config.gct.kla_increment_beta,
        beginning_epoch=config.gct.kla_beginning_epoch,
        max_beta=config.gct.kla_max_beta,
    )

    # Defining the orthoganl annealer
    orthannealer = KLAnnealer(
        initial_beta=config.gct.orth_initial_gamma,
        incriment_beta=config.gct.orth_increment_gamma,
        beginning_epoch=config.gct.orth_beginning_epoch,
        max_beta=config.gct.orth_max_gamma,
    )

    # Defining the loss function
    loss_function = LossFunction(symbol_to_index["[nop]"], len(config.data.attribute_columns))

    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(program="pretrain_gct.py", program_relpath="pretrain_gct.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    wandb_run = wandb.init(
        project=config.name,
        name="pretrain_gct",
        group="pretrain_gct",
        reinit=True,
        resume=False,
        entity=config.wandb.entity,
    )

    # Getting number of batches
    num_train_batches = len(trainloader)
    num_val_batches = len(valloader)
    steps = 0
    log_n_steps = config.wandb.log_n_steps

    # Training the model
    for epoch in tqdm(range(config.gct.num_epochs-highest_epoch)):
        # Calculating the beta and gamma for the epoch for weighting the loss function
        beta = klannealer.calculate_beta(epoch)
        gamma = orthannealer.calculate_beta(epoch)

        # Training the model
        model.train()
        train_loss = 0
        train_rce_loss = 0
        train_kl_divergence = 0
        train_orthogonal_loss = 0
        train_accuracy = 0
        for _idx, (src, trg_input, trg_output, attributes) in tqdm(
            enumerate(trainloader), leave=False, total=num_train_batches
        ):
            
            # Getting batch size
            batch_size = src.shape[0]

            # Move data to device
            src = src.to(device)
            trg_output = trg_output.to(device)
            trg_input = trg_input.to(device)
            attributes = attributes.to(device)

            # Clearing the optimizer gradient
            optimizer.zero_grad()

            # Defining the masks
            src_mask = None
            if config.gct.include_conditions_decoder:
                trg_no_peak_mask = make_nopeak_mask(
                    batch_size, device=device, dimension=trg_output.shape[1], n_conditions=n_conditions
                )
                trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=n_conditions)
            else:
                trg_no_peak_mask = make_nopeak_mask(trg_output, device=device)
                trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"])

            trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

            # Making forward pass
            selfie_p, _, _, mu, logvar = model(src, trg_input, attributes, src_mask, trg_mask)

            # Calculating the loss
            loss, rce_loss, kl_divergence, orthogonal_loss = loss_function.pretrain_loss(
                selfie_p, trg_output, beta, gamma, mu, logvar
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Adding the loss to the total loss
            accuracy = torch.argmax(selfie_p, axis=2) == trg_output
            accuracy = torch.mean(accuracy.type(torch.float))
            train_accuracy += accuracy.item() / num_train_batches
            train_loss += loss.item()
            train_rce_loss += rce_loss.item()
            train_kl_divergence += kl_divergence.item()
            train_orthogonal_loss += orthogonal_loss.item()
            steps += 1

            # Logging the loss
            if steps % log_n_steps == 0:
                wandb_run.log(
                    {
                        "train_loss": loss.item(),
                        "train_rce_loss": rce_loss.item(),
                        "train_kl_divergence": kl_divergence.item(),
                        "train_orthogonal_loss": orthogonal_loss.item(),
                        "train_accuracy": accuracy.item(),
                        "train_beta": beta,
                        "train_gamma": gamma,
                        "train_steps": steps,
                    }
                )

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "train_loss": train_loss / num_train_batches,
                "train_rce_loss": train_rce_loss / num_train_batches,
                "train_kl_divergence": train_kl_divergence / num_train_batches,
                "train_orthogonal_loss": train_orthogonal_loss / num_train_batches,
                "train_accuracy": train_accuracy,
                "train_beta": beta,
                "train_gamma": gamma,
                "train_epoch": epoch+highest_epoch,
            }
        )

        # Validating the model
        model.eval()
        val_accuracy = 0
        val_loss = 0
        val_rce_loss = 0
        val_kl_divergence = 0
        val_orthogonal_loss = 0
        with torch.no_grad():
            for _idx, (src, trg_input, trg_output, attributes) in tqdm(
                enumerate(valloader), leave=False, total=num_val_batches
            ):
                
                # Getting batch size
                batch_size = src.shape[0]

                # Move data to device
                src = src.to(device)
                trg_output = trg_output.to(device)
                trg_input = trg_input.to(device)
                attributes = attributes.to(device)

                # Defining the masks
                src_mask = None
                if config.gct.include_conditions_decoder:
                    trg_no_peak_mask = make_nopeak_mask(
                        batch_size, device=device, dimension=trg_output.shape[1], n_conditions=n_conditions
                    )
                    trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=n_conditions)
                else:
                    trg_no_peak_mask = make_nopeak_mask(trg_output, device=device)
                    trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"])

                trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

                # Making forward pass
                selfie_p, _, _, mu, logvar = model(src, trg_input, attributes, src_mask, trg_mask)

                # Calculating the loss
                loss, rce_loss, kl_divergence, orthogonal_loss = loss_function.pretrain_loss(
                    selfie_p, trg_output, beta, gamma, mu, logvar
                )

                # Adding the loss to the total loss
                accuracy = torch.argmax(selfie_p, axis=2) == trg_output
                accuracy = torch.mean(accuracy.type(torch.float))
                val_accuracy += accuracy.item() / num_val_batches
                val_loss += loss.item()
                val_rce_loss += rce_loss.item()
                val_kl_divergence += kl_divergence.item()
                val_orthogonal_loss += orthogonal_loss.item()

        # Updating the learning rate scheduler
        scheduler.step(val_loss)

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "val_loss": val_loss / num_val_batches,
                "val_rce_loss": val_rce_loss / num_val_batches,
                "val_kl_divergence": val_kl_divergence / num_val_batches,
                "val_orthogonal_loss": val_orthogonal_loss / num_val_batches,
                "val_accuracy": val_accuracy,
                "val_beta": beta,
                "val_gamma": gamma,
                "val_epoch": epoch+highest_epoch,
            }
        )

        # Saving the model
        if epoch % config.gct.save_model_epoch == 0:
            torch.save(model.state_dict(), os.path.join(root_dir, config.gct.model_save_path, f"pretrained_{epoch}.pt"))


if __name__ == "__main__":  # pragma: no cover
    main(config)
