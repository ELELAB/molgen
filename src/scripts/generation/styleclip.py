import json
import os

import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import wandb
from src.molgen.data.dataset import SelfiesDataset
from src.molgen.data.utils import selfie_from_tensor, smiles_from_selfies
from src.molgen.evaluation.metrics import (
    calculate_attribute_accuracy,
)
from src.molgen.models.gct import GCT
from src.molgen.models.losses import StyleclipLoss
from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:  # noqa: C901
    """
    This script uses a styleclip architecture to generate molecules, optimizing on the latent space, to make the molecules come closer to the wanted attributes.
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
        include_conditions_encoder=config.gct.include_conditions_encoder,
        include_conditions_decoder=config.gct.include_conditions_decoder,
        include_conditions_reparameterization=config.gct.include_conditions_reparameterization,
        n_attributes=n_conditions,
    )
    model.to(device)

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(root_dir, "models/gct", config.styleclip.model_path)), strict=False)

    # Defining the loss function
    loss_function = StyleclipLoss(
        target_attributes=config.styleclip.target_attributes,
        target_weighting=config.styleclip.target_weighting,
        attribute_columns=config.data.attribute_columns,
        scaler=scaler,
        device=device,
        attribute_min_values=train_dataset.data.loc[:, config.data.attribute_columns].min().values,
        attribute_max_values=train_dataset.data.loc[:, config.data.attribute_columns].max().values,
    )

    # Getting the initial latent space vector
    if config.styleclip.z_initialization_method == "sample":
        z = torch.randn(1, max_selfie_length, config.gct.d_latent_space).to(device)
    elif config.styleclip.z_initialization_method == "zero":
        z = torch.zeros(1, max_selfie_length, config.gct.d_latent_space).to(device)
    elif config.styleclip.z_initialization_method == "molecule":
        smile = config.styleclip.initial_molecule
        if smile in train_dataset.data["smiles"].values:
            smile_row = train_dataset.data.loc[train_dataset.data["smiles"] == smile]
        elif smile in val_dataset.data["smiles"].values:
            smile_row = val_dataset.data.loc[val_dataset.data["smiles"] == smile]
        selfie = smile_row["selfies"].iloc[0]
        selfie_labels = torch.Tensor(train_dataset.selfie_to_labels(selfie)).type(torch.int64)
        attributes = torch.Tensor(smile_row[config.data.attribute_columns].values).type(torch.float32).to(device)

        # Making the src, trg_input and trg_output for the transformer
        src = selfie_labels[1:-1].to(device)  # Removing the [nop] tokens from the start and end
        selfie_labels[:-1].to(device)  # Removing the [nop] token from the end
        selfie_labels[1:].to(device)  # Removing the [nop] token from the start

        # Getting z
        z, _, _ = model.encode(src.unsqueeze(0), attributes.unsqueeze(-1))
    else:
        raise ValueError(f"Unknown z initialization method: {config.styleclip.z_initialization_method}")

    # Converting z to make it trainable
    z = torch.nn.Parameter(z, requires_grad=True)

    # Initial
    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(program="styleclip.py", program_relpath="styleclip.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    wandb_name = "styleclip"
    for _i, attribute in enumerate(config.data.attribute_columns):
        wandb_name += f"_{attribute}{config.styleclip.target_attributes[attribute]}"

    # Initialize wandb run
    wandb_run = wandb.init(
        project=config.name,
        name=wandb_name,
        group="styleclip",
        reinit=True,
        resume=False,
        entity=config.wandb.entity,
    )

    # Defining the optimizer, optimizing on the latent space
    optimizer = Adam([z], lr=config.styleclip.lr)

    # Defining the scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.styleclip.lr_mul_factor,
        patience=config.styleclip.lr_patience,
        verbose=True,
        min_lr=config.gct.lr_schedule_min_lr,
    )

    # Defining target attributes as tensor
    target_attributes_scaled = loss_function.target_attributes_scaled.unsqueeze(0).to(device)

    # Making model eval mode as this is not a training loop for the model
    model.eval()

    # Optimizing the latent space to produce molecules with wanted attributes.
    for epoch in tqdm(range(config.styleclip.n_epochs)):
        # Generate molecules
        # generated_molecules, _, _ = model.generate(
        #                             conditions=target_attributes_scaled,
        #                             scaler=scaler,
        #                             max_selfie_length=max_selfie_length,
        #                             symbol_to_index=symbol_to_index,
        #                             batch_size=1,
        #                             config=config,
        #                             n_samples=1,
        #                             device=device,
        #                             method=config.styleclip.generation_method,
        #                             z=z,
        #                             scale_conditions=False,
        #                         )

        attributes_z, generated_molecules = model.predict_attr_from_z(
            conditions=target_attributes_scaled,
            scaler=scaler,
            max_selfie_length=max_selfie_length,
            symbol_to_index=symbol_to_index,
            batch_size=1,
            config=config,
            n_samples=1,
            device=device,
            method=config.styleclip.generation_method,
            z=z,
            scale_conditions=False,
        )

        # Calculate loss
        loss = loss_function.loss(attributes_z)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Reset gradients
        optimizer.zero_grad()

        # If the loss is not improving, reduce the learning rate
        scheduler.step(loss)

        # Log the loss
        wandb_run.log({"loss": loss.item()}, step=epoch)

        # Log the generated molecule and its attributes
        if epoch % config.styleclip.log_each_n_iters == 0:
            # Log metrics
            # Get selfies from generated molecules
            molecules_generated_selfies = selfie_from_tensor(
                generated_molecules, index_to_symbol, sos_token_included=True, eos_token="[nop]"  # noqa: S106
            )

            # Get smiles from generated molecules
            molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

            # Backtransform the target attributes
            if scaler is not None:
                target_attributes_scaled_inverse = scaler.inverse_transform(target_attributes_scaled.cpu().numpy())
                target_attributes_scaled_inverse = torch.Tensor(target_attributes_scaled_inverse)

            # Calculate metrics
            accuracy, _, real_generated_conditions = calculate_attribute_accuracy(
                molecules_generated_smiles, target_attributes_scaled_inverse, config=config, scaler=None
            )

            # Make image of molecule
            mol = Chem.MolFromSmiles(molecules_generated_smiles[0])
            molecule_image = Draw.MolToImage(mol)
            molecule_image = wandb.Image(molecule_image)

            # Attributes predictions backtransformed
            if scaler is not None:
                attributes_z_backscaled = scaler.inverse_transform(attributes_z.detach().cpu().numpy())
                attributes_z_backscaled = torch.Tensor(attributes_z_backscaled)

            # Log metrics and image
            log_dict = {}
            log_dict["smile"] = molecules_generated_smiles[0]
            log_dict["generated_molecule"] = molecule_image
            log_dict["accuracy"] = accuracy
            log_dict["loss"] = loss.item()
            log_dict["epoch"] = epoch
            for i, attribute in enumerate(scaler.get_feature_names_out()):
                if len(real_generated_conditions) == 0:
                    log_dict[attribute] = None
                else:
                    log_dict[attribute] = real_generated_conditions[attribute][0]
                log_dict[f"pred_{attribute}"] = attributes_z_backscaled[0, i].item()
                log_dict[f"target_{attribute}"] = target_attributes_scaled_inverse[0, i].item()
            wandb_run.log(log_dict, step=epoch)


if __name__ == "__main__":  # pragma: no cover
    main(config)
