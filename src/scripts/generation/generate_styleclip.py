import json
import os

import pandas as pd
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
from src.molgen.models.gct import GCT, AttributePredictor
from src.molgen.models.losses import AttributeLoss
from src.molgen.models.utils import make_nopeak_mask, make_padding_mask
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
        include_conditions_encoder=False,
        include_conditions_decoder=False,
        include_conditions_reparameterization=False,
        n_attributes=n_conditions,
    )
    model.to(device)

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(root_dir, "models/gct", config.styleclip.model_path)), strict=True)

    # Defining the loss function
    loss_function = AttributeLoss(
        attribute_columns=config.data.attribute_columns,
        attribute_weights=config.attribute_predictor.attribute_weights,
        device=device,
    )

    # Defining attribute predictor
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

    # Load attribute predictor weights
    attribute_predictor.load_state_dict(
        torch.load(os.path.join(root_dir, "models/gct", config.styleclip.attribute_predictor_path)),
        strict=True,
    )

    # Initial
    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(program="styleclip.py", program_relpath="styleclip.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    # Generate molecules
    molecules_generated = []
    final_all_conditions = []
    n_molecules_generated = 0
    n_samples = config.evaluation.eval_n_molecules_generated

    # Choosing random conditions from the validation set, to make sure that the compination of conditions make sense together
    all_conditions = torch.from_numpy(val_dataset.data[config.data.attribute_columns].sample(n=n_samples).values)

    model.eval()
    attribute_predictor.eval()

    # Initialize Dataframe
    # smiles	qed	logp	tpsa	weight	sascore	sdc	sdx	sa	dga	dgp	dgtot	mds	logp_target	weight_target	qed_target	tpsa_target	sascore_target	sdx_target	sa_target	dga_target	dgp_target	selfies
    generated_smiles_df = pd.DataFrame(
        columns=[
            "smiles",
            "qed",
            "logp",
            "tpsa",
            "weight",
            "sascore",
            "sdc",
            "sdx",
            "sa",
            "dga",
            "dgp",
            "dgtot",
            "mds",
            "logp_target",
            "weight_target",
            "qed_target",
            "tpsa_target",
            "sascore_target",
            "sdx_target",
            "sa_target",
            "dga_target",
            "dgp_target",
            "selfies",
        ]
    )

    # Generate molecules
    for i in tqdm(range(n_samples), desc="Generating molecules"):
        # Expand conditions to batch size
        conditions = all_conditions[i].expand(config.evaluation.eval_n_molecules_pr_condition, -1)

        # Add noise to conditions
        conditions = conditions + torch.randn_like(conditions) * config.evaluation.eval_condition_noise * torch.Tensor(
            val_dataset.data[config.data.attribute_columns].max()
        )

        conditions = conditions.to(device)

        # Making sure conditions are on the right type
        conditions = conditions.float()

        if i % 10 == 0:
            wandb_name = "styleclip"
            for _j, attribute in enumerate(config.data.attribute_columns):
                wandb_name += f"_{attribute}{conditions[0][_j]}"

            wandb_run = wandb.init(
                project=config.name,
                name=wandb_name,
                group="styleclip",
                reinit=True,
                resume=False,
                entity=config.wandb.entity,
            )

        if config.styleclip.z_initialization_method == "sample":
            z = torch.randn(1, max_selfie_length, config.gct.d_latent_space).to(device)
        else:
            z = torch.zeros(1, max_selfie_length, config.gct.d_latent_space).to(device)

        z = torch.nn.Parameter(z, requires_grad=True)

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

        # Optimizing the latent space to produce molecules with wanted attributes.
        for epoch in range(config.styleclip.n_epochs):
            # Generate molecule
            generated_molecules, _, _ = model.generate(
                conditions=conditions,
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

            # Make sure generated molecules are integers
            generated_molecules = generated_molecules.long()

            src_mask = None
            trg_no_peak_mask = make_nopeak_mask(
                1, device=device, dimension=generated_molecules.shape[1], n_conditions=0
            )
            trg_padding_mask = make_padding_mask(generated_molecules, symbol_to_index["[nop]"], n_conditions=0)

            trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

            decoder_out = model.decode(
                generated_molecules,
                conditions.unsqueeze(-1),
                z,
                src_mask=src_mask,
                trg_mask=trg_mask,
            )

            # Add one more padding molecule to decoder_out as this will be removed in the attribute predictor because of the input in training.
            decoder_out = torch.cat((decoder_out, torch.zeros(1, 1, decoder_out.shape[-1]).to(device)), dim=1)

            attributes_pred = attribute_predictor(decoder_out)

            # Calculate loss
            loss = loss_function.loss(attributes_pred, conditions)

            # Backpropagate
            loss.backward()

            # Update parameters
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            # If the loss is not improving, reduce the learning rate
            scheduler.step(loss)

            # Log the loss
            if i % 10 == 0:
                wandb_run.log({"loss": loss.item()}, step=epoch)

            # Log the generated molecule and its attributes
            if epoch % config.styleclip.log_each_n_iters == 0 and i % 10 == 0:
                # Log metrics
                # Get selfies from generated molecules
                molecules_generated_selfies = selfie_from_tensor(
                    generated_molecules, index_to_symbol, sos_token_included=True, eos_token="[nop]"  # noqa: S106
                )

                # Get smiles from generated molecules
                molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

                # Backtransform the target attributes
                if scaler is not None:
                    target_attributes_scaled_inverse = scaler.inverse_transform(conditions.cpu().numpy())
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
                    attributes_z_backscaled = scaler.inverse_transform(attributes_pred.detach().cpu().numpy())
                    attributes_z_backscaled = torch.Tensor(attributes_z_backscaled)

                # Log metrics and image
                log_dict = {}
                log_dict["smile"] = molecules_generated_smiles[0]
                log_dict["generated_molecule"] = molecule_image
                log_dict["accuracy"] = accuracy
                log_dict["loss"] = loss.item()
                log_dict["epoch"] = epoch
                for j, attribute in enumerate(scaler.get_feature_names_out()):
                    if len(real_generated_conditions) == 0:
                        log_dict[attribute] = None
                    else:
                        log_dict[attribute] = real_generated_conditions[attribute][0]
                    log_dict[f"pred_{attribute}"] = attributes_z_backscaled[0, j].item()
                    log_dict[f"target_{attribute}"] = target_attributes_scaled_inverse[0, j].item()
                wandb_run.log(log_dict, step=epoch)

        # Generate final molecule
        generated_molecules, _, _ = model.generate(
            conditions=conditions,
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

        if i % 10 == 0:
            # Log metrics
            # Get selfies from generated molecules
            molecules_generated_selfies = selfie_from_tensor(
                generated_molecules, index_to_symbol, sos_token_included=True, eos_token="[nop]"  # noqa: S106
            )

            # Get smiles from generated molecules
            molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

            # Backtransform the target attributes
            if scaler is not None:
                target_attributes_scaled_inverse = scaler.inverse_transform(conditions.cpu().numpy())
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
                attributes_z_backscaled = scaler.inverse_transform(attributes_pred.detach().cpu().numpy())
                attributes_z_backscaled = torch.Tensor(attributes_z_backscaled)

            # Log metrics and image
            log_dict = {}
            log_dict["smile"] = molecules_generated_smiles[0]
            log_dict["generated_molecule"] = molecule_image
            log_dict["accuracy"] = accuracy
            log_dict["loss"] = loss.item()
            log_dict["epoch"] = epoch
            for j, attribute in enumerate(scaler.get_feature_names_out()):
                if len(real_generated_conditions) == 0:
                    log_dict[attribute] = None
                else:
                    log_dict[attribute] = real_generated_conditions[attribute][0]
                log_dict[f"pred_{attribute}"] = attributes_z_backscaled[0, j].item()
                log_dict[f"target_{attribute}"] = target_attributes_scaled_inverse[0, j].item()
            wandb_run.log(log_dict, step=epoch)

        # Append molecules to list
        molecules_generated.append(generated_molecules)
        final_all_conditions.append(conditions)

        # Update number of molecules generated
        n_molecules_generated += generated_molecules.shape[0]

        # Save molecules
        if i % 10 == 0 or i == n_samples - 1:
            # Concatenate molecules and append to dataframe
            molecules_generated_tensor = torch.cat(molecules_generated[-10:], dim=0).type(torch.int64)
            final_all_conditions_tensor = torch.cat(final_all_conditions[-10:], dim=0).type(torch.float32)
            # Get selfie molecules
            molecules_generated_selfies = selfie_from_tensor(
                molecules_generated_tensor, index_to_symbol, sos_token_included=True, eos_token="[nop]"  # noqa: S106
            )

            # Get smiles from selfies
            molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

            # Calculate metrics for the generated molecules
            attribute_accuracy, validity, generated_smiles_df_new = calculate_attribute_accuracy(
                molecules_generated_smiles, final_all_conditions_tensor, config=config, scaler=scaler
            )
            # Make dictionary with smiles and attributes
            if generated_smiles_df_new.shape[0] != 0:
                generated_smiles_df_new["selfies"] = molecules_generated_selfies

                # Append to dataframe
                generated_smiles_df = pd.merge(generated_smiles_df, generated_smiles_df_new, how="outer")

            # Save dataframe to csv
            generated_smiles_df.to_csv(
                os.path.join(root_dir, config.styleclip.generated_smiles_path + ".txt"), index=False, sep="\t"
            )


if __name__ == "__main__":  # pragma: no cover
    main(config)
