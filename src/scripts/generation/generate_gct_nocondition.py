import json
import os

import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from src.molgen.data.dataset import SelfiesDataset
from src.molgen.data.utils import selfie_from_tensor, smiles_from_selfies
from src.molgen.evaluation.metrics import calculate_attribute_accuracy
from src.molgen.models.gct import GCT
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
    val_dataset = SelfiesDataset(
        data_dir, config.data.attribute_columns, index_to_symbol, symbol_to_index, max_selfie_length, is_train_set=False
    )
    train_dataset = SelfiesDataset(
        data_dir, config.data.attribute_columns, index_to_symbol, symbol_to_index, max_selfie_length, is_train_set=True
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

    # Print model parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Find and load the latest model
    # Find all models os.path.join(root_dir, config.ctd.model_save_path, f"pretrained")
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
    print(f"Highest epoch: {highest_epoch}, model_path: {models_paths[models_epochs.index(highest_epoch)]}")

    # Load the model
    pretrained_model_path = os.path.join(
        root_dir,
        config.gct.model_save_path,
        f"pretrained_enc{config.gct.n_encoder_blocks}_dec{config.gct.n_decoder_blocks}_nocondition_{highest_epoch}.pt",
    )

    # Load model
    model.load_state_dict(torch.load(pretrained_model_path), strict=True)

    # Generate molecules
    molecules_generated = []
    final_all_conditions = []
    n_molecules_generated = 0
    n_samples = config.evaluation.eval_n_molecules_generated

    # Choosing random conditions from the validation set, to make sure that the compination of conditions make sense together
    all_conditions = torch.from_numpy(val_dataset.data[config.data.attribute_columns].sample(n=n_samples).values)

    # Generate molecules
    for i in tqdm(range(n_samples), desc="Generating molecules"):
        # Expand conditions to batch size
        conditions = all_conditions[i].expand(config.evaluation.eval_n_molecules_pr_condition, -1)

        # Add noise to conditions
        conditions = conditions + torch.randn_like(conditions) * config.evaluation.eval_condition_noise * torch.Tensor(
            val_dataset.data[config.data.attribute_columns].max()
        )

        # Generate molecules
        generated_molecules, _, _ = model.generate(
            conditions=conditions,
            scaler=scaler,
            max_selfie_length=max_selfie_length,
            symbol_to_index=symbol_to_index,
            batch_size=config.gct.batch_size,
            config=config,
            n_samples=config.evaluation.eval_n_molecules_pr_condition,
            device=device,
            method=config.evaluation.eval_generation_method,
            z=None,
            scale_conditions=False,
        )

        # Append molecules to list
        molecules_generated.append(generated_molecules)
        final_all_conditions.append(conditions)

        # Update number of molecules generated
        n_molecules_generated += generated_molecules.shape[0]

    # Concatenate molecules
    molecules_generated = torch.cat(molecules_generated, dim=0).type(torch.int64)
    final_all_conditions = torch.cat(final_all_conditions, dim=0).type(torch.int64)

    # Get selfie molecules
    molecules_generated_selfies = selfie_from_tensor(
        molecules_generated, index_to_symbol, sos_token_included=True, eos_token="[nop]"  # noqa: S106
    )

    # Get smiles from selfies
    molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

    # Calculate metrics for the generated molecules
    attribute_accuracy, validity, generated_smiles_df = calculate_attribute_accuracy(
        molecules_generated_smiles, all_conditions, config=config, scaler=scaler
    )

    # Make dictionary with smiles and attributes
    generated_smiles_df["selfies"] = molecules_generated_selfies

    # Save dataframe to csv
    generated_smiles_df.to_csv(
        os.path.join(root_dir, config.gct.generated_smiles_path + "_nocondition.txt"), index=False, sep="\t"
    )


if __name__ == "__main__":  # pragma: no cover
    main(config)
