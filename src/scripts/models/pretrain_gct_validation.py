import json
import os

import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import wandb
from src.molgen.data.dataset import SelfiesDataset
from src.molgen.data.utils import selfie_from_tensor, smiles_from_selfies
from src.molgen.evaluation.metrics import (
    calculate_attribute_accuracy,
    calculate_novelty,
    calculate_uniqueness,
)
from src.molgen.models.gct import GCT
from src.molgen.models.losses import LossFunction
from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """
    This script validates the GCT models for each saved model.
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

    # Print model parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Find pretrained model epochs if there are pretrained models.
    # Find all models os.path.join(root_dir, config.gct.model_save_path, f"pretrained")
    models_paths = os.listdir(os.path.join(root_dir, config.gct.model_save_path))
    # Find all models related to the current model
    models_paths = [
        path
        for path in models_paths
        if path.startswith(f"pretrained_enc{config.gct.n_encoder_blocks}_dec{config.gct.n_decoder_blocks}")
    ]
    # Find all epochs
    models_epochs = [int(path.split("_")[-1].split(".")[0]) for path in models_paths]

    # Defining the loss function
    LossFunction(
        padding_int=symbol_to_index["[nop]"],
        n_attributes=n_conditions,
    )

    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(program="pretrain_gct.py", program_relpath="pretrain_gct.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    # Looping over the models and validating them
    for model_path, _model_epoch in zip(models_paths, models_epochs):
        # Get full model path
        full_model_path = os.path.join(root_dir, config.gct.model_save_path, model_path)

        # Load model
        model.load_state_dict(torch.load(full_model_path), strict=False)

        # Initialize wandb run
        wandb.init(
            project=config.name,
            name=f"validation_{model_path}",
            group="pretrain_gct_validation",
            reinit=True,
            resume=False,
            entity=config.wandb.entity,
        )

        # Generate molecules for validation
        all_conditions = []
        molecules_generated = []
        n_molecules_generated = 0
        while n_molecules_generated < config.evaluation.eval_n_molecules_generated:
            # Choose random conditions from validation set
            conditions = torch.from_numpy(val_dataset.data[config.data.attribute_columns].sample(n=1).values)

            # Expand conditions to batch size
            conditions = conditions.expand(config.evaluation.eval_n_molecules_pr_condition, -1)

            # Add noise to conditions
            conditions = conditions + torch.randn_like(
                conditions
            ) * config.evaluation.eval_condition_noise * torch.Tensor(
                val_dataset.data[config.data.attribute_columns].max()
            )

            # Adding conditions to list
            all_conditions.append(conditions)

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

            # Update number of molecules generated
            n_molecules_generated += generated_molecules.shape[0]

        # Concatenate molecules
        molecules_generated = torch.cat(molecules_generated, dim=0).type(torch.int64)
        all_conditions = torch.cat(all_conditions, dim=0).type(torch.float32)

        # Get selfie molecules
        molecules_generated_selfies = selfie_from_tensor(
            molecules_generated, index_to_symbol, sos_token_included=True, eos_token="[nop]"  # noqa: S106
        )

        # Get smiles from selfies
        molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

        # Evaluating the model
        novelty = calculate_novelty(molecules_generated_selfies, train_dataset.data["selfies"])
        uniqueness = calculate_uniqueness(molecules_generated_smiles)
        # similarity = calculate_similarity(molecules_generated_smiles, train_dataset.data["smiles"]) #This takes way to long to compute
        attribute_accuracy, validity, generated_smiles_conditions = calculate_attribute_accuracy(
            molecules_generated_smiles, all_conditions, config=config, scaler=scaler
        )

        # Log metrics
        wandb.log(
            {
                "novelty": novelty,
                "uniqueness": uniqueness,
                "attribute_accuracy": attribute_accuracy,
                "validity": validity,
            }
        )


if __name__ == "__main__":  # pragma: no cover
    main(config)
