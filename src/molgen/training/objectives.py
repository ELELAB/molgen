import json
import os

import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.molgen.data.dataset import SelfiesDataset
from src.molgen.models.gct import GCT
from src.molgen.models.kl_annealer import KLAnnealer
from src.molgen.models.losses import LossFunction
from src.molgen.models.utils import make_nopeak_mask, make_padding_mask
from src.molgen.utils import get_root_directory
from src.molgen.training.utils import pretrain_model
from src.molgen.evaluation.metrics import calculate_validity, calculate_uniqueness, calculate_similarity, calculate_attribute_accuracy, calculate_novelty
from src.molgen.data.utils import selfie_from_tensor, smiles_from_selfies

class PretrainObjective(object):
    """
    Object of the optuna study in hyperparameters.
    This trains the model with different hyperparameters to determine the optimal parameters.
    """
    def __init__(self, config, study_name=None):
        self.config = config
        self.study_name = study_name
    
    def __call__(self, trial):
        """
        DESCRIPTION:
            This is one call for the optuna study which chooses the next hyperparameters through
            bayesian optimization, and computes the loss for the model with the given hyperparameters. 
        INPUT: 
            trial: trial from the optuna study.
        OUTPUT:
        """

        #Creating a dictionary of the hyperparameters.
        c = dict(
            #Hyperparameters for the model
            n_encoder_blocks = trial.suggest_int("n_encoder_blocks", 1, 12),
            n_decoder_blocks = trial.suggest_int("n_decoder_blocks", 1, 12),
            d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            d_latent_space = trial.suggest_categorical("d_latent_space", [32, 64, 128, 256]),
            feed_forward_dim = trial.suggest_categorical("feed_forward_dim", [256, 512, 1024, 2048]),
            n_mha_heads_encoder = trial.suggest_categorical("n_mha_heads_encoder", [1, 4, 8, 16, 32]),
            n_mha_heads_decoder = trial.suggest_categorical("n_mha_heads_decoder", [1, 4, 8, 16, 32]),
            dropout_p = trial.suggest_float("dropout_p", 0.0, 0.5),
            include_bias = trial.suggest_categorical("include_bias", [True, False]),
            include_conditions_encoder = trial.suggest_categorical("include_conditions_encoder", [True, False]),
            include_conditions_decoder = trial.suggest_categorical("include_conditions_decoder", [True, False]),
            include_conditions_reparameterization = trial.suggest_categorical("include_conditions_reparameterization", [True, False]),
            #Hyperparameters for the training
            initial_lr = trial.suggest_float("initial_lr", 1e-5, 1e-1, log=True),
            lr_schedule_factor = trial.suggest_float("lr_schedule_factor", 0.1, 1.0),
            n_epochs = trial.suggest_int("n_epochs", 1, 50),
            #Hyperparameters for the loss function
            kla_initial_beta = trial.suggest_float("kla_initial_beta", 0.02, 0.2),
            kla_increase_beta = trial.suggest_float("kla_increase_beta", 0.01, 0.1),
            orth_initial_gamma = trial.suggest_float("orth_initial_gamma", 0.0002, 0.002),
            orth_increase_gamma = trial.suggest_float("orth_increase_gamma", 0.0001, 0.001),
            #Data hyperparameters
            attribute_scaler = trial.suggest_categorical("attribute_scaler", ["minmax", "standard", "none"]),
            )
        
        # Define device, use cuda if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setting seed
        seed = self.config.general.seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        
        # Load the dataset
        root_dir = get_root_directory()
        data_dir = os.path.join(root_dir, self.config.data.processed_data_path)

        # Load selfie alphabet
        symbol_to_index_path = os.path.join(root_dir, self.config.data.general_path, "symbol_to_index.json")
        index_to_symbol_path = os.path.join(root_dir, self.config.data.general_path, "index_to_symbol.json")

        with open(symbol_to_index_path) as f:
            symbol_to_index = json.load(f)

        with open(index_to_symbol_path) as f:
            index_to_symbol = json.load(f)

        # Load max selfie length
        max_selfie_length_path = os.path.join(root_dir, self.config.data.general_path, "max_selfie_length.txt")

        with open(max_selfie_length_path) as f:
            max_selfie_length = int(f.read())

        # Load dataset
        train_dataset = SelfiesDataset(
            data_dir, self.config.data.attribute_columns, index_to_symbol, symbol_to_index, max_selfie_length, is_train_set=True
        )
        val_dataset = SelfiesDataset(
            data_dir, self.config.data.attribute_columns, index_to_symbol, symbol_to_index, max_selfie_length, is_train_set=False
        )

        # Defining data scaler
        if c["attribute_scaler"] == "minmax":
            scaler = MinMaxScaler()
        elif c["attribute_scaler"] == "standard":
            scaler = StandardScaler()
        else:
            scaler = None
        
        # Fit scaler
        if scaler is not None:
            scaler.fit(train_dataset.data[self.config.data.attribute_columns])

            # Scale data
            train_dataset.data.loc[:,self.config.data.attribute_columns] = scaler.transform(train_dataset.data[self.config.data.attribute_columns])
            val_dataset.data.loc[:,self.config.data.attribute_columns] = scaler.transform(val_dataset.data[self.config.data.attribute_columns])

        # Load dataloader
        trainloader = DataLoader(train_dataset, batch_size=self.config.gct.batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=self.config.gct.batch_size, shuffle=True)

        # Number of conditions/attributes
        n_conditions = len(self.config.data.attribute_columns)

        # Defining the model
        model = GCT(
            max_selfie_len=max_selfie_length,
            n_alphabet_elements=len(index_to_symbol.keys()),
            n_encoder_blocks=c["n_encoder_blocks"],
            n_decoder_blocks=c["n_decoder_blocks"],
            d_model=c["d_model"],
            d_ff=c["feed_forward_dim"],
            d_latent_space=c["d_latent_space"],
            n_mha_heads_encoder=c["n_mha_heads_encoder"],
            n_mha_heads_decoder=c["n_mha_heads_decoder"],
            dropout_p=c["dropout_p"],
            normalizer_eps=self.config.gct.normalizer_eps,
            include_bias=c["include_bias"],
            include_conditions_encoder=c["include_conditions_encoder"],
            include_conditions_decoder=c["include_conditions_decoder"],
            include_conditions_reparameterization=c["include_conditions_reparameterization"],
        )
        model.to(device)

        # Defining the optimizer
        optimizer = Adam(model.parameters(), lr=c["initial_lr"], betas=[self.config.gct.optimizer_beta1, self.config.gct.optimizer_beta2])

        # Defining the learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=c["lr_schedule_factor"],
            patience=self.config.gct.lr_scheduler_patience,
            verbose=True,
            min_lr=self.config.gct.lr_schedule_min_lr,
        )

        # Defining the KL annealer
        klannealer = KLAnnealer(
            initial_beta=c["kla_initial_beta"],
            incriment_beta=c["kla_increase_beta"],
            beginning_epoch=self.config.gct.kla_beginning_epoch,
            max_beta=self.config.gct.kla_max_beta,
        )

        # Defining the orthogonality annealer
        orthannealer = KLAnnealer(
            initial_beta=c["orth_initial_gamma"],
            incriment_beta=c["orth_increase_gamma"],
            beginning_epoch=self.config.gct.orth_beginning_epoch,
            max_beta=self.config.gct.orth_max_gamma,
        )

        # Defining the loss function
        loss_function = LossFunction(
            padding_int=symbol_to_index["[nop]"],
            n_attributes=n_conditions,
        )

        # Defining the wandb config
        wandb_run = wandb.init(
            project=self.config.name,
            name=f"gct_{trial.number}",
            entity=self.config.wandb.entity,
            group=self.study_name,
            config=c,
            reinit=True,
        )

        # Training the model
        model, val_loss, val_rce_loss, val_kl_divergence, val_orthogonal_loss, val_accuracy = pretrain_model(
                                                                                                    config=self.config,
                                                                                                    model=model,
                                                                                                    device=device,
                                                                                                    trainloader=trainloader,
                                                                                                    valloader=valloader,
                                                                                                    klannealer=klannealer,
                                                                                                    orthannealer=orthannealer,
                                                                                                    optimizer=optimizer,
                                                                                                    scheduler=scheduler,
                                                                                                    loss_function=loss_function,
                                                                                                    n_conditions=n_conditions,
                                                                                                    n_epochs=c["n_epochs"],
                                                                                                    symbol_to_index=symbol_to_index,
                                                                                                    wandb_run=wandb_run,
                                                                                                    highest_epoch=0,
                                                                                                    save_model=False,
                                                                                                )

        # Saving the model
        if self.config.gct_hyperoptim.save_models:
            torch.save(model.state_dict(), os.path.join(root_dir, self.config.gct.model_save_path, f"gct_trial{trial.number}.pt"))

        # Generating molecules
        all_conditions = []
        molecules_generated = []
        n_molecules_generated = 0
        while n_molecules_generated < self.config.gct_hyperoptim.eval_n_molecules_generated:
            # Choose random conditions from validation set
            conditions = torch.from_numpy(val_dataset.data[self.config.data.attribute_columns].sample(n=1).values)
            
            # Expand conditions to batch size
            conditions = conditions.expand(self.config.gct_hyperoptim.eval_n_molecules_pr_condition, -1)

            # Add noise to conditions
            conditions = conditions + torch.randn_like(conditions) * self.config.gct_hyperoptim.eval_condition_noise * torch.Tensor(val_dataset.data[self.config.data.attribute_columns].max())

            # Adding conditions to list
            all_conditions.append(conditions)

            # Generate molecules
            generated_molecules, _, _ = model.generate(
                                        conditions=conditions,
                                        scaler=scaler,
                                        max_selfie_length=max_selfie_length,
                                        symbol_to_index=symbol_to_index,
                                        batch_size=self.config.gct.batch_size,
                                        config=self.config,
                                        n_samples=self.config.gct_hyperoptim.eval_n_molecules_pr_condition,
                                        device=device,
                                        method=self.config.gct_hyperoptim.eval_generation_method,
                                        z=None,
                                        scale_conditions=False,
                                    )

            # Append molecules to list
            molecules_generated.append(generated_molecules)

            # Update number of molecules generated
            n_molecules_generated += generated_molecules.shape[0]

        # Concatenate molecules
        molecules_generated = torch.cat(molecules_generated, dim=0).type(torch.int64)
        all_conditions = torch.cat(all_conditions, dim=0).type(torch.float)

        # Get selfie molecules
        molecules_generated_selfies = selfie_from_tensor(molecules_generated, index_to_symbol, sos_token_included=True, eos_token="[nop]")

        # Get smiles from selfies
        molecules_generated_smiles = smiles_from_selfies(molecules_generated_selfies)

        # Evaluating the model
        validity = calculate_validity(molecules_generated_smiles)
        novelty = calculate_novelty(molecules_generated_selfies, train_dataset.data["selfies"])
        uniqueness = calculate_uniqueness(molecules_generated_smiles)
        # similarity = calculate_similarity(molecules_generated_smiles, train_dataset.data["smiles"]) #This takes way to long to compute
        attribute_accuracy = calculate_attribute_accuracy(molecules_generated_smiles, all_conditions, config=self.config, scaler=scaler)

        # Calculating the final loss
        final_loss = self.config.gct_hyperoptim.eval_loss_validation_rce_weight * val_rce_loss + \
                     self.config.gct_hyperoptim.eval_loss_validation_kl_weight * val_kl_divergence + \
                     self.config.gct_hyperoptim.eval_loss_validation_orth_weight * val_orthogonal_loss + \
                     self.config.gct_hyperoptim.eval_loss_validity_weight * (1 - validity) + \
                     self.config.gct_hyperoptim.eval_loss_novelty_weight * (1 - novelty) + \
                     self.config.gct_hyperoptim.eval_loss_uniqueness_weight * (1 - uniqueness) + \
                     self.config.gct_hyperoptim.eval_loss_attribute_accuracy_weight * attribute_accuracy
                    #  self.config.gct_hyperoptim.eval_loss_similarity_weight * (1 - similarity) + \

        # Logging the results
        wandb_run.log({
            "validity": validity,
            "uniqueness": uniqueness,
            # "similarity": similarity,
            "novelty": novelty,
            "attribute_accuracy": attribute_accuracy,
            "val_loss": val_loss,
            "val_rce_loss": val_rce_loss,
            "val_kl_divergence": val_kl_divergence,
            "val_orthogonal_loss": val_orthogonal_loss,
            "val_accuracy": val_accuracy,
            "final_loss":final_loss})
        
        # Finishing the wandb run
        wandb_run.finish()

        # Returning the final loss
        return final_loss
        

    