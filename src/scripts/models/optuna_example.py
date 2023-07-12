import joblib
import torch
import torch.optim as optim
import logging
import os
import pickle
import pandas as pd
from omegaconf import OmegaConf
from src.utils import change_workdir_to_standard
import wandb
import optuna
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from src.data.dataloaders import CallDataset
from src.models.modelutils import PositionalEncoder
from src.models.functionutils import attention, make_nopeak_mask, make_padding_mask
from src.models.transformer.utils import Transformer
from src.models.schedulers import CosineWithRestarts, WarmUpDefault
from src.models.annealing import KLAnnealer
from src.models.losses import normal_distribution_loss, log_normal_distribution_loss, mse_loss
from src.models.transformer.utils import test_model
import time
from tqdm import tqdm
import random

class Objective(object):
    """
    Object of the optuna study in hyperparameters.
    This trains the model with different hyperparameters to determine the optimal parameters.
    """
    def __init__(self, train_dataset, validation_dataset, device, config, study_name=None, use_fake_data=False, fake_dataset=None, only_use_encoder=False):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.fake_dataset = fake_dataset
        self.device = device
        self.only_use_encoder = only_use_encoder
        if study_name is None:
            self.study_name = config.transformer.study_name
        else:
            self.study_name = study_name
        self.project_name = config.transformer.project_name
        self.config = config
        self.padding_token = config.preprocessing.padding_token
        self.sos_token = config.preprocessing.sos_token
        self.entity = config.wandb.entity
        self.use_fake_data = use_fake_data
    
    def __call__(self, trial):
        """
        DESCRIPTION:
            This is one call for the optuna study which chooses the next hyperparameters through
            bayesian optimization, and computes the loss for the model with the given hyperparameters. 
        INPUT: 
            trial: trial from the optuna study.
        OUTPUT:
            -total_validation_smile_rce_loss: It returns the negative mean_validation_smile_rce_loss. The optuna model maximizes as this looks better in wandb. 
                                              Therefor it is the negative mean_validation_smile_rce_loss that is returned.
        """

        #Creating a dictionary of the hyperparameters.
        c = dict(
            #Transformer hyperparameters
            train_loss = trial.suggest_categorical("train_loss", ["mse_loss", "normal_distribution_loss"]),
            n_encoder_blocks = trial.suggest_int("n_encoder_blocks", 1, 20),
            n_decoder_blocks = trial.suggest_int("n_decoder_blocks", 1, 12),
            feed_forward_dim = trial.suggest_categorical("feed_forward_dim", [128, 256, 512, 1024, 2048, 4096]),
            d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256, 512, 1024]),
            mha_heads_encoder = trial.suggest_categorical("mha_heads_encoder", [1, 2, 4, 8, 16, 32]),
            mha_heads_decoder = trial.suggest_categorical("mha_heads_decoder", [1, 2, 4, 8, 16, 32]),
            dropout_p = trial.suggest_float("dropout_p", 0.05, 0.5),
            # initial_lr = trial.suggest_float("initial_lr", 0.000001, 0.001),
            # include_bias = trial.suggest_categorical("include_bias", [True, False]),
            # training_epochs = trial.suggest_categorical("training_epochs", [10, 20, 35, 50, 75, 100, 150, 200, 300]),
            # optimizer_warmup_steps = trial.suggest_categorical("optimizer_warmup_steps", [i for i in range(1000,21000,1000)])
            optimizer_warmup_steps = trial.suggest_categorical("optimizer_warmup_steps", [self.config.transformer.param_warmup_steps]),
            initial_lr = trial.suggest_categorical("initial_lr", [self.config.transformer.param_initial_lr]),
            include_bias = trial.suggest_categorical("include_bias", [self.config.transformer.param_include_bias]),
            training_epochs = trial.suggest_categorical("training_epochs", [self.config.transformer.param_training_epochs]),
            )
        
        if self.only_use_encoder:
            c["n_decoder_blocks"] = 1
            c["mha_heads_decoder"] = 1

        n_intervals = int((config.preprocessing.current_opening_hours[1] - config.preprocessing.current_opening_hours[0])*60/config.preprocessing.grouping_interval_length)
        n_attributes = self.train_dataset.additional_metrics.shape[1]
        #Defining positional encoder
        positional_encoder = PositionalEncoder(c["d_model"], max_seq_len=n_intervals+1) #+1 because of start of sequence token makes the interval 1 longer
        positional_encoder.to(self.device)

        #Defining attention_function
        attention_function = attention

        #Defining the gct model with the given hyperparameters
        model = Transformer(positional_encoder=positional_encoder,
                            attention_function=attention_function,
                            n_encoder_blocks=c["n_encoder_blocks"],
                            n_decoder_blocks=c["n_decoder_blocks"],
                            d_model=c["d_model"],
                            d_ff=c["feed_forward_dim"],
                            mha_heads_encoder=c["mha_heads_encoder"],
                            mha_heads_decoder=c["mha_heads_decoder"],
                            n_attributes=n_attributes,
                            n_intervals=n_intervals,
                            only_use_encoder=self.only_use_encoder,
                            dropout_p=c["dropout_p"],
                            include_bias=c["include_bias"]
                            )

        model.to(self.device)

        #Defining the optimizer
        optimizer = optim.Adam(model.parameters(), 
                                lr=c["initial_lr"], 
                                betas=(config.transformer.optimizer_beta1, config.transformer.optimizer_beta2))

        #Defining the loss function
        if c["train_loss"] == "normal_distribution_loss":
            loss_function = normal_distribution_loss
        elif c["train_loss"] == "log_normal_distribution_loss":
            loss_function = log_normal_distribution_loss
        elif c["train_loss"] == "mse_loss":
            loss_function = mse_loss

        #Defining the Schedular
        if config.transformer.paper_lr_schedule == "SGDR":
            schedular = CosineWithRestarts(optimizer, len(train_loader))
        elif config.transformer.paper_lr_schedule == "WarmUp":
            schedular = WarmUpDefault(optimizer, c["optimizer_warmup_steps"], c["d_model"])
        else:
            schedular = None

        #Initializing wandb run, to track and visualize runs.
        run = wandb.init(
            project = self.project_name,
            name=f"trial_{trial.number}",
            group=self.study_name,
            config=c,
            reinit=True,
            entity=self.entity,
        )

        log.info(f"***STARTING NEW TRIAL - no {trial.number + 1}***")
        log.info(f"Hyperparams dictionary: {c}")

        train_loader = DataLoader(self.train_dataset, batch_size=config.transformer.batch_size, shuffle=config.transformer.shuffle_dataloader)
        validation_loader = DataLoader(self.validation_dataset, batch_size=config.transformer.batch_size, shuffle=config.transformer.shuffle_dataloader)
        if self.use_fake_data:
            fake_loader = DataLoader(self.fake_dataset, batch_size=config.transformer.batch_size, shuffle=config.transformer.shuffle_dataloader)
        else:
            fake_loader = None

        validation_loss, validation_accuracy, train_loss, train_accuracy, current_lr = self.train_model(trial=trial,
                                                                                        model=model,
                                                                                        optimizer=optimizer, 
                                                                                        schedular=schedular,
                                                                                        trainloader=train_loader,
                                                                                        validationloader=validation_loader,
                                                                                        fakeloader=fake_loader,
                                                                                        wandb_run=run,
                                                                                        device=self.device,
                                                                                        padding_int=config.preprocessing.padding_token,
                                                                                        c=c,
                                                                                        loss_fun=loss_function)

        
        validation_generation_loss, validation_generation_accuracy, max_validation_loss = test_model(model=model,
                                                                                                    test_loader=validation_loader,
                                                                                                    device=self.device,
                                                                                                    loss_fun=loss_function,
                                                                                                    config=config)

        _, train_generation_accuracy, max_train_loss = test_model(model=model,
                                        test_loader=train_loader,
                                        device=self.device,
                                        loss_fun=loss_function,
                                        config=config)

        optimizer_loss = self.config.transformer.study_optimize_train_max_loss_weight * max_train_loss
        optimizer_loss += self.config.transformer.study_optimize_validation_max_loss_weight * max_validation_loss
        optimizer_loss += self.config.transformer.study_optimize_train_accuracy_weight * train_generation_accuracy
        optimizer_loss += self.config.transformer.study_optimize_validation_accuracy_weight * validation_generation_accuracy

        run.log({"validation_loss": validation_loss,
                 "validation_accuracy": validation_accuracy,
                 "validation_generation_loss": validation_generation_loss,
                 "validation_generation_accuracy": validation_generation_accuracy,
                 "train_loss": train_loss,
                 "train_accuracy": train_accuracy,
                 "max_validation_loss": max_validation_loss,
                 "max_train_loss": max_train_loss,
                 "weighted_loss": optimizer_loss,
                 "lr": current_lr,
                 })

        run.finish()


        return optimizer_loss
    
def train_model(self, trial, model, optimizer, schedular, trainloader, validationloader, fakeloader, wandb_run, device, padding_int, c, loss_fun):
        """
        This is the training script for each run of the optuna model.
        """
        log.info("Starting training of model.")

        start_time = time.time()

        current_step = 0
        for epoch in tqdm(range(c["training_epochs"]), total=c["training_epochs"]):
            mean_train_loss = 0
            mean_train_accuracy = 0
            train_size = 0
            for idx, (decoder_input, decoder_target, metrics) in enumerate(trainloader):
                current_step += 1
                model.train()
                optimizer.zero_grad()
                batch_size = decoder_input.shape[0]
                decoder_input, decoder_target, metrics = decoder_input.to(device), decoder_target.to(device), metrics.to(device)
                
                #Conditions are only concatenated with the input in decoder
                src_mask = None #This means no mask is used for source.
                trg_mask = make_nopeak_mask(batch_size, decoder_input.shape[-1]).to(device)

                output_mu, output_std = model(decoder_input, metrics, src_mask, trg_mask)

                output_mu = output_mu.view(batch_size, -1)
                output_std = output_std.view(batch_size, -1)
                decoder_target = decoder_target.view(batch_size, -1)
                
                loss = loss_fun(output_mu, output_std, decoder_target)

                loss.backward()
                optimizer.step()

                mean_train_accuracy += torch.sum(torch.abs(decoder_target-output_mu))
                train_size += (batch_size * (decoder_input.shape[-1]-1))
                mean_train_loss += loss.item()

                if schedular:
                    schedular.step(current_step)
                
                current_lr = optimizer.param_groups[0]["lr"]
            
            mean_train_loss = mean_train_loss / len(trainloader)
            mean_train_accuracy = mean_train_accuracy / train_size

            mean_fake_loss = 0
            mean_fake_accuracy = 0
            if self.use_fake_data:
                fake_size = 0
                for idx, (decoder_input, decoder_target, metrics) in enumerate(fakeloader):
                    model.train()
                    optimizer.zero_grad()
                    batch_size = decoder_input.shape[0]
                    decoder_input, decoder_target, metrics = decoder_input.to(device), decoder_target.to(device), metrics.to(device)

                    #Conditions are only concatenated with the input in decoder
                    src_mask = None #This means no mask is used for source.
                    trg_mask = make_nopeak_mask(batch_size, decoder_input.shape[-1]).to(device)

                    output_mu, output_std = model(decoder_input, metrics, src_mask, trg_mask)

                    output_mu = output_mu.view(batch_size, -1)
                    output_std = output_std.view(batch_size, -1)
                    decoder_target = decoder_target.view(batch_size, -1)

                    loss = loss_fun(output_mu, output_std, decoder_target)

                    loss.backward()
                    optimizer.step()

                    mean_fake_accuracy += torch.sum(torch.abs(decoder_target-output_mu))
                    fake_size += (batch_size * (decoder_input.shape[-1]-1))
                    mean_fake_loss += loss.item()
                mean_fake_loss = mean_fake_loss / len(fakeloader)
                mean_fake_accuracy = mean_fake_accuracy / fake_size

            with torch.no_grad():
                model.eval()
                mean_validation_loss = 0
                mean_validation_accuracy = 0
                validation_size = 0
                for idx, (decoder_input, decoder_target, metrics) in enumerate(validationloader):
                    batch_size = decoder_input.shape[0]
                    decoder_input, decoder_target, metrics = decoder_input.to(device), decoder_target.to(device), metrics.to(device)
                    
                    #Conditions are only concatenated with the input in decoder
                    src_mask = None #This means no mask is used for source.
                    trg_mask = make_nopeak_mask(batch_size, decoder_input.shape[-1]).to(device)

                    output_mu, output_std = model(decoder_input, metrics, src_mask, trg_mask)

                    output_mu = output_mu.view(batch_size, -1)
                    output_std = output_std.view(batch_size, -1)
                    decoder_target = decoder_target.view(batch_size, -1)
                    
                    loss = loss_fun(output_mu, output_std, decoder_target)

                    mean_validation_accuracy += torch.sum(torch.abs(decoder_target-output_mu))
                    validation_size += batch_size * (decoder_input.shape[-1]-1)
                    mean_validation_loss += loss.item()

                mean_validation_loss = mean_validation_loss / len(validationloader)
                mean_validation_accuracy = mean_validation_accuracy / validation_size

            log.info(f"Epoch: {epoch+1}/{c['training_epochs']}, Train loss: {mean_train_loss:.4f}, Train accuracy: {mean_train_accuracy:.4f}, Validation loss: {mean_validation_loss:.4f}, Validation accuracy: {mean_validation_accuracy:.4f}, Fake loss: {mean_fake_loss:.4f}, Fake accuracy: {mean_fake_accuracy:.4f}, LR: {current_lr:.6f}")
        
        trial.set_user_attr("final_lr", current_lr)
        trial.set_user_attr("training_time", time.time() - start_time)

        return mean_validation_loss, mean_validation_accuracy, mean_train_loss, mean_train_accuracy, current_lr

change_workdir_to_standard()
config = OmegaConf.load("configs/config.yaml")

#Setting up logging if file it is not run from main.py. 
log = logging.getLogger(_name_)
if len(log.handlers) == 0:
    log.setLevel(config.logging.level)
    formater = logging.Formatter(fmt='%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s')
    filename = _file_.split("src")[1].strip(".py")
    folder = "\\".join(f"logging{filename}.log".split("\\")[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    file_handler = logging.FileHandler(f"logging{filename}.log")
    file_handler.setLevel(config.logging.level)
    file_handler.setFormatter(formater)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)

def main(config):
    """
    This script finds the optimal hyperparameters for the embedding model. 
    The optimal model is then used when training the final embedding model in "train_best_model.py"
    """
    log.info(f"Starting {os.path.basename(_file_)}")

    log.info("Loading data and initializing model.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Setting seed
    seed = config.transformer.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #Training settings
    use_scaled_data = config.transformer.training_use_scaled_data
    use_uncorrelated_data = config.transformer.training_use_uncorrelated_data
    use_fake_data = config.transformer.training_use_fake_data
    fake_dataset = config.transformer.training_fake_dataset
    only_use_encoder = config.transformer.only_use_encoder

    train_path = "data/processed/final_data"
    validation_path = "data/processed/final_data"
    fake_train_path = "data/processed/fake_data"
    study_path = config.transformer.study_path.split(".pkl")[0]
    study_name = config.transformer.study_name
    
    if use_uncorrelated_data:
        train_path += "_uncorrelated"
        validation_path += "_uncorrelated"
        fake_train_path += "_uncorrelated"
        study_path += "_uncorrelated"
        study_name += "_uncorrelated"
    
    if use_scaled_data:
        train_path += "_scaled"
        validation_path += "_scaled"
        fake_train_path += "_scaled"
        study_path += "_scaled"
        study_name += "_scaled"

    if fake_dataset == "dimittend":
        fake_train_path += "_dimittend"

    train_path += "_train.csv"
    validation_path += "_val.csv"
    fake_train_path += "_train.csv"

    if only_use_encoder:
        study_path += "_encoder"
        study_name += "_encoder"

    train = pd.read_csv(train_path, delimiter=";", header=0)
    validation = pd.read_csv(validation_path, delimiter=";", header=0)

    train = train.loc[train["Closed"] == 0]
    validation = validation.loc[validation["Closed"] == 0]
    interval_columns = [i for i in train.columns if "min" in i.lower()]
    train = train[train[interval_columns[0]].astype(str) != "nan"]
    validation = validation[validation[interval_columns[0]].astype(str) != "nan"]

    #Loading data
    if use_fake_data:
        study_path += "_fake"
        study_name += "_fake"
        if fake_dataset == "dimittend":
            study_path += "_dimittend"
            study_name += "_dimittend"
        fake_train = pd.read_csv(fake_train_path, delimiter=";", header=0)
        fake_train = fake_train.loc[fake_train["Closed"] == 0]
        fake_train = fake_train[fake_train[interval_columns[0]].astype(str) != "nan"]
    else:
        fake_train = None

    #order columns to be same order as in training
    columns = train.columns
    validation = validation[columns]
    if use_fake_data:
        fake_train = fake_train[columns]

    study_path += ".pkl"

    #Loading data
    train_dataset = CallDataset(train, sos_token=config.preprocessing.sos_token, pad_token=config.preprocessing.padding_token, cheat_columns=config.transformer.cheat_columns)
    validation_dataset = CallDataset(validation, sos_token=config.preprocessing.sos_token, pad_token=config.preprocessing.padding_token, cheat_columns=config.transformer.cheat_columns)
    if use_fake_data:
        fake_train_dataset = CallDataset(fake_train, sos_token=config.preprocessing.sos_token, pad_token=config.preprocessing.padding_token, cheat_columns=config.transformer.cheat_columns)
    else:
        fake_train_dataset = None

    wandb_settings = wandb.Settings(program="optuna_transformer.py", program_relpath="optuna_transformer.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    log.info(f"Starting Study.")
    if os.path.exists(study_path) and not config.transformer.restart_optuna:
        study = joblib.load(study_path)
    else:
        study = optuna.create_study(study_name=study_name, direction="minimize")
    for _ in range(config.transformer.n_study_runs):
        study.optimize(Objective(train_dataset, validation_dataset, device, config, study_name, use_fake_data, fake_train_dataset, only_use_encoder), n_trials=1)
        joblib.dump(study, study_path)

if _name_ == "_main_":
    main(config)