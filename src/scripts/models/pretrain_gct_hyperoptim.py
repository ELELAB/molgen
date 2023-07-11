import joblib
import os
import optuna

import wandb
from omegaconf import DictConfig, OmegaConf
from src.molgen.utils import get_root_directory
from src.molgen.training.objectives import PretrainObjective

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """
    This script trains the GCT model with the predefined hyperparameters in the config file.
    """

    # Defining the wandb logger to track the training
    wandb_settings = wandb.Settings(program="pretrain_gct_hyperoptim.py", program_relpath="pretrain_gct_hyperoptim.py")
    wandb.setup(wandb_settings)
    wandb.login(key=config.wandb.WANDB_KEY, relogin=True)

    study_path = config.gct_hyperoptim.study_path
    restart_study = config.gct_hyperoptim.restart_study
    study_name = config.gct_hyperoptim.study_name
    n_study_runs = config.gct_hyperoptim.n_study_runs

    if os.path.exists(study_path) and not restart_study:
        study = joblib.load(study_path)
    else:
        study = optuna.create_study(study_name=study_name, direction="minimize")
    for _ in range(n_study_runs):
        objective = PretrainObjective(config, study_name)
        study.optimize(objective, n_trials=1)
        joblib.dump(study, study_path)

if __name__ == "__main__":  # pragma: no cover
    main(config)
