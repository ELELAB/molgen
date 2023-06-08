import os

from datasets import load_dataset
from molgen.utils import get_root_directory
from omegaconf import DictConfig, OmegaConf

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """Summary line.

    Script to download the ZINC20 dataset, and save it under data/raw.

    Args:
        config (DictConfig): Configuration file located in config/config.yaml

    Returns:
        None
    """
    # Load the dataset
    root_dir = get_root_directory()
    zinc = load_dataset("sagawa/ZINC-canonicalized", cache_dir=root_dir + "\\data\\raw\\ZINC-canonicalized")

    # Save the dataset

    zinc.save_to_disk("data/raw/zinc_canonicalized")


if __name__ == "__main__":  # pragma: no cover
    main(config)
