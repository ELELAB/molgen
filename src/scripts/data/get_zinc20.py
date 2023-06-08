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
    zinc20 = load_dataset("zpn/zinc20", cache_dir=root_dir + "\\data\\raw\\zinc20")

    # Save the dataset

    zinc20.save_to_disk("data/raw/zinc20")


if __name__ == "__main__":  # pragma: no cover
    main(config)
