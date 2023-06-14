import multiprocessing as mp
import os

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.molgen.data.attribute_calculators import (
    JazzyAttributeCalculator,
    RdkitAttributeCalculator,
    SAScorerAttributeCalculator,
)
from src.molgen.data.dataset import DatasetFromFolder
from src.molgen.data.selfie import SmileSelfieConverter
from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def process_file(file_path: str) -> None:
    """
    This function is used for parallel processing of the files in the dataset.

    Args:
        file_path (str): Path to the file to be processed

    Returns:
        None, but saves the processed file in data/processed/zinc15
    """

    # Loading data
    data = pd.read_csv(file_path, delimiter="\t")

    # Defining the attribute calculators
    smile_selfie_converter = SmileSelfieConverter()
    rdkit_attribute_calculator = RdkitAttributeCalculator()
    jazzy_attribute_calculator = JazzyAttributeCalculator()
    sascorer_attribute_calculator = SAScorerAttributeCalculator()

    # Checking if all files should be repreprocessed
    restart_preprocessing = config.data.restart_preprocessing

    # Make new filepath and make directory if not present
    new_path = file_path.replace("raw", "processed")
    folder_path = os.path.dirname(new_path)
    os.makedirs(folder_path, exist_ok=True)

    # Checking if file already exists, if it does and restart_preprocessing is False, skip the file
    if os.path.isfile(new_path) and not restart_preprocessing:
        return

    # Add the selfie representation to the file
    data = smile_selfie_converter.add_selfie_to_file(
        data, smile_column="smiles", selfie_column="selfies"
    )  # Takes around 0.0005 seconds per molecule

    # Add the attributes to the file
    data = rdkit_attribute_calculator.add_all_to_file(
        data, smile_column="smiles"
    )  # Takes 0.004 seconds per molecule for qed, logp, tpsa, weight, but 0.25 for volume

    # Add the sascorer attributes to the file
    data = sascorer_attribute_calculator.add_sascorer_to_file(
        data, smile_column="smiles"
    )  # Takes around 0.0008 seconds per molecule

    # Add the jazzy attributes to the file
    data = jazzy_attribute_calculator.add_deltag_to_file(
        data, smile_column="smiles"
    )  # Takes around 0.15 seconds per molecule
    data = jazzy_attribute_calculator.add_molecular_vector_to_file(
        data, smile_column="smiles"
    )  # Takes around 0.1 seconds per molecule

    # Save the file
    data.to_csv(new_path, sep="\t", index=False)

    return


def main(config: DictConfig) -> None:
    """Summary line.

    Script to load the ZINC15 dataset from the ZINC15 dataset repository and add attributes for the smiles.

    Args:
        config (DictConfig): Configuration file located in config/config.yaml

    Returns:
        Saves the processed dataset in data/processed/zinc15
    """
    # Load the dataset
    root_dir = get_root_directory()
    zinc15_folder = os.path.join(root_dir, "data", "raw", "druglike-instock")

    # Create a dataset object
    dataset = DatasetFromFolder(zinc15_folder)
    all_files = dataset.file_paths

    # Parallel processing of the files
    n_cores = min(mp.cpu_count(), config.data.max_cores_preprocessing)
    pool = mp.Pool(processes=n_cores)

    # Process the files
    pool.map(process_file, all_files)

    # Close the pool
    pool.close()

    ### Below is for single core processing
    # # Converters and attribute calculators
    # smile_selfie_converter = SmileSelfieConverter()
    # rdkit_attribute_calculator = RdkitAttributeCalculator()
    # jazzy_attribute_calculator = JazzyAttributeCalculator()
    # sascorer_attribute_calculator = SAScorerAttributeCalculator()

    # # Checking if all files should be repreprocessed
    # restart_preprocessing = config.data.restart_preprocessing

    # # Iterate over the files and add attributes and selfie representation
    # for i in tqdm(range(len(dataset))):
    #     # Load the file and starting preprocessing
    #     data, file_path = dataset[i]

    #     # Make new filepath and make directory if not present
    #     new_path = file_path.replace("raw", "processed")
    #     folder_path = os.path.dirname(new_path)
    #     os.makedirs(folder_path, exist_ok=True)

    #     # Checking if file already exists, if it does and restart_preprocessing is False, skip the file
    #     if os.path.isfile(new_path) and not restart_preprocessing:
    #         continue

    #     # Add the selfie representation to the file
    #     data = smile_selfie_converter.add_selfie_to_file(
    #         data, smile_column="smiles", selfie_column="selfies"
    #     )  # Takes around 0.0005 seconds per molecule

    #     # Add the attributes to the file
    #     data = rdkit_attribute_calculator.add_all_to_file(
    #         data, smile_column="smiles"
    #     )  # Takes 0.004 seconds per molecule for qed, logp, tpsa, weight, but 0.25 for volume

    #     # Add the sascorer attributes to the file
    #     data = sascorer_attribute_calculator.add_sascorer_to_file(data, smile_column="smiles") # Takes around 0.0008 seconds per molecule

    #     # Add the jazzy attributes to the file
    #     data = jazzy_attribute_calculator.add_deltag_to_file(
    #         data, smile_column="smiles"
    #     )  # Takes around 0.15 seconds per molecule
    #     data = jazzy_attribute_calculator.add_molecular_vector_to_file(
    #         data, smile_column="smiles"
    #     )  # Takes around 0.1 seconds per molecule

    #     # Save the file
    #     data.to_csv(new_path, sep="\t", index=False)


if __name__ == "__main__":  # pragma: no cover
    main(config)
