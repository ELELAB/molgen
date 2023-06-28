import json
import os

import pandas as pd
import selfies as sf
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.molgen.utils import get_root_directory

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """
    This script looks through all files in the zinc15 dataset and makes a list of all unique characters in the dataset.
    This is saved as the alphabet.

    The script also finds the maximum length of a selfie in the dataset.
    """
    root_dir = get_root_directory()
    zinc15_folder = os.path.join(root_dir, config.data.processed_data_path)

    file_paths = []
    file_lengths = []
    for root, _, files in os.walk(zinc15_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                data = pd.read_csv(file_path, delimiter="\t", dtype=str)  # Adjust the delimiter if needed
                file_lengths.append(data.shape[0])

    alphabet = set()
    max_selfie_length = 0
    max_selfie = ""
    for file_path in tqdm(file_paths):
        file_data = pd.read_csv(file_path, delimiter="\t", dtype=str)
        selfies = file_data["selfies"].dropna().astype(str)
        file_alphabet = sf.get_alphabet_from_selfies(selfies)
        alphabet = alphabet.union(file_alphabet)
        file_max_selfie_lengths = [sf.len_selfies(s) for s in selfies]
        file_max_selfie_length = max(file_max_selfie_lengths)
        if file_max_selfie_length > max_selfie_length:
            max_selfie_length = file_max_selfie_length
            max_selfie = selfies[file_max_selfie_lengths.index(file_max_selfie_length)]
    alphabet.add("[nop]")
    alphabet = sorted(alphabet)

    symbol_to_index = {}
    index_to_symbol = {}
    for i, symbol in enumerate(alphabet):
        symbol_to_index[symbol] = i
        index_to_symbol[i] = symbol

    # Save the alphabet
    alphabet_path = os.path.join(root_dir, config.data.general_path)
    with open(alphabet_path + "alphabet.txt", "w") as f:
        for symbol in alphabet:
            f.write(symbol + "\n")

    # Save symbol_to_index
    symbol_to_index_path = os.path.join(root_dir, config.data.general_path, "symbol_to_index.json")
    index_to_symbol_path = os.path.join(root_dir, config.data.general_path, "index_to_symbol.json")

    with open(symbol_to_index_path, "w") as f:
        json.dump(symbol_to_index, f)

    with open(index_to_symbol_path, "w") as f:
        json.dump(index_to_symbol, f)

    # Save max_selfie_length
    max_selfie_length_path = os.path.join(root_dir, config.data.general_path, "max_selfie_length.txt")
    with open(max_selfie_length_path, "w") as f:
        f.write(str(max_selfie_length))

    print(f"Max selfie length: {max_selfie_length} \n Selfie: {max_selfie}")


if __name__ == "__main__":  # pragma: no cover
    main(config)
