import os

import pandas as pd
import selfies as sf
import torch
from torch.utils.data import Dataset


class DatasetFromFolder(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.file_paths = self._get_file_paths()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        data = pd.read_csv(file_path, delimiter="\t")  # Adjust the delimiter if needed

        if self.transform:
            data = self.transform(data)

        return data, file_path

    def _get_file_paths(self):
        file_paths = []
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths


class SelfiesDataset(Dataset):
    def __init__(
        self,
        data_folder,
        attribute_columns,
        index_to_symbol,
        symbol_to_index,
        max_selfie_len,
        is_train_set=True,
        train_ratio=0.9,
        transform=None,
    ):
        self.data_folder = data_folder
        self.transform = transform
        self.attribute_columns = attribute_columns
        self.is_train_set = is_train_set
        self.symbol_to_idx = symbol_to_index
        self.idx_to_symbol = index_to_symbol
        self.pad_to_len = max_selfie_len
        self.train_ratio = train_ratio
        self.file_paths, self.file_lengths = self._get_file_paths()
        self.train_idx, self.val_idx = self._divide_data_idx()
        # self.alphabet = self._make_alphabet()
        # self.max_selfie_length = self._get_max_selfie_length()
        # self.symbol_to_index, self.index_to_symbol = self._make_symbol_to_index()

    def __len__(self):
        if self.is_train_set:
            return len(self.train_idx)
        else:
            return len(self.val_idx)

    def __getitem__(self, idx):
        # Converting idx of train and test set to the real idx in the files
        idx = self.train_idx[idx] if self.is_train_set else self.val_idx[idx]

        # Finding the file idx
        file_idx = 0
        while idx >= self.file_lengths[file_idx]:
            idx -= self.file_lengths[file_idx]
            file_idx += 1
        file_path = self.file_paths[file_idx]

        # Loading the data row
        row = pd.read_csv(
            file_path, delimiter="\t", skiprows=lambda x: x not in [0, idx]
        )  # Adjust the delimiter if needed
        selfie = row["selfies"][0]

        # Making the one hot encoding of the selfie
        selfie_onehot = torch.Tensor(self.selfie_to_onehot(selfie))
        try:
            attributes = torch.Tensor(row[self.attribute_columns].values)
        except KeyError:
            print(f"File path: {file_path} Does not have attribute columns.")

        return selfie_onehot, attributes

    def _get_file_paths(self):
        file_paths = []
        file_lengths = []
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    data = pd.read_csv(file_path, delimiter="\t")  # Adjust the delimiter if needed
                    file_lengths.append(data.shape[0])
        return file_paths, file_lengths

    def _make_alphabet(self):
        alphabet = set()
        for file_path in self.file_paths:
            file_data = pd.read_csv(file_path, delimiter="\t")
            file_alphabet = sf.get_alphabet_from_selfies(file_data["selfies"])
            alphabet = alphabet.union(file_alphabet)
        alphabet.add("[nop]")
        alphabet = sorted(alphabet)
        return alphabet

    def _get_max_selfie_length(self):
        max_length = 0
        for file_path in self.file_paths:
            file_data = pd.read_csv(file_path, delimiter="\t")
            file_max_length = max(sf.len_selfies(s) for s in file_data["selfies"])
            max_length = max(max_length, file_max_length)
        return max_length

    def _make_symbol_to_index(self):
        symbol_to_index = {}
        index_to_symbol = {}
        for i, symbol in enumerate(self.alphabet):
            symbol_to_index[symbol] = i
            index_to_symbol[i] = symbol
        return symbol_to_index, index_to_symbol

    def _divide_data_idx(self):
        total_length = sum(self.file_lengths)
        train_length = int(total_length * self.train_ratio)
        train_idx = torch.randperm(total_length)[:train_length]
        val_idx = torch.randperm(total_length)[train_length:]
        return train_idx, val_idx

    def selfie_to_onehot(self, selfie):
        one_hot = sf.selfies_to_encoding(
            selfies=selfie, vocab_stoi=self.symbol_to_idx, pad_to_len=self.pad_to_len, enc_type="one_hot"
        )
        return one_hot
