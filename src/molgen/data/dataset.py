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
        seed=42,
    ):
        self.data_folder = data_folder
        self.transform = transform
        self.attribute_columns = attribute_columns
        self.is_train_set = is_train_set
        self.symbol_to_idx = symbol_to_index
        self.idx_to_symbol = index_to_symbol
        self.pad_to_len = max_selfie_len
        self.train_ratio = train_ratio
        self.seed = seed
        self.file_paths, self.file_lengths = self._get_file_paths()
        self.data = self._get_data()
        # self.alphabet = self._make_alphabet()
        # self.max_selfie_length = self._get_max_selfie_length()
        # self.symbol_to_index, self.index_to_symbol = self._make_symbol_to_index()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Getting the data row
        row = self.data.iloc[idx]

        # Getting the selfie
        selfie = row["selfies"]

        # Making the one hot encoding of the selfie
        selfie_labels = torch.Tensor(self.selfie_to_labels(selfie)).type(torch.int64)

        # Getting the attributes
        attributes = torch.Tensor(row[self.attribute_columns]).type(torch.float)

        # Making the src, trg_input and trg_output for the transformer
        src = selfie_labels[1:-1]  # Removing the [nop] tokens from the start and end
        trg_input = selfie_labels[:-1]  # Removing the [nop] token from the end
        trg_output = selfie_labels[1:]  # Removing the [nop] token from the start

        return src, trg_input, trg_output, attributes

    def _get_file_paths(self):
        file_paths = []
        file_lengths = []
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    data = pd.read_csv(
                        file_path, delimiter="\t", dtype={"features": str}
                    )  # Adjust the delimiter if needed
                    # Drop data where attribute columns or selfie is NaN
                    data = data.dropna(subset=self.attribute_columns)
                    data = data.dropna(subset=["selfies"])
                    file_lengths.append(data.shape[0])
        return file_paths, file_lengths

    def _make_alphabet(self):
        alphabet = set()
        for file_path in self.file_paths:
            file_data = pd.read_csv(file_path, delimiter="\t", dtype={"features": str})
            file_alphabet = sf.get_alphabet_from_selfies(file_data["selfies"])
            alphabet = alphabet.union(file_alphabet)
        alphabet.add("[nop]")
        alphabet = sorted(alphabet)
        return alphabet

    def _get_max_selfie_length(self):
        max_length = 0
        for file_path in self.file_paths:
            file_data = pd.read_csv(file_path, delimiter="\t", dtype={"features": str})
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

    def _get_data(self):
        # Divide the data into train and validation
        torch.manual_seed(self.seed)
        total_length = sum(self.file_lengths)
        train_length = int(total_length * self.train_ratio)
        shuffle_idx = torch.randperm(total_length)
        train_idx = shuffle_idx[:train_length]
        val_idx = shuffle_idx[train_length:]

        # Loading data to memory to speed up indexing
        all_data = []
        total_lengths = 0
        for idx, file_path in enumerate(self.file_paths):
            # Loading the file
            data = pd.read_csv(file_path, delimiter="\t", dtype={"features": str})
            data = data.dropna(subset=self.attribute_columns)
            data = data.dropna(subset=["selfies"])
            file_length = self.file_lengths[idx]

            if self.is_train_set:
                # Getting the training and validation idxes for this file
                file_train_idxes = (
                    train_idx[torch.logical_and(train_idx < total_lengths + file_length, train_idx >= total_lengths)]
                    - total_lengths
                )

                # Adding the idxes to the train and validation data
                all_data.append(data.iloc[file_train_idxes])

            else:
                # Getting the training and validation idxes for this file
                file_val_idxes = (
                    val_idx[torch.logical_and(val_idx < total_lengths + file_length, val_idx >= total_lengths)]
                    - total_lengths
                )

                # Adding the idxes to the train and validation data
                all_data.append(data.iloc[file_val_idxes])

            # Updating the total lengths
            total_lengths += file_length

        # Concatenating the dataframes
        all_data = pd.concat(all_data)

        return all_data

    def selfie_to_onehot(self, selfie):
        # Adding the [nop] token to the start and end of the selfie to make sure the transformer learns to start and stop
        selfie_nop = "[nop]" + selfie + "[nop]"
        # pad_to_len=self.pad_to_len + 2 because we add the [nop] token to start and stop
        one_hot = sf.selfies_to_encoding(
            selfies=selfie_nop, vocab_stoi=self.symbol_to_idx, pad_to_len=self.pad_to_len + 2, enc_type="one_hot"
        )
        return one_hot

    def selfie_to_labels(self, selfie):
        # Adding the [nop] token to the start and end of the selfie to make sure the transformer learns to start and stop
        selfie_nop = "[nop]" + selfie + "[nop]"
        # pad_to_len=self.pad_to_len + 2 because we add the [nop] token to start and stop
        selfie_labels = sf.selfies_to_encoding(
            selfies=selfie_nop, vocab_stoi=self.symbol_to_idx, pad_to_len=self.pad_to_len + 2, enc_type="label"
        )
        return selfie_labels


class SelfieGeneratorDataset(Dataset):
    def __init__(self, z, conditions, max_selfie_len, symbol_to_index):
        self.z = z
        self.conditions = conditions
        self.max_selfie_len = max_selfie_len
        self.symbol_to_index = symbol_to_index

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        z = self.z[idx]
        z = z.type(torch.float)
        conditions = self.conditions[idx]
        conditions = conditions.type(torch.float)
        selfie_start = torch.ones([self.max_selfie_len]) * self.symbol_to_index["[nop]"]
        selfie_start = selfie_start.type(torch.int64)
        return z, selfie_start, conditions
