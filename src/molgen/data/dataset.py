import os

import pandas as pd
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
