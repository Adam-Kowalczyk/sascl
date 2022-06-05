import numpy as np
import os

import torch
from torch.utils.data import Dataset

class RAVENDataset(Dataset):
    def __init__(self, data_path, set_type='train'):
        self.data_path = data_path
        self.file_names = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(self.data_path)] for val in sublist if val.endswith(f'{set_type}.npz')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        item_path = self.file_names[idx]
        item_data = np.load(item_path)
        label = item_data['target']
        all_images = item_data['image']
        all_images = np.expand_dims(all_images, axis=1)
        all_images = np.divide(all_images, 255).astype(np.double)
        splited = np.split(all_images, 2)
        return torch.Tensor(splited[0]), torch.Tensor(splited[1]), torch.from_numpy(label).long() 