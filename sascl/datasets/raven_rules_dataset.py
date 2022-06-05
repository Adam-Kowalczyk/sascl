import numpy as np
import random
import os

import torch
from torch.utils.data import Dataset


class RAVENRulesDataset(Dataset):
    def __init__(self, data_path, set_type='train', answers_to_pick = 2, no_attr = 5, no_rules = 4, strucutres_number = 2, structure_meta_size = 4):
        self.data_path = data_path
        self.file_names = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(self.data_path)] for val in sublist if val.endswith(f'{set_type}.npz')]
        self.file_answer_pairs = [(path, answer) for path in self.file_names for answer in [-1, *random.choices(range(0, 8 - 1), k = answers_to_pick - 1)]]
        self.no_attr = no_attr
        self.no_rules = no_rules
        self.strucutres_number = strucutres_number
        self.structure_meta_size = structure_meta_size

    def __len__(self):
        return len(self.file_answer_pairs)

    def __getitem__(self, idx):
        #item_path, answer_nr = self.file_answer_pairs[idx]
        item_path, answer_id = self.file_answer_pairs[idx]
        item_data = np.load(item_path)
        if answer_id == -1:
            answer_nr = item_data['target']
        else:
            incorrect_answers = list(range(0,8))
            incorrect_answers.remove(item_data['target'])
            answer_nr = incorrect_answers[answer_id]
        # label = item_data['target']
        all_images = item_data['image']
        # all_images = all_images[:, ::2, ::2]
        all_images = np.expand_dims(all_images, axis=1)
        all_images = np.divide(all_images, 255).astype(np.double)
        splited = np.split(all_images, 2)
        answer = splited[1][answer_nr, ...]
        answer = np.expand_dims(answer, axis=0)
        all_panels = np.concatenate((splited[0], answer), axis = 0)
        # rules:
        meta_data = item_data['meta_matrix']
        # answer_changes = item_data['meta_answer_mods']

        rules_on_attr = []

        for s in range(self.strucutres_number):
            structure_meta = meta_data[range(self.structure_meta_size * s, self.structure_meta_size * (s + 1))]
            if item_data['meta_answer_mods'][2*answer_nr + s].any():
                for mod in np.nonzero(item_data['meta_answer_mods'][2*answer_nr + s])[0]:
                    structure_meta[structure_meta[:,self.no_rules + mod] == 1]*= 0
            
            for i in range(self.no_attr):
                row = structure_meta[structure_meta[:,self.no_rules + i] == 1]
                if row.size == 0:
                    rules_on_attr.append(0)
                else:
                    row = row[0] #only first
                    rules_on_attr.append(np.nonzero(row[:self.no_rules])[0][0] + 1)

        return torch.Tensor(all_panels), torch.Tensor(rules_on_attr).long() 