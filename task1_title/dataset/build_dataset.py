from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os


class FakeDataset(Dataset):
    def __init__(self, tokenizer):
        # tokenizer
        self.tokenizer = tokenizer

    def load_dataset(self, datadir, split, saved_data_path=False):
        if self.saved_data_path:
            self.data = torch.load(os.path.join(saved_data_path, f'{split}.pt'))
        else:
            data = json.load(open(os.path.join(datadir, f'{split}.json'),'r'))
            data_info = pd.read_csv(os.path.join(datadir, f'{split}_info.csv'))

            setattr(self, 'data_info', data_info)
        setattr(self, 'data', data)

    def transform(self, sent_list):
        raise NotImplementedError

    def padding(self, doc):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

