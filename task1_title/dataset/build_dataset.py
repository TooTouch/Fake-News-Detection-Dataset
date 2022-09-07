from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os


class FakeDataset(Dataset):
    def __init__(self, datadir, split, tokenizer):

        self.split = split

        # load data
        self.data = json.load(open(os.path.join(datadir, f'{split}.json'),'r'))
        self.data_info = pd.read_csv(os.path.join(datadir, f'{split}_info.csv'))
        
        # tokenizer
        self.tokenizer = tokenizer

    def transform(self, sent_list):
        raise NotImplementedError

    def padding(self, doc):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

