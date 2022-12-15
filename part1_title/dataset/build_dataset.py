from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os
from glob import glob

import logging
from typing import Union

_logger = logging.getLogger('train')

class FakeDataset(Dataset):
    def __init__(self, tokenizer):
        # tokenizer
        self.tokenizer = tokenizer

    def load_dataset(self, data_dir, split, saved_data_path=False):
        data_info = glob(os.path.join(data_dir, split, '*/*/*'))
        setattr(self, 'saved_data_path', saved_data_path)

        if saved_data_path:
            _logger.info('load saved data')
            data = torch.load(os.path.join(saved_data_path, f'{split}.pt'))
        else:
            _logger.info('load raw data')
                
            data = {}
            for filename in data_info:
                f = json.load(open(filename,'r'))
                data[filename] = f

        setattr(self, 'data_info', data_info)
        setattr(self, 'data', data)

    def transform(self):
        raise NotImplementedError

    def padding(self):
        raise NotImplementedError

    def __getitem__(self, i: int) -> Union[dict, int]:
        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]

            return doc, label
        
        else:
            news_info = self.data[self.data_info[i]]
        
            # label
            label = 1 if 'NonClickbait_Auto' not in self.data_info[i] else 0
        
            # transform and padding
            doc = self.transform(
                title = news_info['labeledDataInfo']['newTitle'], 
                text  = news_info['sourceDataInfo']['newsContent'].split('\n')
            )

            return doc, label

    def __len__(self):
        raise NotImplementedError

