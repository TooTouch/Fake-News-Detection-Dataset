from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os

import logging

_logger = logging.getLogger('train')

class FakeDataset(Dataset):
    def __init__(self, tokenizer):
        # tokenizer
        self.tokenizer = tokenizer

    def load_dataset(self, data_dir, data_info_dir, split, saved_data_path=False):
        data_info = pd.read_csv(os.path.join(data_info_dir, f'{split}_info.csv'))
        setattr(self, 'saved_data_path', saved_data_path)

        if saved_data_path:
            _logger.info('load saved data')
            data = torch.load(os.path.join(saved_data_path, f'{split}.pt'))
        else:
            _logger.info('load raw data')
                
            data = {}
            for filename in data_info.filename:
                f = json.load(open(os.path.join(data_dir, filename),'r'))
                data[filename] = f

        setattr(self, 'data_info', data_info)
        setattr(self, 'data', data)

    def transform(self):
        raise NotImplementedError

    def padding(self):
        raise NotImplementedError

    def __getitem__(self, i):
        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]

            return doc, label
        
        else:
            news_idx = self.data_info.iloc[i]
            news_info = self.data[news_idx['filename']]
        
            # label
            label = 1 if news_idx['label']=='fake' else 0
        
            # transform and padding
            doc = self.transform(
                title = news_info['labeledDataInfo']['newTitle'], 
                text  = news_info['sourceDataInfo']['newsContent'].split('\n')
            )

            return doc, label

    def __len__(self):
        raise NotImplementedError

