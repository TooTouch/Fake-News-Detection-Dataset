import pandas as pd
import csv

from konlpy.tag import Mecab
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def create_dataset(name, data_path, split, tokenizer, vocab, **kwargs):
    dataset = __import__('dataset').__dict__[f'{name}Dataset'](
        datadir         = data_path,
        split           = split,
        tokenizer       = tokenizer,
        vocab           = vocab,
        **kwargs
    )
    return dataset


def create_dataloader(dataset, batch_size, num_workers, shuffle=False):
    dataloader = DataLoader(
        dataset, 
        batch_size  = batch_size, 
        num_workers = num_workers, 
        shuffle     = shuffle
    )

    return dataloader