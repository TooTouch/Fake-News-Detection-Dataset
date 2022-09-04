import pandas as pd
import csv

from konlpy.tag import Mecab
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from .build_dataset import SegData


def create_dataset(data_path, window_size, max_word_len, saved_data_path, split, tokenizer, vocab):
    dataset = SegData(
        datadir         = data_path,
        split           = split,
        window_size     = window_size,
        tokenizer       = tokenizer,
        vocab           = vocab,
        max_word_len    = max_word_len, 
        saved_data_path = saved_data_path
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