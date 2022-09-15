import pandas as pd
import csv

from konlpy.tag import Mecab
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from .build_dataset import *
from .tokenizer import FNDTokenizer


def extract_word_embedding(vocab_path, max_vocab_size=-1):
    word_embed = pd.read_csv(
        filepath_or_buffer = vocab_path, 
        header             = None, 
        sep                = " ", 
        quoting            = csv.QUOTE_NONE
    ).values

    word_embed = word_embed[:max_vocab_size] if max_vocab_size != -1 else word_embed

    vocab = list(word_embed[:,0])
    word_embed = word_embed[:,1:]

    return vocab, word_embed



def create_tokenizer(name, vocab_path, max_vocab_size, pretrained_name):
    if name == 'mecab':
        vocab, word_embed = extract_word_embedding(vocab_path = vocab_path, max_vocab_size = max_vocab_size)
        tokenizer = FNDTokenizer(vocab = vocab, tokenizer = Mecab())
    elif name == 'bert':
        word_embed = None
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    return tokenizer, word_embed 


def create_dataset(name, data_path, split, tokenizer, saved_data_path, **kwargs):
    dataset = __import__('dataset').__dict__[f'{name}Dataset'](
        tokenizer = tokenizer,
        **kwargs
    )

    dataset.load_dataset(
        datadir         = data_path, 
        split           = split, 
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