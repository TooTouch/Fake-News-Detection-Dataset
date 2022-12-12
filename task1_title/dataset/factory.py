import pandas as pd
import csv

from konlpy.tag import Mecab
from torch.utils.data import DataLoader

import gluonnlp as nlp

from kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

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



def create_tokenizer(name, vocab_path, max_vocab_size):
    if name == 'mecab':
        vocab, word_embed = extract_word_embedding(vocab_path = vocab_path, max_vocab_size = max_vocab_size)
        tokenizer = FNDTokenizer(vocab = vocab, tokenizer = Mecab())
    elif name == 'bert':
        word_embed = None
        _, vocab = get_pytorch_kobert_model(cachedir=".cache")
        tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

    return tokenizer, word_embed 


def create_dataset(name, data_path, split, tokenizer, saved_data_path, **kwargs):
    dataset = __import__('dataset').__dict__[f'{name}Dataset'](
        tokenizer = tokenizer,
        **kwargs
    )

    dataset.load_dataset(
        data_dir        = data_path, 
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