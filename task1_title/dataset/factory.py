import pandas as pd
import csv

from konlpy.tag import Mecab
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from .build_dataset import FNDTokenizer, FNDDataset


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



def create_tokenizer(tokenizer, vocab_path, max_vocab_size, pretrained_name):
    if tokenizer == 'mecab':
        vocab, word_embed = extract_word_embedding(vocab_path = vocab_path, max_vocab_size = max_vocab_size)
        tokenizer = FNDTokenizer(vocab = vocab, tokenizer = Mecab())
    elif tokenizer == 'bert':
        word_embed = None
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    return tokenizer, word_embed 


def create_dataset(modelname, data_path, split, tokenizer, max_word_len, max_sent_len, use_saved_data):
    dataset = FNDDataset(
        modelname      = modelname,
        datadir        = data_path,
        split          = split,  
        tokenizer      = tokenizer, 
        max_word_len   = max_word_len, 
        max_sent_len   = max_sent_len,
        use_saved_data = use_saved_data
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