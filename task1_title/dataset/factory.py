import pandas as pd
import csv

from konlpy.tag import Mecab
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from .dataloader import FNDTokenizer, FNDDataset


def extract_word_embedding(vocab_path, max_vocab_size=-1):
    word_embed = pd.read_csv(
        filepath_or_buffer = vocab_path, 
        header             = None, 
        sep                = " ", 
        quoting            = csv.QUOTE_NONE
    ).values

    word_embed = word_embed[:-max_vocab_size] if max_vocab_size != -1 else word_embed

    vocab = list(word_embed[:,0])
    word_embed = word_embed[:,1:]

    return vocab, word_embed



def create_tokenizer(args):
    if args.tokenizer == 'mecab':
        vocab, word_embed = extract_word_embedding(vocab_path = args.vocab_path, max_vocab_size = args.max_vocab_size)
        tokenizer = FNDTokenizer(vocab = vocab, tokenizer = Mecab())
    elif args.tokenizer == 'bert':
        word_embed = None
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)

    return tokenizer, word_embed 


def create_dataset(args, split, tokenizer):
    dataset = FNDDataset(
        modelname      = args.modelname,
        datadir        = args.data_path,
        split          = split,  
        tokenizer      = tokenizer, 
        max_word_len   = args.max_word_len, 
        max_sent_len   = args.max_sent_len,
        use_saved_data = args.use_saved_data
    )

    return dataset


def create_dataloader(args, dataset, shuffle=False):

    dataloader = DataLoader(
        dataset, 
        batch_size  = args.batch_size, 
        num_workers = args.num_workers, 
        shuffle     = shuffle
    )

    return dataloader