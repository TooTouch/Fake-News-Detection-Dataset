
from tqdm.auto import tqdm
import torch 
import os 
import argparse 

from dataset import create_dataset, create_dataloader

import gluonnlp as nlp
from kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

def save(split, dataloader, savedir):
    
    doc_dict = {}
    label_list = []
    for i, batch in enumerate(tqdm(dataloader, desc=split)):
        if len(doc_dict) == 0:
            for k in batch.keys():
                doc_dict[k] = []
            
        for k in batch.keys():
            doc_dict[k].append(batch[k])

    for k in doc_dict.keys():
        doc_dict[k] = torch.cat(doc_dict[k])

    torch.save(doc_dict, os.path.join(savedir,f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='KoBERTSeg', choices=['KoBERTSeg','BTS'])
    parser.add_argument("--saved_data_path", type=str, default=None, help='save data path')
    parser.add_argument("--data_path", type=str, default="../data/task2/")
    parser.add_argument('--savedir', type=str, default='../data/task2/')
    
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
    
    parser.add_argument('--window_size', type=int, default=3, help='window_size')
    parser.add_argument('--max_word_len', type=int, default=512, help='maximum word length')
    args = parser.parse_args()


    dataname = f'{args.modelname}_ws{args.window_size}_max-len{args.max_word_len}'
    args.savedir = os.path.join(args.savedir, dataname)

    os.makedirs(args.savedir, exist_ok=True)

    # tokenizer
    _, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

    # Build datasets
    for split in ['train','valid','test']:
        dataset = create_dataset(args, split, tokenizer, vocab)
        dataloader = create_dataloader(args, dataset)
   
        # save
        save(split, dataloader, args.savedir)