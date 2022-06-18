
from tqdm.auto import tqdm
import torch 
import os 
import argparse 

from factory import create_tokenizer, create_dataset, create_dataloader

def save(split, dataloader, savedir):
    
    doc_list = []
    label_list = []
    for i, (doc, label) in enumerate(tqdm(dataloader, desc=split)):
        doc_list.append(doc['input_ids'])
        label_list.append(label)

    doc = torch.cat(doc_list, dim=0)
    label = torch.cat(label_list)

    torch.save({'doc':doc, 'label':label}, os.path.join(savedir,f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='HAN')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
    parser.add_argument("--data_path", type=str, default="../data/task1/")
    parser.add_argument('--savedir', type=str, default='../data/task1/')
    parser.add_argument('--tokenizer', type=str, default='mecab')
    parser.add_argument("--vocab_path", type=str, default="../word-embeddings/glove/glove.txt")
    parser.add_argument('--max_vocab_size', type=int, default=-1, help='maximum vocab size')
    parser.add_argument('--max_word_len', type=int, default=64, help='maximum word length')
    parser.add_argument('--max_sent_len', type=int, default=16, help='maximum word length')
    args = parser.parse_args()

    args.savedir = os.path.join(args.savedir, f'{args.modelname}_s{args.max_sent_len}_w{args.max_word_len}')

    os.makedirs(args.savedir, exist_ok=True)

    # tokenizer
    tokenizer, word_embed = create_tokenizer(args)

    # Build datasets
    trainset = create_dataset(args, 'train', tokenizer)
    validset = create_dataset(args, 'valid', tokenizer)
    testset = create_dataset(args, 'test', tokenizer)

    trainloader = create_dataloader(args, trainset, 'train')
    validloader = create_dataloader(args, validset, 'valid')
    testloader = create_dataloader(args, testset, 'test')

    # save
    save('train', trainloader, args.savedir)
    save('valid', validloader, args.savedir)
    save('test', testloader, args.savedir)