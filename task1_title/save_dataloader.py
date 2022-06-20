
from tqdm.auto import tqdm
import torch 
import os 
import argparse 

from dataset import create_tokenizer, create_dataset, create_dataloader
import transformers

transformers.logging.set_verbosity_error()

def save(split, dataloader, savedir):
    
    doc_dict = {}
    label_list = []
    for i, (doc, label) in enumerate(tqdm(dataloader, desc=split)):
        if len(doc_dict) == 0:
            for k in doc.keys():
                doc_dict[k] = []
            
        for k in doc.keys():
            doc_dict[k].append(doc[k])
        label_list.append(label)

    for k in doc_dict.keys():
        doc_dict[k] = torch.cat(doc_dict[k])
    label_list = torch.cat(label_list)

    torch.save({'doc':doc_dict, 'label':label_list}, os.path.join(savedir,f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='HAN', choices=['HAN','FNDNet','BTS'])
    parser.add_argument("--use_saved_data", action='store_true', help='use saved data')
    parser.add_argument("--pretrained_name", type=str, default='klue/bert-base')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
    parser.add_argument("--data_path", type=str, default="../data/task1/")
    parser.add_argument('--savedir', type=str, default='../data/task1/')
    parser.add_argument('--tokenizer', type=str, default='mecab', choices=['mecab','bert'])
    parser.add_argument("--vocab_path", type=str, default="../word-embeddings/glove/glove.txt")
    parser.add_argument('--max_vocab_size', type=int, default=-1, help='maximum vocab size')
    parser.add_argument('--max_word_len', type=int, default=64, help='maximum word length')
    parser.add_argument('--max_sent_len', type=int, default=16, help='maximum sent length')
    args = parser.parse_args()

    if args.modelname == 'HAN':
        dataname = f'{args.modelname}_s{args.max_sent_len}_w{args.max_word_len}'
    elif args.modelname in ['FNDNet','BTS']:
        dataname = f'{args.modelname}_w{args.max_word_len}'

    args.savedir = os.path.join(args.savedir, dataname)

    os.makedirs(args.savedir, exist_ok=True)

    # tokenizer
    tokenizer, word_embed = create_tokenizer(args)

    # Build datasets
    trainset = create_dataset(args, 'train', tokenizer)
    validset = create_dataset(args, 'valid', tokenizer)
    testset = create_dataset(args, 'test', tokenizer)

    trainloader = create_dataloader(args, trainset)
    validloader = create_dataloader(args, validset)
    testloader = create_dataloader(args, testset)

    # save
    save('train', trainloader, args.savedir)
    save('valid', validloader, args.savedir)
    save('test', testloader, args.savedir)