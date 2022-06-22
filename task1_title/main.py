import numpy as np
import wandb
import logging
import os
import torch
import argparse

from models import create_model 
from dataset import create_dataset, create_dataloader, create_tokenizer
from transformers import get_cosine_schedule_with_warmup
from train import training, evaluate

from log import setup_default_logging
from utils import torch_seed

import pandas as pd

_logger = logging.getLogger('train')


def get_args(notebook=False):
    parser = argparse.ArgumentParser(description='Fake News Detection - Task1')

    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--modelname', type=str, default='HAN', choices=['HAN','FNDNet','BTS'],help='model name')
    parser.add_argument('--seed', type=int, default=223, help='seed')

    parser.add_argument('--do_train', action='store_true', help='training mode')
    parser.add_argument('--do_test', action='store_true', help='testing mode')

    parser.add_argument('--savedir', type=str, default='./saved_model', help='save directory')
    parser.add_argument('--result_path', type=str, default='./results.csv', help='result file')

    # training
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument("--num_training_steps", type=int, default=1, help='number of training steps') 
    parser.add_argument("--log_interval", type=int, default=1, help='log interval')
    parser.add_argument("--eval_interval", type=int, default=1000, help='eval interval')
    parser.add_argument("--accumulation_steps", type=int, default=1, help='number of accumulation steps')

    # optimizer
    parser.add_argument("--use_scheduler", action='store_true', help='use scheduler')
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help='learning rate warmup ratio')
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # dataset
    parser.add_argument("--use_saved_data", action='store_true', help='use saved data')
    parser.add_argument("--tokenizer", type=str, default='mecab', choices=['mecab','bert'], help='tokenizer name')
    parser.add_argument("--vocab_path", type=str, default="../word-embeddings/glove/glove.txt")
    parser.add_argument("--data_path", type=str, default="../data/task1/")
    parser.add_argument("--num_classes", type=int, default=2, help='number of class')
    parser.add_argument('--max_vocab_size', type=int, default=-1, help='maximum vocab size')
    parser.add_argument("--max_sent_len", type=int, default=128, help='maximum number of sentences in a document')
    parser.add_argument('--max_word_len', type=int, default=128, help='maximum number of words in a sentence')
    parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
    parser.add_argument('--use_pin_memory', action='store_true', help='use pin memory')

    # models
    parser.add_argument("--pretrained_name", type=str, default='klue/bert-base')
    parser.add_argument("--pretrained_path", type=str, default=None, help='pretrained model path')
    parser.add_argument("--dims", type=int, default=128, help='embedding dimension')
    parser.add_argument("--word_dims", type=int, default=32)
    parser.add_argument("--sent_dims", type=int, default=64)
    parser.add_argument("--use_pretrained_word_embed", action='store_true', help='use pretrained word embedding')
    parser.add_argument("--freeze_word_embed", action='store_true', help='freeze pretrained word embedding')

    # wandb
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')

    
    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
        
    return args


def run(args):

    # setting seed and device
    setup_default_logging()
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # tokenizer
    tokenizer, word_embed = create_tokenizer(args)
    
    # Build Model
    model = create_model(args, word_embed, tokenizer)
    model.to(device)

    _logger.info('# of trainable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    if args.do_train:
        # wandb
        if args.use_wandb:
            wandb.init(name=args.exp_name, project='Fake New Detection - Task1', config=args)

        # savedir
        savedir = os.path.join(args.savedir, args.exp_name)
        os.makedirs(savedir, exist_ok=True)

        # Build datasets
        trainset = create_dataset(args, 'train', tokenizer)
        validset = create_dataset(args, 'valid', tokenizer)
        trainloader = create_dataloader(args, trainset, True)
        validloader = create_dataloader(args, validset)

        # Set training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        # scheduler
        if args.use_scheduler:
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps   = int(args.num_training_steps * args.warmup_ratio), 
                num_training_steps = args.num_training_steps)
        else:
            scheduler = None

        # Fitting model
        training(
            model              = model, 
            num_training_steps = args.num_training_steps, 
            trainloader        = trainloader, 
            validloader        = validloader, 
            criterion          = criterion, 
            optimizer          = optimizer, 
            scheduler          = scheduler,
            log_interval       = args.log_interval,
            eval_interval      = args.eval_interval,
            savedir            = savedir,
            accumulation_steps = args.accumulation_steps,
            device             = device,
            use_wandb          = args.use_wandb
        )

    elif args.do_test:
        trainset = create_dataset(args, 'train', tokenizer)
        validset = create_dataset(args, 'valid', tokenizer)
        testset = create_dataset(args, 'test', tokenizer)

        trainloader = create_dataloader(args, trainset)
        validloader = create_dataloader(args, validset)
        testloader = create_dataloader(args, testset)

        criterion = torch.nn.CrossEntropyLoss()

        # Build Model
        pretrained_path = os.path.join(args.savedir, args.exp_name, 'best_model.pt')
        model = create_model(args, word_embed, tokenizer, pretrained_path)
        model.to(device)

        # result path
        if os.path.isfile(args.result_path):
            df = pd.read_csv(args.result_path)
        else:
            df = pd.DataFrame()

        total_metrics = {}
        for split, dataloader in {'train':trainloader, 'valid':validloader, 'test':testloader}.items():
            metrics = evaluate(
                model        = model, 
                dataloader   = dataloader, 
                criterion    = criterion,
                log_interval = args.log_interval,
                device       = device
            )

            for k, v in metrics.items():
                total_metrics[f'{split}_{k}'] = v

        total_metrics['exp_name'] = args.exp_name
        df = df.append(total_metrics, ignore_index=True)
        df.to_csv(args.result_path, index=False)
        


if __name__=='__main__':
    args = get_args()

    run(args)