import numpy as np
import wandb
import logging
import os
import torch
import argparse

from models import create_model 
from dataset import create_dataset, create_dataloader
from transformers import get_cosine_schedule_with_warmup

import torch
import gluonnlp as nlp

from kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from kobert.utils import download as _download

from train import training, evaluate
from log import setup_default_logging
from utils import torch_seed

import pandas as pd

_logger = logging.getLogger('train')


def get_args(notebook=False):
    parser = argparse.ArgumentParser(description='Fake News Detection - Task2')

    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--modelname', type=str, default='kobertseg', help='model name')
    parser.add_argument('--seed', type=int, default=223, help='seed')

    parser.add_argument('--do_train', action='store_true', help='training mode')
    parser.add_argument('--do_test', action='store_true', help='testing mode')

    # directory
    parser.add_argument("--data_path", type=str, default="../data/task2")
    parser.add_argument('--savedir', type=str, default='./saved_model', help='save directory')
    parser.add_argument('--result_path', type=str, default='./results.csv', help='result file')

    # training
    parser.add_argument("--batch_size", type=int, default=8, help='batch size')
    parser.add_argument("--test_batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--num_training_steps", type=int, default=10000, help='number of training steps') 
    parser.add_argument("--log_interval", type=int, default=1, help='log interval')
    parser.add_argument("--eval_interval", type=int, default=1000, help='eval interval')
    parser.add_argument("--accumulation_steps", type=int, default=1, help='number of accumulation steps')

    # optimizer
    parser.add_argument("--use_scheduler", action='store_true', help='use scheduler')
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help='learning rate warmup ratio')
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # dataset
    parser.add_argument("--saved_data_path", type=str, default=None, help='save data path')
    parser.add_argument("--num_classes", type=int, default=2, help='number of class')
    parser.add_argument('--max_word_len', type=int, default=512, help='maximum number of words in a sentence')
    parser.add_argument('--num_workers', default=12, type=int, help='number of workers')

    # models
    parser.add_argument("--checkpoint_path", type=str, default=None, help='use checkpoint path')
    parser.add_argument("--finetune_bert", action='store_true')
    parser.add_argument("--window_size", type=int, default=3, help='size of window')
    parser.add_argument("--pretrained", action='store_true', help='download pretrained model')

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
    args.device = device
    
    _logger.info('Device: {}'.format(device))

    # tokenizer
    _, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
    
    # Build Model
    model = create_model(
        modelname  = args.modelname, 
        pretrained = args.pretrained, 
        tokenizer  = tokenizer, 
        args       = args
    )
    model.to(device)
    
    # kyoosung
    args.checkpoint = None

    _logger.info('# of trainable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    if args.do_train:
        # wandb
        if args.use_wandb:
            wandb.init(name=args.exp_name, project='Fake New Detection - Task2', config=args)

        # savedir
        savedir = os.path.join(args.savedir, args.exp_name)
        os.makedirs(savedir, exist_ok=True)

        # Build datasets
        trainset = create_dataset(
            data_path       = args.data_path, 
            window_size     = args.window_size, 
            max_word_len    = args.max_word_len, 
            saved_data_path = args.saved_data_path, 
            split           = 'train', 
            tokenizer       = tokenizer, 
            vocab           = vocab
        )
        validset = create_dataset(
            data_path       = args.data_path, 
            window_size     = args.window_size, 
            max_word_len    = args.max_word_len, 
            saved_data_path = args.saved_data_path, 
            split           = 'valid', 
            tokenizer       = tokenizer, 
            vocab           = vocab
        )
        
        trainloader = create_dataloader(
            dataset     = trainset, 
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            shuffle     = True
        )
        validloader = create_dataloader(
            dataset     = validset, 
            batch_size  = args.test_batch_size,
            num_workers = args.num_workers
        )

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
        model = create_model(args.modelname, args.pretrained, word_embed, tokenizer, args)
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