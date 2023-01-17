import numpy as np
import wandb
import logging
import os
import torch
import argparse
import json
import yaml

from models import create_model 
from dataset import create_dataset, create_dataloader
from transformers import get_cosine_schedule_with_warmup

import torch
import gluonnlp as nlp

from kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

from train import training, evaluate
from log import setup_default_logging
from utils import torch_seed

import pandas as pd


_logger = logging.getLogger('train')


def run(cfg):
    # setting seed and device
    setup_default_logging()
    torch_seed(cfg['SEED'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # savedir
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    # tokenizer
    _, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
    
    # Build Model
    model = create_model(
        modelname  = cfg['MODEL']['modelname'], 
        hparams    = cfg['MODEL']['PARAMETERS'],
        checkpoint_path = cfg['MODEL']['CHECKPOINT']['checkpoint_path']
    )
    model.to(device)

    _logger.info('# of trainable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
    
    # Objective function
    criterion = torch.nn.CrossEntropyLoss()

    if cfg['MODE']['do_train']:
        # wandb
        if cfg['TRAIN']['use_wandb']:
            wandb.init(name=cfg['EXP_NAME'], project='Fake-News-Detection-Task2', config=cfg)

        # Build datasets
        trainset = create_dataset(
            name            = cfg['DATASET']['name'],
            data_path       = cfg['DATASET']['data_path'], 
            split           = 'train', 
            tokenizer       = tokenizer, 
            vocab           = vocab,
            **cfg['DATASET']['PARAMETERS']
        )
        validset = create_dataset(
            name            = cfg['DATASET']['name'],
            data_path       = cfg['DATASET']['data_path'], 
            split           = 'validation', 
            tokenizer       = tokenizer, 
            vocab           = vocab,
            **cfg['DATASET']['PARAMETERS']
        )
        
        trainloader = create_dataloader(
            dataset     = trainset, 
            batch_size  = cfg['TRAIN']['batch_size'],
            num_workers = cfg['TRAIN']['num_workers'],
            shuffle     = True
        )
        validloader = create_dataloader(
            dataset     = validset, 
            batch_size  = cfg['TRAIN']['batch_size'],
            num_workers = cfg['TRAIN']['num_workers']
        )

        # Set training
        
        optimizer = torch.optim.AdamW(
            params       = filter(lambda p: p.requires_grad, model.parameters()), 
            lr           = cfg['OPTIMIZER']['lr'], 
            weight_decay = cfg['OPTIMIZER']['weight_decay']
        )

        # scheduler
        if cfg['SCHEDULER']['use_scheduler']:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps   = int(cfg['TRAIN']['num_training_steps'] * cfg['SCHEDULER']['warmup_ratio']), 
                num_training_steps = cfg['TRAIN']['num_training_steps'])
        else:
            scheduler = None

        # Fitting model
        training(
            model              = model, 
            num_training_steps = cfg['TRAIN']['num_training_steps'], 
            trainloader        = trainloader, 
            validloader        = validloader, 
            criterion          = criterion, 
            optimizer          = optimizer, 
            scheduler          = scheduler,
            log_interval       = cfg['LOG']['log_interval'],
            eval_interval      = cfg['LOG']['eval_interval'],
            savedir            = savedir,
            accumulation_steps = cfg['TRAIN']['accumulation_steps'],
            device             = device,
            use_wandb          = cfg['TRAIN']['use_wandb']
        )

    elif cfg['MODE']['do_test']:
        for split in cfg['MODE']['test_list']:
            try:
                _logger.info('{} evaluation'.format(split.upper()))
                dataset = create_dataset(
                    name            = cfg['DATASET']['name'],
                    data_path       = cfg['DATASET']['data_path'], 
                    split           = split, 
                    tokenizer       = tokenizer, 
                    vocab           = vocab,
                    **cfg['DATASET']['PARAMETERS']
                )
                
                dataloader = create_dataloader(
                    dataset     = dataset, 
                    batch_size  = cfg['TRAIN']['batch_size'],
                    num_workers = cfg['TRAIN']['num_workers']
                )

                metrics, exp_results = evaluate(
                    model        = model, 
                    dataloader   = dataloader, 
                    criterion    = criterion,
                    log_interval = cfg['LOG']['log_interval'],
                    device       = device,
                    sample_check = True
                )
                
                # save exp result
                exp_results = pd.concat([pd.DataFrame({'filename':dataset.data_info}), pd.DataFrame(exp_results)], axis=1)
                exp_results['label'] = exp_results['filename'].apply(lambda x: 0 if 'NonClickbait' in x else 1)
                exp_results.to_csv(os.path.join(savedir, f'exp_results_{split}.csv'), index=False)

                # save result metrics
                json.dump(metrics, open(os.path.join(savedir, f"{cfg['RESULT']['result_name']}_{split}.json"),'w'), indent=4)
            except:
                print(f'{split} folder does not exist')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fake News Detection - Task2')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    run(cfg)