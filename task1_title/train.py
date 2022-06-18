import time
import json
import os 
import wandb
import logging
from collections import OrderedDict

import torch
from utils import convert_device

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train(model, dataloader, criterion, optimizer, log_interval, accumulation_steps=1, device='cpu'):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        # optimizer condition
        opt_cond = (idx + 1) % accumulation_steps == 0

        inputs, targets = convert_device(inputs, device), targets.to(device)

        # predict
        outputs = model(**inputs)
        loss = criterion(outputs, targets)
        # loss for accumulation steps
        loss /= accumulation_steps        
        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()

            losses_m.update(loss.item()*accumulation_steps)

            # accuracy
            preds = outputs.argmax(dim=1) 
            acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
            
            batch_time_m.update(time.time() - end)
        
            if (idx // accumulation_steps) % log_interval == 0 or idx == 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                             'Acc: {acc.avg:.3%} '
                             'LR: {lr:.3e} '
                             'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                             'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                             (idx+1)//accumulation_steps, len(dataloader)//accumulation_steps, 
                             loss       = losses_m, 
                             acc        = acc_m, 
                             lr         = optimizer.param_groups[0]['lr'],
                             batch_time = batch_time_m,
                             rate       = inputs['input_ids'].size(0) / batch_time_m.val,
                             rate_avg   = inputs['input_ids'].size(0) / batch_time_m.avg,
                             data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
def evaluate(model, dataloader, criterion, log_interval, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = convert_device(inputs, device), targets.to(device)
            
            # predict
            outputs = model(**inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
                
def training(
    model, epochs, trainloader, validloader, criterion, optimizer, scheduler, 
    savedir, log_interval, accumulation_steps=1, device='cpu', use_wandb=False
):
    
    best_acc = 0

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, accumulation_steps, device)
        eval_metrics = evaluate(model, validloader, criterion, log_interval, device)

        scheduler.step()

        # wandb
        metrics = OrderedDict(epoch=epoch)
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        if use_wandb:
            wandb.log(metrics)
    
        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save best score
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent=4)

            # save best model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

    # save best model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))
