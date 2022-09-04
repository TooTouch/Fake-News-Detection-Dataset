import time
import json
import os 
import wandb
import logging
from collections import OrderedDict

import torch
from utils import convert_device
import transformers

_logger = logging.getLogger('train')
transformers.logging.set_verbosity_error()

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


def training(model, num_training_steps, trainloader, validloader, criterion, optimizer, scheduler,
             log_interval, eval_interval, savedir, use_wandb, accumulation_steps=1, device='cpu'):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    best_acc = 0
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()

    step = 0
    train_mode = True
    while train_mode:
        for batch in trainloader:
            # batch
            batch = convert_device(batch, device)
            src       = batch['src']
            segs      = batch['segs']
            clss      = batch['clss']
            mask_src  = batch['mask_src']
            mask_cls  = batch['mask_cls']
            src_txt   = batch['src_txt']
            targets   = batch['seg_label']

            data_time_m.update(time.time() - end)

            # optimizer condition
            opt_cond = (step + 1) % accumulation_steps == 0

            # predict
            outputs = model(src, segs, clss, mask_src, mask_cls)
            loss = criterion(outputs, targets)

            # loss for accumulation steps
            loss /= accumulation_steps        
            loss.backward()

            if opt_cond:
                # loss update
                optimizer.step()
                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

                losses_m.update(loss.item()*accumulation_steps)

                # accuracy
                preds = outputs.argmax(dim=1)

                acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
                batch_time_m.update(time.time() - end)

                # wandb
                if use_wandb:
                    wandb.log({
                        'train_acc':acc_m.val,
                        'train_loss':losses_m.val
                    },
                    step=step)
                
                if ((step+1) // accumulation_steps) % log_interval == 0 or step == 0:
                    _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Acc: {acc.avg:.3%} '
                                'LR: {lr:.3e} '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (step+1)//accumulation_steps, num_training_steps,
                                loss       = losses_m,
                                acc        = acc_m, 
                                lr         = optimizer.param_groups[0]['lr'],
                                batch_time = batch_time_m,
                                rate       = src.size(0) / batch_time_m.val,
                                rate_avg   = src.size(0) / batch_time_m.avg,
                                data_time  = data_time_m))

                if ((step+1) // accumulation_steps) % eval_interval == 0 and step != 0: 
                    eval_metrics = evaluate(model, validloader, criterion, log_interval, device)
                    model.train()
                    # wandb
                    if use_wandb:
                        wandb.log({
                            'eval_acc':eval_metrics['acc'],
                            'eval_loss':eval_metrics['loss']
                        },
                        step=step)

                    # checkpoint
                    if best_acc < eval_metrics['acc']:
                        # save best score
                        state = {'best_step':step, 'best_acc':eval_metrics['acc']}
                        json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent=4)

                        # save best model
                        torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                        
                        _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

                        best_acc = eval_metrics['acc']

            end = time.time()
            step += 1
            
            if (step // accumulation_steps) >= num_training_steps:
                train_mode = False
                break
        # reset dataset
        
        
    # save best model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(state['best_acc'], state['best_step']))
    
        
def evaluate(model, dataloader, criterion, log_interval, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = convert_device(batch, device)
            src       = batch['src']
            segs      = batch['segs']
            clss      = batch['clss']
            mask_src  = batch['mask_src']
            mask_cls  = batch['mask_cls']
            src_txt   = batch['src_txt']
            targets   = batch['seg_label']
            #inputs, targets = convert_device(inputs, device), targets.to(device)

            # predict
            outputs = model(src, segs, clss, mask_src, mask_cls)
            
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
        