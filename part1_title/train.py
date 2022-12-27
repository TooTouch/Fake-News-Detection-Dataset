import time
import json
import os 
import wandb
import logging

import torch
import numpy as np
from utils import convert_device
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
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



def training(model, num_training_steps: int, trainloader, validloader, criterion, optimizer, scheduler,
             log_interval: int, eval_interval: int, savedir: str, use_wandb: bool, accumulation_steps: int = 1, device: str = 'cpu'):   
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
        for inputs, targets in trainloader:
            # batch
            inputs, targets = convert_device(inputs, device), targets.to(device)

            data_time_m.update(time.time() - end)

            # optimizer condition
            opt_cond = (step + 1) % accumulation_steps == 0

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
                                rate       = inputs['input_ids'].size(0) / batch_time_m.val,
                                rate_avg   = inputs['input_ids'].size(0) / batch_time_m.avg,
                                data_time  = data_time_m))


                if (((step+1) // accumulation_steps) % eval_interval == 0 and step != 0) or step+1 == num_training_steps: 
                    eval_metrics = evaluate(model, validloader, criterion, log_interval, device)
                    model.train()

                    eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                    # wandb
                    if use_wandb:
                        wandb.log(eval_log, step=step)

                    # checkpoint
                    if best_acc < eval_metrics['acc']:
                        # save best score
                        state = {'best_step':step}
                        state.update(eval_log)
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

    # save best model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_acc, state['best_step']))
    
        
def evaluate(model, dataloader, criterion, log_interval: int, device: str = 'cpu', sample_check: bool = False):
    correct = 0
    total = 0
    total_loss = 0
    total_score = []
    total_preds = []
    total_targets = []

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = convert_device(inputs, device), targets.to(device)
            
            # predict
            outputs = model(**inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            correct += targets.eq(preds).sum().item()
            total += targets.size(0)

            total_score.extend(outputs.cpu().tolist())
            total_preds.extend(preds.cpu().tolist())
            total_targets.extend(targets.cpu().tolist())
            
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    metrics = calc_metrics(
        y_true  = total_targets,
        y_score = np.array(total_score)[:,1],
        y_pred  = total_preds
    )
    
    metrics.update([('acc',correct/total), ('loss',total_loss/len(dataloader))])

    _logger.info('TEST: Loss: %.3f | Acc: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%' % 
                (metrics['loss'], 100.*metrics['acc'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))

    if sample_check:
        results = {
            'targets': total_targets,
            'preds'  : total_preds,
            'outputs': total_score
        }
        return metrics, results
    else:
        return metrics


def calc_metrics(y_true: list, y_score: np.ndarray, y_pred: list) -> dict:
    auroc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, tp: {tp}')

    return {
        'auroc'    : auroc, 
        'f1'       : f1, 
        'recall'   : recall, 
        'precision': precision
    }
