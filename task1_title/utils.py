import os
import random
import torch
import numpy as np
import re
import pandas as pd

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def convert_device(inputs, device):
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)

    return inputs


def check_data(data_info, results, target, auto=None, topN=10):
    data = pd.concat([pd.DataFrame(results), pd.DataFrame(data_info)], axis=1)
    data['Auto'] = list(
        map(lambda x: re.search(r'(\w+)_(\w+)/(\w+)/(\w+).json', x).group(2) == 'Auto', data['filename']))
    data[['output_0', 'output_1']] = pd.DataFrame(data.outputs.tolist(), index=data.index)

    if auto is None:
        data = data.query(f'(targets=={target})')
    else:
        data = data.query(f'(targets=={target}) and (Auto=={auto})')
    
    err_filename = data.query(f'(targets != preds)').sort_values(
        by=f'output_{0 if target == 1 else 1}', ascending=False).head(topN)['filename']

    results = dict()
    results['num of data'] = len(data)
    results['num of errors'] = len(data.query(f'(targets != preds)'))
    results['rate of errors'] = round(results['num of errors'] / results['num of data'], 4)
    results['filenames of errors'] = list(err_filename.values)

    return results