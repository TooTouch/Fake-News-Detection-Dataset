import os
import random
import torch
import numpy as np
import pandas as pd

def torch_seed(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def convert_device(inputs: dict, device: str) -> dict:
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)

    return inputs


def extract_wrong_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # extract category: Clickbait_Auto, Clickbait_Direct, NoneClickbait_Auto
    df['category'] = df['filename'].apply(lambda x: os.path.basename(os.path.abspath(os.path.join(x, '../../'))))

    # wonrg case
    df_incorrect = df[df.targets!=df.preds]

    # total count per category
    cnt_cat = df.category.value_counts().reset_index()
    cnt_cat.columns = ['category','total_cnt']

    # wrong count per category
    cnt_wrong_cat = df_incorrect.category.value_counts().reset_index()
    cnt_wrong_cat.columns = ['category','wrong_cnt']

    # merge and summary
    cnt_df = pd.merge(cnt_cat, cnt_wrong_cat, on='category', how='inner')
    cnt_df['wrong / total (%)'] = cnt_df.apply(
        lambda x: f'{x.wrong_cnt} / {x.total_cnt} ({x.wrong_cnt/x.total_cnt:.2%})', axis=1)
    cnt_df = cnt_df[['category','wrong / total (%)']]
    
    return cnt_df


def select_wrong_case_topN(df: pd.DataFrame, cat: str, n: int):
    assert cat in ['Clickbait_Direct','Clickbait_Auto','NonClickbait_Auto'], "cat should be either 'Clickbait_Direct','Clickbait_Auto','NonClickbait_Auto'"
    # define wrong pred 
    if cat in ['Clickbait_Direct','Clickbait_Auto']:
        pred = 0
    elif cat == 'NonClickbait_Auto':
        pred = 1
    
    # extract category: Clickbait_Auto, Clickbait_Direct, NonClickbait_Auto
    df['category'] = df['filename'].apply(lambda x: os.path.basename(os.path.abspath(os.path.join(x, '../../'))))
    
    # wonrg case
    df_incorrect = df[df.targets!=df.preds]
    
    # select top N
    wrong_case = pd.concat([
            df_incorrect[df_incorrect.category==cat]['filename'],
            df_incorrect[df_incorrect.category==cat]['outputs'].apply(lambda x: eval(x)[pred])
        ],
        axis=1
    ).sort_values('outputs',ascending=False).head(n)

    return wrong_case