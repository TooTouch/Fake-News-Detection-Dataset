import os
import random
import torch
import numpy as np
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
        if k != 'src_txt':
            inputs[k] = inputs[k].to(device)
    return inputs


def extract_wrong_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # extract category: Clickbait_Auto, Clickbait_Direct, NoneClickbait_Auto
    df['category'] = df['filename'].apply(lambda x: os.path.basename(os.path.abspath(os.path.join(x, '../../'))))

    # wonrg case
    df_incorrect = df[df.pred_per_article==0]

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


def select_wrong_case_topN(df: pd.DataFrame, cat: str, sort_target: str, n: int):
    assert cat in ['Clickbait_Direct','Clickbait_Auto','NonClickbait_Auto'], "cat should be either 'Clickbait_Direct','Clickbait_Auto','NonClickbait_Auto'"
    assert sort_target in ['cnt_wrong','ratio_wrong','max_wrong_score'], "cat should be either 'cnt_wrong','ratio_wrong','max_wrong_score'"
        
    def get_max_score(y_true, y_pred, y_score):
        y_true = eval(y_true)
        y_pred = eval(y_pred)
        y_score = eval(y_score)

        return max([y_score[idx][y_pred[idx]] for idx, y_true_i in enumerate(y_true) if y_true_i != y_pred[idx]])
    
    # extract category: Clickbait_Auto, Clickbait_Direct, NoneClickbait_Auto
    df['category'] = df['filename'].apply(lambda x: os.path.basename(os.path.abspath(os.path.join(x, '../../'))))
    
    # wrong case 
    df_incorrect = df[(df.pred_per_article==0) & (df.category==cat)]
    
    # concat 
    concat_list = [df_incorrect['filename']]
    
    # add criterion
    df_incorrect['cnt'] = df_incorrect.apply(lambda x: len(eval(x.y_pred)), axis=1)
    df_incorrect['cnt_wrong'] = df_incorrect.apply(
        lambda x: np.sum(np.array(eval(x.y_true)) != np.array(eval(x.y_pred))), axis=1
    )
    df_incorrect['ratio_wrong'] = df_incorrect.apply(lambda x: x.cnt_wrong / x.cnt, axis=1)
    df_incorrect['max_wrong_score'] = df_incorrect.apply(lambda x: get_max_score(x.y_true, x.y_pred, x.y_score), axis=1)
    
    # extend
    concat_list.extend([
        df_incorrect['cnt'], 
        df_incorrect['cnt_wrong'], 
        df_incorrect['ratio_wrong'], 
        df_incorrect['max_wrong_score']
    ])
    
    # select top N
    wrong_case = pd.concat(concat_list,axis=1).sort_values(sort_target,ascending=False).head(n)

    return wrong_case