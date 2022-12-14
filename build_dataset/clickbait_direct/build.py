from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random

from tqdm.auto import tqdm
from methods import get_similar_filepath_dict, extract_nouns, extract_text

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



def update_label_info(file: dict, new_title: str) -> dict:
    '''
    update label information in dictionary 
    '''

    file['labeledDataInfo'] = {
        'newTitle': new_title,
        'clickbaitClass': 0,
        'referSentenceInfo': [
            {'sentenceNo':i+1, 'referSentenceyn': 'N'} for i in range(len(file['sourceDataInfo']['sentenceInfo']))
        ]
    }
    
    return file


def make_fake_title(file_list: list, save_list: list, cfg_method: dict, sim_filepath_dict: dict = None) -> None:
    '''
    make fake title using selected method
    '''

    for file_path, save_path in tqdm(zip(file_list, save_list), total=len(file_list)):

        # source file name and category
        category_name = os.path.basename(os.path.dirname(file_path))

        # load source file
        source_file = json.load(open(file_path, 'r'))
        
        # extract fake title
        if cfg_method['select_name'] == 'random_select':
            kwargs = {
                'file_path' : file_path,
                'file_list' : file_list
            }
        elif cfg_method['select_name'] == 'random_category_select':
            kwargs = {
                'file_path' : file_path,
                'category'  : category_name,
                'file_list' : file_list
            }
        elif cfg_method['name'] in ['tfidf','bow','ngram','sentence_embedding']:
            kwargs = {
                'sim_filepath' : sim_filepath_dict[category_name][file_path]
            }

        fake_title = __import__('methods').__dict__[cfg_method['select_name']](**kwargs)

        # update label infomation
        source_file = update_label_info(file=source_file, new_title=fake_title)
        
        # save source file
        json.dump(
            obj          = source_file, 
            fp           = open(save_path, 'w', encoding='utf-8'), 
            indent       = '\t',
            ensure_ascii = False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # set seed
    torch_seed(cfg['SEED'])

    # update save directory
    cfg['savedir'] = os.path.join(cfg['savedir'], cfg['METHOD']['select_name'])

    # load file list
    file_list = glob(os.path.join(cfg['datadir'], '[!sample]*/Clickbait_Auto/*/*'))
    save_list = [p.replace(cfg['datadir'], cfg['savedir']) for p in file_list]

    # make directory to save files
    parition_path = glob(os.path.join(cfg['datadir'], '[!sample]*/Clickbait_Auto/*'))
    parition_path = [p.replace(cfg['datadir'], cfg['savedir']) for p in parition_path]
    for path in parition_path:
        os.makedirs(path, exist_ok=True)    

    # find article index most similar to article and save indices
    sim_filepath_dict = None
    if cfg['METHOD']['name'] != 'random':
        sim_filepath_dict = get_similar_filepath_dict(
            make_sim_matrix_func = __import__('methods').__dict__[f"{cfg['METHOD']['name']}_sim_matrix"],
            extract_text_func    = extract_text if cfg['METHOD'] == 'sentence_embedding' else extract_nouns,
            file_list            = file_list,
            category_list        = os.listdir(os.path.join(cfg['savedir'],'train/Clickbait_Auto')),
            target               = cfg['METHOD']['target'],
            savedir              = cfg['savedir']
        )

    # run
    make_fake_title(
        file_list      = file_list, 
        save_list      = save_list, 
        cfg_method     = cfg['METHOD'],
        sim_filepath_dict = sim_filepath_dict
    )
