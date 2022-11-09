import numpy as np
import json

def random_select(file_path: str, file_list: list) -> str:
    """
    select randomly news title among file list except for source file
    """

    # define file indice
    file_idxs = np.arange(len(file_list))

    # select news index except for source file index
    select_idx = np.random.choice(np.delete(file_idxs, file_list.index(file_path)), size=1)[0]

    # target file
    target_file = json.load(open(file_list[select_idx], 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']
        
    return fake_title



def random_category_select(file_path: str, category: str, file_list: list) -> str:
    """
    select randomly news title among file list of same category as source file
    """

    # define file list of same category as source file
    file_list_cat = [f for f in file_list if category in f]

    # define file indice
    file_idxs = np.arange(len(file_list_cat))

    # select news index except for source file index
    select_idx = np.random.choice(np.delete(file_idxs, file_list_cat.index(file_path)), size=1)[0]

    # target file
    target_file = json.load(open(file_list_cat[select_idx], 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']
        
    return fake_title