import numpy as np
import os
import json

from tqdm.auto import tqdm
from konlpy.tag import Mecab
from typing import List




def get_similar_filepath_dict(make_sim_matrix_func, file_list: list, category_list: list, target: str, savedir: str) -> None:
    # define save path
    savepath = os.path.join(savedir, f'sim_index_{target}.json')

    # load sim_filepath_dict
    if not os.path.isfile(savepath):
        sim_filepath_dict = dict()
    else:
        sim_filepath_dict = json.load(open(savepath, 'r'))

    # define progress bar
    pbar = tqdm(category_list, total=len(category_list))

    for category in pbar:
        pbar.set_description(f'Category: {category}')

        # run if category not in sim_filepath_dict
        if not category in sim_filepath_dict.keys():
            # set default
            sim_filepath_dict.setdefault(category, {})

            # extract file path in category
            file_list_cat = [f for f in file_list if category in f]

            # load similarity matrix
            sim_filepath_dict = extract_sim_filepath(
                make_sim_matrix_func = make_sim_matrix_func,
                file_list            = file_list_cat,
                category             = category,
                target               = target,
                sim_filepath_dict    = sim_filepath_dict
            )

            # save sim_filepath_dict
            json.dump(sim_filepath_dict, open(savepath, 'w'), indent=4)

    return sim_filepath_dict


def extract_sim_filepath(make_sim_matrix_func, file_list: list, category: str, target: str, sim_filepath_dict: dict) -> None:
    """
    extract filepath most similar to filepath1 using ngram similarity
    """
    
    # extract nouns
    nouns_list = extract_nouns(file_list=file_list, target=target)

    # make similarity matrix
    sim_matrix = make_sim_matrix_func(text=nouns_list)
    sim_matrix[np.arange(sim_matrix.shape[0]), np.arange(sim_matrix.shape[0])] = -1
    
    # find argmax
    sim_index = sim_matrix.argmax(axis=1)

    # update sim_filepath_dict
    for file_path, idx in zip(file_list, sim_index):
        sim_filepath_dict[category][file_path] = file_list[idx]
    
    return sim_filepath_dict


def extract_nouns(file_list: list, target: str, join: bool = True) -> List[list]:
    """
    extract nouns from target text
    """
    # extract morphs
    mecab = Mecab()

    # define list
    nouns_list = []

    for file_path in tqdm(file_list, desc=f'Extract Morphs({target})', total=len(file_list), leave=False):
        # load source file
        source_file = json.load(open(file_path, "r"))

        if target == 'title':
            text = source_file['sourceDataInfo']['newsTitle']
        elif target == 'content':
            text = source_file['sourceDataInfo']['newsContent']

        if join:
            nouns_list.append(' '.join(mecab.nouns(text)))
        else:
            nouns_list.append(mecab.nouns(text))

    return nouns_list