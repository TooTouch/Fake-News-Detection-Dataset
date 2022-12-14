import numpy as np
import json

from tqdm.auto import tqdm

# ========================
# select
# ========================

def ngram_category_select(sim_filepath: str) -> str:
    """
    select news title among file list using ngram similarity
    """
    # target file
    target_file = json.load(open(sim_filepath, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title

def ngram_title_category_select(sim_filepath: str) -> str:
    return ngram_category_select(sim_filepath=sim_filepath)


def ngram_content_category_select(sim_filepath: str) -> str:
    return ngram_category_select(sim_filepath=sim_filepath)

            

# ========================
# similarity
# ========================


def ngram_sim_matrix(text: list) -> np.ndarray:
    ngram_set = [make_ngram_set(text, n_list=[2,3]) for text in tqdm(text, total=len(text), desc='N-gram set', leave=False)]

    nb_set = len(ngram_set)
    sim_matrix = np.zeros((nb_set, nb_set), dtype=np.float16)
    
    for idx1 in tqdm(range(nb_set), desc='Similarity Matrix', leave=False):
        for idx2 in range(idx1+1, nb_set):
            score = get_ngram_score(
                ngram_set1 = ngram_set[idx1], 
                ngram_set2 = ngram_set[idx2]
            )
            sim_matrix[idx1][idx2] = score
            sim_matrix[idx2][idx1] = score

    return sim_matrix


def get_ngram_score(ngram_set1: list, ngram_set2: list) -> float:
    n_gram_scores = [diff_ngram(ngram1, ngram2) for ngram1, ngram2 in zip(ngram_set1, ngram_set2)]
        
    return sum(n_gram_scores) / len(ngram_set1)


def diff_ngram(ngram1: list, ngram2: list) -> float:
    overlap_ngram = ngram1 & ngram2

    try:
        return (2*len(overlap_ngram)) / (len(ngram1)+len(ngram2))
    except:
        return 0

def make_ngram(text: str, n: int) -> list:
    text_len = len(text) - n + 1
    ngrams = [tuple(text[i:i+n]) for i in range(text_len)]
    
    return ngrams

def make_ngram_set(text: str, n_list: list) -> list:
    return [set(make_ngram(text, n)) for n in n_list]
