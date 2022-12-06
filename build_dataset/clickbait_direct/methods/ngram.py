import numpy as np
import pandas as pd
import json
import os
from konlpy.tag import Okt
from tqdm.auto import tqdm


def ngram_title_category_select(file_path: str, sim_argmax: dict) -> str:
    """
    select news title among file list using ngram similarity
    """
    # select news title
    similar_newsFile_path = sim_argmax[file_path]['title']

    # target file
    target_file = json.load(open(similar_newsFile_path, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title


def sim_preprocess(category_list: list,
                    file_list: list, 
                    morphs_extract_dir: str, 
                    morphs_type: str='nouns', 
                    sim_argmax_dir: str
                    ) -> None:            
    for category in tqdm(category_list):
        file_list_cat = [f for f in file_list if category in f]
        os.makedirs(sim_argmax_dir, exist_ok=True)
        morphs_extract_path = os.path.join(morphs_extract_dir,f'{category}_morphs_extracted.json')
        save_sim_argmax(file_list_cat,
                        category,
                        sim_argmax_dir,
                        morphs_extract_path,
                        morphs_type)


def save_sim_argmax(file_list: list,
                    category: str,
                    sim_argmax_dir: str,
                    morphs_extract_path: str,
                    morphs_type: str
                    ) -> None:

    if not os.path.exists(morphs_extract_path):
        print(f"morphs_extract_path {morphs_extract_path.split('/')[-1]} not found")
        os.makedirs(os.path.dirname(morphs_extract_path), exist_ok=True)
        morphs_extracted = morphs_extract(file_list, morphs_extract_path, morphs_type=morphs_type)
    else:
        print(f"morphs_extract_path {morphs_extract_path.split('/')[-1]} found")
        morphs_extracted = json.load(open(morphs_extract_path, 'r'))

    # extract titles, contents, and file path
    newsTitles = [item['newsTitle'] for item in morphs_extracted.values()]
    newsContents = [item['newsContent'] for item in morphs_extracted.values()]
    newsFile_paths = [item['newsFile_path'] for item in morphs_extracted.values()]

    # ngram setting
    ngram_set_title = [ngram_set(text, n_list=[2,3]) for text in tqdm(newsTitles)]
    ngram_set_content = [ngram_set(text, n_list=[2,3]) for text in tqdm(newsContents)]

    # make similarity matrix
    if not os.path.exists(f'{sim_argmax_dir}/sim_argmax.json'):
        sim_argmax = dict()
        sim_argmax.setdefault(category, {})
    else:
        sim_argmax = json.load(open(f'{sim_argmax_dir}/sim_argmax.json', 'r'))

    ## title
    title_sim_matrix = make_sim_matrix(ngram_set_title)
    title_sim_argmax = title_sim_matrix.argmax(axis=1)

    for file_path, ts_index in zip(newsFile_paths, title_sim_argmax):
        sim_argmax[category].setdefault(file_path, {})
        sim_argmax[category][file_path]['title'] = newsFile_paths[ts_index]

    json.dump(sim_argmax, open(f'{sim_argmax_dir}/sim_argmax.json', 'w'), indent=4)

    ## content
    content_sim_matrix = make_sim_matrix(ngram_set_content)
    content_sim_argmax = content_sim_matrix.argmax(axis=1)

    for file_path, cs_index in zip(newsFile_paths, content_sim_argmax):
        sim_argmax[category][file_path]['content'] = newsFile_paths[cs_index]

    json.dump(sim_argmax, open(f'{sim_argmax_dir}/sim_argmax.json', 'w'), indent=4)
    
            

def morphs_extract(file_list: list, morphs_extract_path: str, morphs_type: str) -> dict:
    """
    extract morphs from news title
    """
    morphs_extracted = dict()

    print('extracting morphemes...')

    for file_path in tqdm(file_list):
        # load source file
        source_file = json.load(open(file_path, "r"))

        newsID = source_file['sourceDataInfo']['newsID']
        newsTitle = source_file['sourceDataInfo']['newsTitle']
        newsContent = source_file['sourceDataInfo']['newsContent']

        # extract morphs
        okt = Okt()
        morphs_extracted[newsID] = {
            'newsTitle': eval(f"okt.{morphs_type}(newsTitle)"),
            'newsContent': eval(f"okt.{morphs_type}(newsContent)"),
            'newsFile_path': file_path
        }

    # save morphs
    with open(morphs_extract_path, 'w') as f:
        f.write(json.dumps(morphs_extracted))

    print('morphemes extracted')

    return morphs_extracted


def make_sim_matrix(ngram_set: list) -> np.ndarray:
    nb_set = len(ngram_set)
    sim_matrix = np.zeros((nb_set, nb_set), dtype=np.float16)
    
    for idx1 in tqdm(range(nb_set)):
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

    return (2*len(overlap_ngram)) / (len(ngram1)+len(ngram2))

def ngram(text: str, n: int) -> list:
    text_len = len(text) - n + 1
    ngrams = [tuple(text[i:i+n]) for i in range(text_len)]
    
    return ngrams

def ngram_set(text: str, n_list: list) -> list:
    return [set(ngram(text, n)) for n in n_list]