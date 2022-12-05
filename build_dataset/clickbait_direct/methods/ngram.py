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
        morphs_extract_path = f'{morphs_extract_dir}/{category}_morphs_extracted.json'
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

    newsTitles = [item['newsTitle'] for item in morphs_extracted.values()]
    newsContents = [item['newsContent'] for item in morphs_extracted.values()]
    newsFile_paths = [item['newsFile_path'] for item in morphs_extracted.values()]

    title_sim_matrix = make_sim_matrix(newsTitles)
    title_sim_argmax = title_sim_matrix.argmax(axis=1)

    if not os.path.exists(f'{sim_argmax_dir}/sim_argmax.json'):
        sim_argmax = dict()
    else:
        sim_argmax = json.load(open(f'{sim_argmax_dir}/sim_argmax.json', 'r'))
    for file_path, ts_index in zip(newsFile_paths, title_sim_argmax):
        sim_argmax.setdefault(category, {})
        sim_argmax[category][file_path] = {'title': newsFile_paths[ts_index]}
        
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
            'newsTitle': eval(f"' '.join(okt.{morphs_type}(newsTitle))"),
            'newsContent': eval(f"' '.join(okt.{morphs_type}(newsContent))"),
            'newsFile_path': file_path
        }

    # save morphs
    with open(morphs_extract_path, 'w') as f:
        f.write(json.dumps(morphs_extracted))

    print('morphemes extracted')

    return morphs_extracted


def make_sim_matrix(texts: list) -> np.ndarray:
    sim_matrix = [[[] for i in range(len(texts))] for j in range(len(texts))]
    
    for idx1, text in enumerate(tqdm(texts)):
        sim_matrix[idx1][idx1] = 0
        for idx2 in range(idx1+1, len(texts)):
            score = round(get_ngram_score(text, texts[idx2], 2, 3), 4)
            sim_matrix[idx1][idx2] = score
            sim_matrix[idx2][idx1] = score

    return np.array(sim_matrix)


def get_ngram_score(text1: str, text2: str, n_min: int, n_max: int) -> float:
    n_gram_scores = []
    for n in range(n_min, n_max+1):
        n_gram_scores.append(diff_ngram(text1, text2, n))
    return sum(n_gram_scores) / 2


def diff_ngram(text1: str, text2: str, n: int) -> float:
    text1_ngram = ngram(text1, n)
    text2_ngram = ngram(text2, n)
    overlap_ngram = set(text1_ngram) & set(text2_ngram)
    try:
        return (2*len(overlap_ngram)) / (len(text1_ngram)+len(text2_ngram))
    except ZeroDivisionError:
        return 0
    
def ngram(text: str, n: int) -> list:
    ngrams = []
    text_len = len(text) - n + 1
    for i in range(text_len):
        text_n = text[i:i+n]
        ngrams.append(text_n)
    return ngrams