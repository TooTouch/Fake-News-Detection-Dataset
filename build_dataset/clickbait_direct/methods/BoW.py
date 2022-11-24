import numpy as np
import pandas as pd
import json
import os
from konlpy.tag import Okt
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def bow_title_category_select(file_path: str,
                          sim_argmax: dict) -> str:
    """
    select news title among file list using BoW between news title
    """
    # select news title
    similar_newsFile_path = sim_argmax[file_path]['title']

    # target file
    target_file = json.load(open(similar_newsFile_path, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']
    # print(f"orgin_title: {json.load(open(file_path, 'r'))['sourceDataInfo']['newsTitle']}")
    # print(f'fake_title: {fake_title}')

    return fake_title


def bow_content_category_select(file_path: str,
                          sim_argmax: dict) -> str:
    """
    select news title among file list using BoW between news contents
    """
    # select news title
    similar_newsFile_path = sim_argmax[file_path]['content']

    # target file
    target_file = json.load(open(similar_newsFile_path, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']
    # print(f"orgin_title: {json.load(open(file_path, 'r'))['sourceDataInfo']['newsTitle']}")
    # print(f'fake_title: {fake_title}')

    return fake_title


def sim_preprocess(category_list, file_list, bow_dir, morphs_extract_dir, morphs_type):
    for category in tqdm(category_list):

        file_list_cat = [f for f in file_list if category in f]

        # load BoW matrix
        os.makedirs(bow_dir, exist_ok=True)
        
        morphs_extract_path = f'{morphs_extract_dir}/{category}_{morphs_type}_extracted.json'

        save_bow_sim_matrix(file_list_cat,
                        category,
                        bow_dir,
                        morphs_extract_path,
                        morphs_type)

def morphs_extract(file_list: list, morphs_extract_path: str, morphs_type: str='morphs') -> dict:
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

    # morphs_type={'morphs', 'nouns'}
    '''
        'morphs'인경우에도 굳이 [,/ 등 특수문자, 숫자를 제거할 필요는 없을 것 같다 어짜피 얘는 얼마안되니까
        없애야하는건, ., !, ? 등 자주 반복되는 문장부호만 없애면 될 것 같다.
    '''
    with open(f'{morphs_extract_path}', 'w') as f:
        f.write(json.dumps(morphs_extracted,ensure_ascii = False, indent = 4))

    print('morphemes extracted')

    return morphs_extracted

def count_words(news_TitleOrContent:list) -> dict:
    '''
    make bag of words
    news_TitleOrContent: list of news title:str or content:str
    '''
    word_to_index = {}
    bow_vector= []
    BoW={}
    for word in news_TitleOrContent:  
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)  
            # add default 1 to BoW
            bow_vector.insert(len(word_to_index) - 1, 1)
        else:
            # get index of word already exists in word_to_index
            index = word_to_index.get(word)
            # add 1 to BoW ofr word already exists in word_to_index
            bow_vector[index] = bow_vector[index] + 1
    for word in word_to_index.keys():
        BoW[word]=bow_vector[word_to_index[word]]
    return BoW

def make_bag_of_words(file_list: list, category: str, bow_dir: str, morphs_extract_path: str, morphs_type: str='morphs') -> dict:
    '''
    from extracted morphs, build bag of words
    '''
    # load morphs
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

    BoW_titles={}
    BoW_contents={}
    
    for i in tqdm(range(len(file_list))):
        newsID = list(morphs_extracted.keys())[i]
        newsTitle=newsTitles[i].split()
        newsContent=newsContents[i].split()
        newsFile_path=newsFile_paths[i]       
        
        BoW_titles[newsID] =count_words(newsTitle) #newsID 대신 newsFile_path를 넣어도 될 것 같다.
        BoW_contents[newsID] = count_words(newsContent)

    BoW_titles_total=pd.DataFrame(BoW_titles).T.fillna(0)
    BoW_contents_total=pd.DataFrame(BoW_contents).T.fillna(0)

    BoW_titles_total.to_csv(f'{bow_dir}/{category}_BoW_titles_total.csv')
    BoW_contents_total.to_csv(f'{bow_dir}/{category}_BoW_contents_total.csv')

    return BoW_titles_total, BoW_contents_total

def save_bow_sim_matrix(file_list: list, category: str, bow_dir: str, morphs_extract_path: str, morphs_type: str='morphs') -> dict:
    '''
    make & save BoW similarity matrix + save argmax of BoW similarity matrix
    '''

    # make & save BoW similarity matrix
    if not os.path.exists(f'{bow_dir}/{category}_BoW_titles_total.csv'):
        BoW_titles_total, BoW_contents_total = make_bag_of_words(file_list, category, bow_dir, morphs_extract_path, morphs_type=morphs_type)
    else:
        BoW_titles_total = pd.read_csv(f'{bow_dir}/{category}_BoW_titles_total.csv', index_col=0)
        BoW_contents_total = pd.read_csv(f'{bow_dir}/{category}_BoW_contents_total.csv', index_col=0)

    BoW_titles_total=BoW_titles_total.to_numpy()
    BoW_contents_total=BoW_contents_total.to_numpy()
    
    BoW_titles_sim_matrix = cosine_similarity(BoW_titles_total)
    BoW_contents_sim_matrix = cosine_similarity(BoW_contents_total)
    BoW_titles_sim_path=f'{bow_dir}/BoW_titles_{category}_{morphs_type}.npy'
    BoW_contents_sim_path=f'{bow_dir}/BoW_contents_{category}_{morphs_type}.npy'

    np.save(BoW_titles_sim_path, BoW_titles_sim_matrix)
    np.save(BoW_contents_sim_path, BoW_contents_sim_matrix)

    # save argmax of BoW similarity matrix
    BoW_titles_sim_matrix[np.arange(BoW_titles_sim_matrix.shape[0]), np.arange(BoW_titles_sim_matrix.shape[0])] = -1
    BoW_contents_sim_matrix[np.arange(BoW_contents_sim_matrix.shape[0]), np.arange(BoW_contents_sim_matrix.shape[0])] = -1
    title_sim_argmax = BoW_titles_sim_matrix.argmax(axis=1)
    content_sim_argmax = BoW_contents_sim_matrix.argmax(axis=1)


    morphs_extracted = json.load(open(morphs_extract_path, 'r'))
    newsFile_paths = [item['newsFile_path'] for item in morphs_extracted.values()]

    if not os.path.exists(f'{bow_dir}/sim_argmax.json'):
        sim_argmax = dict()
    else:
        sim_argmax = json.load(open(f'{bow_dir}/sim_argmax.json', 'r'))
    for file_path, ts_index, cs_index in zip(newsFile_paths, title_sim_argmax, content_sim_argmax):
        sim_argmax.setdefault(category, {})
        sim_argmax[category][file_path] = {'title': newsFile_paths[ts_index],
                                           'content': newsFile_paths[cs_index]}
    json.dump(sim_argmax, open(f'{bow_dir}/sim_argmax.json', 'w'), indent=4)
    
