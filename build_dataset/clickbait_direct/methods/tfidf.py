import numpy as np
import json
import os
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


def tfidf_title_category_select(file_path: str,
                          sim_argmax: dict) -> str:
    """
    select news title among file list using tfidf similarity
    """
    # select news title
    similar_newsFile_path = sim_argmax[file_path]['title']

    # target file
    target_file = json.load(open(similar_newsFile_path, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title


def tfidf_content_category_select(file_path: str,
                          sim_argmax: dict) -> str:
    """
    select news title among file list using tfidf similarity
    """
    # select news title
    similar_newsFile_path = sim_argmax[file_path]['content']

    # target file
    target_file = json.load(open(similar_newsFile_path, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title


def sim_preprocess(category_list, file_list, morphs_extract_dir, morphs_type, matrix_dir):
    for category in tqdm(category_list):

        file_list_cat = [f for f in file_list if category in f]

        # load similarity matrix
        os.makedirs(matrix_dir, exist_ok=True)
        morphs_extract_path = f'{morphs_extract_dir}/{category}_morphs_extracted.json'
        save_sim_matrix(file_list_cat,
                        category,
                        matrix_dir,
                        morphs_extract_path,
                        morphs_type)


def save_sim_matrix(file_list: list,
                    category: str,
                    matrix_dir: str,
                    morphs_extract_path: str,
                    morphs_type: str='nouns',
                    ) -> None:
    """
    save similarity matrix using tfidf similarity
    """

    if not os.path.exists(morphs_extract_path):
        print(f"morphs_extract_path {morphs_extract_path.split('/')[-1]} not found")
        os.makedirs(os.path.dirname(morphs_extract_path), exist_ok=True)
        morphs_extracted = morphs_extract(file_list, morphs_extract_path, morphs_type=morphs_type)
    else:
        print(f"morphs_extract_path {morphs_extract_path.split('/')[-1]} found")
        morphs_extracted = json.load(open(morphs_extract_path, 'r'))

    # define tfidf vectorizer
    newsTitles = [item['newsTitle'] for item in morphs_extracted.values()]
    newsContents = [item['newsContent'] for item in morphs_extracted.values()]
    newsFile_paths = [item['newsFile_path'] for item in morphs_extracted.values()]

    # make similarity matrix
    title_sim_matrix_path = f'{matrix_dir}/{category}_title_sim_matrix.npy'
    content_sim_matrix_path = f'{matrix_dir}/{category}_content_sim_matrix.npy'

    if not os.path.exists(title_sim_matrix_path):
        print('make title similarity matrix')
        title_sim_matrix = make_sim_matrix(newsTitles)
        print('make title similarity matrix done')
        np.save(title_sim_matrix_path, title_sim_matrix)
    else:
        print('title similarity matrix already exists')
        title_sim_matrix = np.load(title_sim_matrix_path)
    if not os.path.exists(content_sim_matrix_path):
        print('make content similarity matrix')
        content_sim_matrix = make_sim_matrix(newsContents)
        print('make content similarity matrix done')
        np.save(content_sim_matrix_path, content_sim_matrix)
    else:
        print('content similarity matrix already exists')
        content_sim_matrix = np.load(content_sim_matrix_path)

    title_sim_matrix[np.arange(title_sim_matrix.shape[0]), np.arange(title_sim_matrix.shape[0])] = -1
    content_sim_matrix[np.arange(content_sim_matrix.shape[0]), np.arange(content_sim_matrix.shape[0])] = -1
    title_sim_argmax = title_sim_matrix.argmax(axis=1)
    content_sim_argmax = content_sim_matrix.argmax(axis=1)

    if not os.path.exists(f'{matrix_dir}/sim_argmax.json'):
        sim_argmax = dict()
    else:
        sim_argmax = json.load(open(f'{matrix_dir}/sim_argmax.json', 'r'))
    for file_path, ts_index, cs_index in zip(newsFile_paths, title_sim_argmax, content_sim_argmax):
        sim_argmax.setdefault(category, {})
        sim_argmax[category][file_path] = {'title': newsFile_paths[ts_index],
                                           'content': newsFile_paths[cs_index]}
    json.dump(sim_argmax, open(f'{matrix_dir}/sim_argmax.json', 'w'), indent=4)


def make_sim_matrix(text: list) -> np.ndarray:
    """
    make similarity matrix using tfidf similarity
    """
    tf_idf_model = TfidfVectorizer().fit(text)
    tf_idf_df = tf_idf_model.transform(text).toarray()
    cos_sim = cosine_similarity(tf_idf_df, tf_idf_df)

    return cos_sim


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