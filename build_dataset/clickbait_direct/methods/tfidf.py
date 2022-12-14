import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# ========================
# select
# ========================

def tfidf_category_select(sim_filepath: str) -> str:
    """
    select news title among file list using tfidf similarity
    """
    # target file
    target_file = json.load(open(sim_filepath, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title

def tfidf_title_category_select(sim_filepath: str) -> str:
    return tfidf_category_select(sim_filepath=sim_filepath)


def tfidf_content_category_select(sim_filepath: str) -> str:
    return tfidf_category_select(sim_filepath=sim_filepath)



# ========================
# similarity
# ========================


def tfidf_sim_matrix(text: list, **kwargs) -> np.ndarray:
    """
    make similarity matrix using tfidf similarity
    """
    tf_idf_model = TfidfVectorizer().fit(text)
    tf_idf_df = tf_idf_model.transform(text).toarray()
    cos_sim = cosine_similarity(tf_idf_df, tf_idf_df)

    return cos_sim

