import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity





# ========================
# select
# ========================

def bow_category_select(sim_filepath: str) -> str:
    """
    select news title among file list using bow similarity
    """
    # target file
    target_file = json.load(open(sim_filepath, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title

def bow_title_category_select(sim_filepath: str) -> str:
    return bow_category_select(sim_filepath=sim_filepath)


def bow_content_category_select(sim_filepath: str) -> str:
    return bow_category_select(sim_filepath=sim_filepath)



# ========================
# similarity
# ========================

def bow_sim_matrix(text: list) -> np.ndarray:
    """
    make similarity matrix using bow similarity
    """
    bow_model = CountVectorizer().fit(text)
    bow_df = bow_model.transform(text).toarray()
    cos_sim = cosine_similarity(bow_df, bow_df)

    return cos_sim
