import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import kss


# ========================
# select
# ========================

def sentence_embedding_category_select(sim_filepath: str) -> str:
    """
    select news title among file list using sentence_embedding similarity
    """
    # target file
    target_file = json.load(open(sim_filepath, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title

def sentence_embedding_title_category_select(sim_filepath: str) -> str:
    return sentence_embedding_category_select(sim_filepath=sim_filepath)


def sentence_embedding_content_category_select(sim_filepath: str) -> str:
    return sentence_embedding_category_select(sim_filepath=sim_filepath)


# ========================
# similarity
# ========================

def sentence_embedding_sim_matrix(text: list, target: str) -> np.ndarray:
    '''
    reference: https://github.com/jhgan00/ko-sentence-transformers
    '''
    # model setting
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    if target == 'title':
        embedding = get_sentence_embedding(model=model, text=text)
    elif target == 'content':
        embedding = []
        split_doc_list = split_sentences(text)
        for doc in tqdm(split_doc_list, total=len(split_doc_list), leave=False, desc='Extract Content Embedding'):
            embedding_i = get_sentence_embedding(model=model, text=doc).mean(dim=0)
            embedding.append(embedding_i)

        embedding = torch.stack(embedding)

    # get similarity
    length = len(embedding)
    sim_matrix = torch.zeros((length, length), dtype=torch.float16)
    for i in tqdm(range(length), desc="Similarity Matrix"):
        similarity = F.cosine_similarity(embedding[[i]], embedding)
        sim_matrix[i] += similarity

    sim_matrix = sim_matrix.numpy()

    return sim_matrix

def split_sentences(text):
    total = []
    for t in tqdm(text, total=len(text), leave=False, desc='Split Sentence'):
        total.append(sum(kss.split_sentences([t], num_workers=1),[]))

    return total

def get_sentence_embedding(model, text: list) -> torch.Tensor:
    # get embeddings of titles in numpy array
    with torch.no_grad():
        embedding = model.encode(
            sentences         = text,
            batch_size        = 16,
            show_progress_bar = True,
            convert_to_tensor = True,
            device            = 'cuda:0',
        )
    
    return embedding.cpu()
