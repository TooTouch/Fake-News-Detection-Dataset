import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models

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
    # model setting
    model = set_model()

    if target == 'title':
        embedding = get_sentence_embedding(model=model, text=text)
    elif target == 'content':
        embedding = []
        for doc in kss.split_sentences(text):
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

def set_model() -> SentenceTransformer:
    # backbone setting
    word_embedding_model = models.Transformer('Huffon/sentence-klue-roberta-base')
    # pooling setting
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean'
    )
    # aggregation
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device='cuda:0'
    )

    return model
