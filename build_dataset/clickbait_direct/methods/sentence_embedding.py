# !pip install sentence-transformers
import pdb
import re
import json
import pickle
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models, util


def set_model(model_name: str, pooling_mode: str, device: int) -> SentenceTransformer:
    # Backbone setting
    word_embedding_model = models.Transformer(model_name)
    # Pooling setting
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode
    )
    # Aggregation
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device=device
    )

    return model


def clean_news(content):
    pattern = re.compile('[\n\\\\"]')
    cleaned_content = re.sub(pattern, "", content)
    return cleaned_content


def get_argmax_columns(
    cand_list: list,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    pooling_mode: str,
) -> np.array:

    # print(max([len(c) for c in cand_list]))
    # pdb.set_trace()
    # Device setting
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")

    # Model setting
    model = set_model(model_name, pooling_mode, device)
    print(len(model.tokenizer(cand_list[0])["input_ids"]))
    pdb.set_trace()
    print(model.tokenizer(cand_list[1]))

    # Get embeddings of titles in numpy array
    with torch.no_grad():
        embedding_list = model.encode(
            cand_list,
            batch_size=4,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
        )
    length = len(cand_list)
    sim_matrix = torch.zeros((length, length), device=device)
    for i in tqdm(range(length), desc="Get Sim-list"):
        similarity = F.cosine_similarity(embedding_list[i], embedding_list, dim=-1)
        sim_matrix[i] += similarity
        sim_matrix[i][i] = 0

    sim_matrix = sim_matrix.cpu().numpy()
    argmax_columns = np.argmax(sim_matrix, axis=1)
    pickle.dump(obj=argmax_columns, file=open(argmax_columns_path, "wb"))

    return argmax_columns


def get_fake_title(
    type: str,
    file_list_cat: list,
    file_path: np.array,
    argmax_columns: np.array,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
) -> str:

    """
    i) Calculate the similarity between given file and the others in the same category
    ii) Switch the original title with the most similar title among another titles
    """

    # Get query index
    query_idx = file_list_cat.index(file_path)

    # argmax columns pickle file check
    if argmax_columns is None:
        # file idx in file list category
        file_idxs = np.arange(len(file_list_cat))

        # candidate of titles
        if type == "title":
            cand_list = [
                json.load(open(file_list_cat[idx], "r"))["sourceDataInfo"]["newsTitle"]
                for idx in file_idxs
            ]
            pooling_mode = "mean"
        elif type == "contents":
            cand_list = [
                clean_news(
                    json.load(open(file_list_cat[idx], "r"))["sourceDataInfo"][
                        "newsContent"
                    ]
                )
                for idx in file_idxs
            ]
            # print(cand_list[0])
            pooling_mode = "cls"
        argmax_columns = get_argmax_columns(
            cand_list,
            argmax_columns_path,
            model_name,
            device_num,
            pooling_mode,
        )

    # Get the index of the most similar title
    fake_title = json.load(open(file_list_cat[argmax_columns[query_idx]], "r"))[
        "sourceDataInfo"
    ]["newsTitle"]

    return fake_title


def sentence_embedding_title(
    file_list_cat: list,
    file_path: np.array,
    argmax_columns: np.array,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
) -> str:

    """
    i) Calculate the similarity of titles between given file and the others in the same category
    ii) Switch the original title with the most similar title among another titles
    """
    type = "title"
    fake_title = get_fake_title(
        type,
        file_list_cat,
        file_path,
        argmax_columns,
        argmax_columns_path,
        model_name,
        device_num,
    )

    return fake_title


def sentence_embedding_contents(
    file_list_cat: list,
    file_path: np.array,
    argmax_columns: np.array,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
) -> str:

    """
    i) Calculate the similarity of contents embedding vectors between given file and the others in the same category
    ii) Switch the original title with the most similar contents among another titles
    """
    type = "contents"
    fake_title = get_fake_title(
        type,
        file_list_cat,
        file_path,
        argmax_columns,
        argmax_columns_path,
        model_name,
        device_num,
    )

    return fake_title
