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
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models, util


def clean_news(content):
    pattern = re.compile('[\n\\\\"]')
    cleaned_content = re.sub(pattern, "", content)
    return cleaned_content


# ------------------- Ver.1-----------------------------------#

# def set_model(model_name: str, pooling_mode: str, device: int) -> SentenceTransformer:
#     # Backbone setting
#     word_embedding_model = models.Transformer(model_name)
#     # Pooling setting
#     pooling_model = models.Pooling(
#         word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode
#     )
#     # Aggregation
#     model = SentenceTransformer(
#         modules=[word_embedding_model, pooling_model], device=device
#     )

#     return model


# def get_argmax_columns(
#     cand_list: list,
#     argmax_columns_path: str,
#     model_name: str,
#     device_num: int,
#     pooling_mode: str,
# ) -> np.array:

#     device = torch.device(device_num if torch.cuda.is_available() else "cpu")
#     # Model setting
#     model = set_model(model_name, pooling_mode, device)

#     # Get embeddings of titles in numpy array
#     with torch.no_grad():

#         embedding_list = model.encode_multi_process(
#             cand_list,
#             model.start_multi_process_pool(),
#         )
#         embedding_list = torch.from_numpy(embedding_list).to(device)
#         # embedding_list = model.encode(
#         #     cand_list,
#         #     batch_size=4,
#         #     show_progress_bar=True,
#         #     convert_to_tensor=True,
#         #     device=device,
#         # )
#     length = len(cand_list)
#     sim_matrix = torch.zeros((length, length), device=device)
#     for i in tqdm(range(length), desc="Get Sim-list"):
#         similarity = F.cosine_similarity(embedding_list[i], embedding_list, dim=-1)
#         sim_matrix[i] += similarity
#         sim_matrix[i][i] = 0

#     sim_matrix = sim_matrix.cpu().numpy()
#     argmax_columns = np.argmax(sim_matrix, axis=1)
#     pickle.dump(obj=argmax_columns, file=open(argmax_columns_path, "wb"))

#     return argmax_columns


# ----------------------ver.2----------------------------#


def set_model_tokenizer(model_name: str, device: int):
    # Model and Tokenizer
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    token_embeddings[
        input_mask_expanded == 0
    ] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embedding_list(
    pooling_mode: str, model_output: torch.tensor, encoded_input: torch.tensor
):
    pooling_dict = {"mean": mean_pooling, "cls": cls_pooling, "max": max_pooling}
    pooling = pooling_dict[pooling_mode]
    embedding_list = pooling(model_output, encoded_input["attention_mask"])

    return embedding_list


def get_argmax_columns(
    cand_list: list,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    pooling_mode: str,
) -> np.array:

    # Device setting
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")

    # Model setting
    model, tokenizer = set_model_tokenizer(model_name, device)

    # Tokenize texts
    encoded_input = tokenizer(
        cand_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    # Get embeddings of titles in numpy array
    with torch.no_grad():
        model_output = model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        embedding_list = get_embedding_list(pooling_mode, model_output, encoded_input)

    pdb.set_trace()
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
