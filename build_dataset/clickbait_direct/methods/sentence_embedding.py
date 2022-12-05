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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, models, util

import kss


def sentence_embedding_title(
    file_list_cat: list,
    file_path: np.array,
    argmax_columns: np.array,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    batch_size: int = None,
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
        batch_size,
    )

    return fake_title


def sentence_embedding_contents(
    file_list_cat: list,
    file_path: np.array,
    argmax_columns: np.array,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    batch_size: int = 4,
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
        batch_size,
    )

    return fake_title


def get_fake_title(
    type: str,
    file_list_cat: list,
    file_path: np.array,
    argmax_columns: np.array,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    batch_size: int = None,
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

            argmax_columns = get_argmax_title_columns(
                cand_list,
                argmax_columns_path,
                model_name,
                device_num,
                pooling_mode,
                batch_size,
            )

        elif type == "contents":

            cand_list = [
                json.load(open(file_list_cat[idx], "r"))["sourceDataInfo"][
                    "newsContent"
                ]
                for idx in file_idxs
            ]
            pooling_mode = "mean"

            argmax_columns = get_argmax_contents_columns(
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


# ------------------- title-title -----------------------------------#


def get_argmax_title_columns(
    cand_list: list,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    pooling_mode: str,
    batch_size: int,
) -> np.array:

    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    # Model setting
    model = set_model(model_name, pooling_mode, device)

    # Get embeddings of titles in numpy array
    with torch.no_grad():

        embedding_list = model.encode(
            cand_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
        )

    argmax_columns = get_argmax_columns(embedding_list, device)
    pickle.dump(obj=argmax_columns, file=open(argmax_columns_path, "wb"))

    return argmax_columns


# -------------------- contents-contents ----------------------------#
def get_argmax_contents_columns(
    cand_list: list,
    argmax_columns_path: str,
    model_name: str,
    device_num: int,
    pooling_mode: str,
) -> np.array:

    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    # Model setting
    s_model, h_model, tokenizer, max_input_length = set_model_tokenizer(
        model_name, pooling_mode, device
    )

    # Get embeddings of titles in numpy array
    doc_embeddings_list = []
    for content in tqdm(cand_list, desc="In the cand_list"):
        # sentences = sentence_split(content)
        sentences = kss.split_sentences(
            content,
        )
        with torch.no_grad():
            if s_model.tokenize(sentences)["input_ids"].shape[1] < 512:
                output_embedding = s_model.encode(
                    sentences,
                    batch_size=128,
                    convert_to_tensor=False,
                )  # numpy 출력
                doc_embedding = output_embedding.mean(axis=0)

            else:
                encoded_inputs = tokenizer(
                    sentences,
                    padding="max_length",
                    truncation=True,
                    max_length=max_input_length - 2,
                    return_tensors="pt",
                )
                encoded_inputs = {
                    key: value.to(device) for key, value in encoded_inputs.items()
                }
                output = h_model(**encoded_inputs)
                doc_embedding = (
                    output.last_hidden_state.cpu().squeeze().numpy().mean(axis=0)
                )

        doc_embeddings_list.append(doc_embedding)
    embedding_list = torch.from_numpy(np.array(doc_embeddings_list)).to(device)
    argmax_columns = get_argmax_columns(embedding_list, device)
    pickle.dump(obj=argmax_columns, file=open(argmax_columns_path, "wb"))

    return argmax_columns


# ------utils------- #
def get_argmax_columns(embedding_list, device):
    length = len(embedding_list)
    sim_matrix = torch.zeros((length, length), device=device)
    for i in tqdm(range(length), desc="Get Sim-list"):
        similarity = F.cosine_similarity(embedding_list[i], embedding_list, dim=-1)
        sim_matrix[i] += similarity
        sim_matrix[i][i] = 0

    sim_matrix = sim_matrix.cpu().numpy()
    argmax_columns = np.argmax(sim_matrix, axis=1)

    return argmax_columns


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


def set_model_tokenizer(model_name: str, pooling_mode: str, device: int):

    # Model and Tokenizer
    s_model = set_model(model_name, pooling_mode, device)
    h_model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_input_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
    return s_model, h_model, tokenizer, max_input_length
