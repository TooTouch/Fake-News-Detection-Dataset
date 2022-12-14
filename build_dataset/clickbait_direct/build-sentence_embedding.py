from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random
import pickle

from tqdm.auto import tqdm


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)


def update_label_info(file: dict, new_title: str) -> dict:
    """
    update label information in dictionary
    """

    file["labeledDataInfo"] = {
        "newTitle": new_title,
        "clickbaitClass": 0,
        "referSentenceInfo": [
            {"sentenceNo": i, "referSentenceyn": "N"}
            for i in range(len(file["sourceDataInfo"]["sentenceInfo"]))
        ],
    }

    return file


def make_fake_title(
    file_list: list, savedir: str, cfg_method: dict, category_list: list
) -> None:
    """
    make fake title using selected method
    """
    # Loading a list of files in each category just once
    file_list_cat_dict = {
        category: [f for f in file_list if category in f] for category in category_list
    }

    argmax_columns = None

    for file_path in tqdm(file_list):

        # source file name and category
        category_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)

        # load source file
        source_file = json.load(open(file_path, "r"))

        # extract fake title
        if cfg_method["name"] == "random_select":
            kwargs = {"file_path": file_path, "file_list": file_list}
        elif cfg_method["name"] == "random_category_select":
            kwargs = {
                "file_path": file_path,
                "category": category_name,
                "file_list": file_list,
            }

        elif "sentence_embedding" in cfg_method["name"]:

            # Check pickle file of argmax_columns
            argmax_columns_path = os.path.join(
                savedir, category_name, f"{cfg_method['name']}_argmax_columns.pkl"
            )
            if os.path.isfile(argmax_columns_path) and argmax_columns is None:
                argmax_columns = pickle.load(open(argmax_columns_path, "rb"))

            elif not os.path.isfile(argmax_columns_path):
                print(f"category:{category_name}")
                argmax_columns = None

            kwargs = {
                "file_list_cat": file_list_cat_dict[category_name],
                "file_path": file_path,
                "argmax_columns": argmax_columns,
                "argmax_columns_path": argmax_columns_path,
                "model_name": cfg_method["model_name"],
                "device_num": cfg_method["device_num"],
                "batch_size": cfg_method["batch_size"],
            }

        fake_title = __import__("methods").__dict__[cfg_method["name"]](**kwargs)

        # update label infomation
        source_file = update_label_info(file=source_file, new_title=fake_title)

        # save source file
        category_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        json.dump(
            obj=source_file,
            fp=open(
                os.path.join(savedir, category_name, file_name), "w", encoding="utf-8"
            ),
            indent="\t",
            ensure_ascii=False,
        )


def make_label(file_list: list, savedir: str) -> None:
    """
    make label for NonClickbait_Auto
    """
    for file_path in tqdm(file_list):
        # source file name and category
        category_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)

        # load source file
        source_file = json.load(open(file_path, "r"))

        # extract new title
        new_title = source_file["sourceDataInfo"]["newsTitle"]

        # update label information
        source_file = update_label_info(file=source_file, new_title=new_title)

        # save source file
        category_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        json.dump(
            obj=source_file,
            fp=open(
                os.path.join(savedir, category_name, file_name), "w", encoding="utf-8"
            ),
            indent="\t",
            ensure_ascii=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, help="config filename")
    args = parser.parse_args()

    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)

    # set seed
    torch_seed(cfg["SEED"])

    # make directory to save files
    if cfg["BUILD"].get("METHOD", False):
        cfg["BUILD"]["savedir"] = (
            cfg["BUILD"]["savedir"] + "_" + cfg["BUILD"]["METHOD"]["name"]
        )
    os.makedirs(cfg["BUILD"]["savedir"], exist_ok=True)

    category_list = os.listdir(cfg["BUILD"]["datadir"])
    for cat in category_list:
        os.makedirs(os.path.join(cfg["BUILD"]["savedir"], cat), exist_ok=True)

    # load file list
    file_list = glob(os.path.join(cfg["BUILD"]["datadir"], "*/*"))

    # run
    if cfg["BUILD"].get("METHOD", False):
        make_fake_title(
            file_list=file_list,
            savedir=cfg["BUILD"]["savedir"],
            cfg_method=cfg["BUILD"]["METHOD"],
            category_list=category_list,
        )
    else:
        make_label(file_list=file_list, savedir=cfg["BUILD"]["savedir"])
