# Task1 

제목 내용 일치성

# Directory Tree

```
task1_title
.
├── configs
│   ├── BERT
│   │   ├── BERT_save_dataloader.yaml
│   │   ├── BERT-test.yaml
│   │   └── BERT-train.yaml
│   ├── FNDNet
│   └── HAN
├── dataset
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── bert.py
│   ├── factory.py
│   ├── fndnet.py
│   ├── han.py
│   └── tokenizer.py
├── demo
│   ├── example.md
│   ├── inference.ipynb
│   └── README.md
├── models
│   ├── __init__.py
│   ├── bert.py
│   ├── factory.py
│   ├── fndnet.py
│   ├── han.py
│   └── registry.py
├── saved_model
│   ├── BERT
│   │   ├── best_model.pt
│   │   ├── best_score.json
│   │   ├── exp_results_test.csv
│   │   ├── exp_results_train.csv
│   │   ├── exp_results_validation.csv
│   │   └── latest_model.pt
│   ├── FNDNet
│   └── HAN
├── log.py
├── main.py
├── save_dataloader.py
├── train.py
├── utils.py
├── task1 - result analysis.ipynb
└──README.md

```

# Models

- HAN[^1] [description]()
- FNDNet[^2] [description]()
- BERT[^3] [description]()


# Configurations

학습을 위해 사용되는 arguments 구성

**BERT**

- `./configs/BERT/BERT_save_dataloader.yaml`
- `./configs/BERT/BERT-train.yaml`
- `./configs/BERT/BERT-test.yaml`


**FNDNet**

- `./configs/FNDNet/FNDNet_save_dataloader.yaml`
- `./configs/FNDNet/FNDNet-train.yaml`
- `./configs/FNDNet/FNDNet-test.yaml`


**HAN**

- `./configs/HAN/HAN_save_dataloader.yaml`
- `./configs/HAN/HAN-train.yaml`
- `./configs/HAN/HAN-test.yaml`



# Run

# Save Datasets

학습 시 전처리 시간을 줄이기 위해 datasets을 사전에 저장하여 사용

yaml에서 data_path 또는 data_info_path를 확인 후 실행

**ex)** `./configs/BERT/BERT_save_dataloader.yaml`

```yaml
EXP_NAME: BERT
SEED: 223
    
DATASET:
    name: BERT
    data_path: ../data/labeled_fake_news/Part1 # news article directory
    saved_data_path: false
    PARAMETERS:
        max_word_len: 512

TOKENIZER:
    name: bert

TRAIN:
    batch_size: 256
    num_workers: 12

RESULT:
    savedir: ../data/labeled_fake_news/Part1
    dataname: 'BERT_w512'
```


**run**

```bash
python main.py --yaml_config ${config_file_path}
python save_dataloader.py --yaml_config ${config_file_path}
```


# Reference

[^1]: Jeong, H. (2021). Hierarchical Attention Networks for Fake News Detection (Doctoral dissertation, The Florida State University).
[^2]: Kaliyar, R. K., Goswami, A., Narang, P., & Sinha, S. (2020). FNDNet–a deep convolutional neural network for fake news detection. Cognitive Systems Research, 61, 32-44.
[^3]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT (1), 4171-4186