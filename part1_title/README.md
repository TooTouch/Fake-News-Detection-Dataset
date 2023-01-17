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
│   └── HAND
├── dataset
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── bert.py
│   ├── factory.py
│   ├── fndnet.py
│   ├── hand.py
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
│   ├── hand.py
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
│   └── HAND
├── log.py
├── main.py
├── save_dataloader.py
├── train.py
├── utils.py
├── task1 - result analysis.ipynb
└──README.md

```

**Korean word-embeddings**

- 한국어 임베딩 [ [github](https://github.com/ratsgo/embedding) ]
- word-embeddings [ [download](https://drive.google.com/file/d/1FeGIbSz2E1A63JZP_XIxnGaSRt7AhXFf/view) ]

# Datasets

<details>
<summary><strong>Train dataset (291,466건)</strong></summary>
<div markdown="1">

**Target**

| target                 |   count |
|:-----------------------|--------:|
| Clickbait_Direct       |   40106 |
| Clickbait_Auto         |  106014 |
| NonClickbait_Auto      |  145346 |

**Category**

낚시-직접생성(Clickbait_Direct)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    7007 | 17.47%  |
| ET         |    5419 | 13.51%  |
| GB         |    7141 | 17.81%  |
| IS         |    5502 | 13.72%  |
| LC         |    3477 | 8.67%   |
| PO         |    4571 | 11.40%  |
| SO         |    6989 | 17.43%  |

낚시-자동생성(Clickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |   14393 | 13.58%  |
| ET         |   13101 | 12.36%  |
| GB         |   16284 | 15.36%  |
| IS         |   14809 | 13.97%  |
| LC         |   10814 | 10.20%  |
| PO         |   14646 | 13.82%  |
| SO         |   21967 | 20.72%  |

비낚시-자동생성(NonClickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |   20664 | 14.22%  |
| ET         |   17501 | 12.04%  |
| GB         |   21552 | 14.83%  |
| IS         |   19982 | 13.75%  |
| LC         |   17350 | 11.94%  |
| PO         |   20162 | 13.87%  |
| SO         |   28135 | 19.36%  |

</div>
</details>

<details>
<summary><strong>Validation dataset (36,434건)</strong></summary>
<div markdown="1">

**Target**

| target                 |   count |
|:-----------------------|--------:|
| Clickbait_Direct |    5012 |
| Clickbait_Auto         |   13253 |
| NonClickbait_Auto      |   18169 |

**Category**

낚시-직접생성(Clickbait_Direct)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |     876 | 17.48%  |
| ET         |     677 | 13.51%  |
| GB         |     892 | 17.80%  |
| IS         |     688 | 13.73%  |
| LC         |     434 | 8.66%   |
| PO         |     571 | 11.39%  |
| SO         |     874 | 17.44%  |

낚시-자동생성(Clickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    1799 | 13.57%  |
| ET         |    1638 | 12.36%  |
| GB         |    2036 | 15.36%  |
| IS         |    1851 | 13.97%  |
| LC         |    1352 | 10.20%  |
| PO         |    1831 | 13.82%  |
| SO         |    2746 | 20.72%  |

비낚시-자동생성(NonClickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    2583 | 14.22%  |
| ET         |    2188 | 12.04%  |
| GB         |    2694 | 14.83%  |
| IS         |    2498 | 13.75%  |
| LC         |    2169 | 11.94%  |
| PO         |    2520 | 13.87%  |
| SO         |    3517 | 19.36%  |

</div>
</details>


<details>
<summary><strong>Test dataset (36,433건)</strong></summary>
<div markdown="1">

**Target**

| target                 |   count |
|:-----------------------|--------:|
| Clickbait_Direct |    5013 |
| Clickbait_Auto         |   13251 |
| NonClickbait_Auto      |   18169 |

**Category**

낚시-직접생성(Clickbait_Direct)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |     876 | 17.47%  |
| ET         |     677 | 13.50%  |
| GB         |     893 | 17.81%  |
| IS         |     688 | 13.72%  |
| LC         |     434 | 8.66%   |
| PO         |     571 | 11.39%  |
| SO         |     874 | 17.43%  |

낚시-자동생성(Clickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    1799 | 13.58%  |
| ET         |    1637 | 12.35%  |
| GB         |    2035 | 15.36%  |
| IS         |    1851 | 13.97%  |
| LC         |    1352 | 10.20%  |
| PO         |    1831 | 13.82%  |
| SO         |    2746 | 20.72%  |

비낚시-자동생성(NonClickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    2583 | 14.22%  |
| ET         |    2188 | 12.04%  |
| GB         |    2694 | 14.83%  |
| IS         |    2498 | 13.75%  |
| LC         |    2169 | 11.94%  |
| PO         |    2520 | 13.87%  |
| SO         |    3517 | 19.36%  |

</div>
</details>

# Models

- HAND[^1] [ [description](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/model_description/HAND.md) | [checkpoint](https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part1/HAND.zip) ]
- FNDNet[^2] [ [description](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/model_description/FNDNet.md) | [checkpoint](https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part1/FNDNet.zip) ]
- BERT[^3] [ [description](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/model_description/BERT.md) | [checkpoint](https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part1/BERT.zip) ]


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


**HAND**

- `./configs/HAND/HAND_save_dataloader.yaml`
- `./configs/HAND/HAND-train.yaml`
- `./configs/HAND/HAND-test.yaml`



# Run

# Save Datasets

학습 시 전처리 시간을 줄이기 위해 datasets을 사전에 저장하여 사용

**ex)** `./configs/BERT/BERT_save_dataloader.yaml`

```yaml
EXP_NAME: BERT
SEED: 223
    
DATASET:
    name: BERT
    data_path: ../data/Part1 # news article directory
    saved_data_path: false
    PARAMETERS:
        max_word_len: 512

TOKENIZER:
    name: bert

TRAIN:
    batch_size: 256
    num_workers: 12

RESULT:
    savedir: ../data/Part1
    dataname: 'BERT_w512'
```


**run**

```bash
python save_dataloader.py --yaml_config ./configs/${modelname}/${modelname}_save_dataloader.yaml
python main.py --yaml_config ./configs/${modelname}/${modelname}-train.yaml
```

**fine-tuning**

Fine-tuning을 수행하는 경우 `configs` 내 모델 yaml 파일에서 `checkpoint_path`에 학습이 완료된 모델 저장 경로를 설정하여 학습 진행

ex) `./configs/HAND/HAND-train.yaml`

```yaml
MODEL:
    modelname: hand
    freeze_word_embed: True
    use_pretrained_word_embed: True
    PARAMETERS:
        num_classes: 2
        vocab_len: 50002
        dropout: 0.1
        word_dims: 32
        sent_dims: 64
        embed_dims: 100
    CHECKPOINT:
        checkpoint_path: './저장된_모델_경로/모델이름.pt'
```

**test**

```bash
python main.py --yaml_config ./configs/${modelname}/${modelname}-test.yaml
```

# Results

**1. Training History**

<p align='center'>
    <img width="1098" alt="image" src="https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/figures/part1_history.png?raw=true">
</p>

**2. Model Performance**

Metrics	| AUROC	| F1	| RECALL	| PRECISION	| ACC
---|:---:|:---:|:---:|:---:|:---:
FNDNet	| 0.897	| 0.811	| 0.802	| 0.820	| 0.813
HAND	| 0.945	| 0.867	| 0.842	| 0.893	| 0.870
BERT	| 0.998	| 0.978	| 0.977	| 0.978	| 0.978


**3. Misclassification Case**

 category	| HAND wrong / total (%)	| BERT wrong / total (%)	| FNDNet wrong / total (%)
---|---:|---:|---:
NonClickbait_Auto	| 1841 / 18169 (10.13%)	| 399 / 18169 (2.20%)	| 3203 / 18169 (17.63%)
Clickbait_Auto	| 2161 / 13251 (16.31%)	| 51 / 13251 (0.38%)	| 2978 / 13251 (22.47%)
Clickbait_Direct	| 717 / 5013 (14.30%)	| 368 / 5013 (7.34%)	| 646 / 5013 (12.89%)

# Reference

[^1]: Jeong, H. (2021). Hierarchical Attention Networks for Fake News Detection (Doctoral dissertation, The Florida State University).
[^2]: Kaliyar, R. K., Goswami, A., Narang, P., & Sinha, S. (2020). FNDNet–a deep convolutional neural network for fake news detection. Cognitive Systems Research, 61, 32-44.
[^3]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT (1), 4171-4186