# Task2

- **task**: 본문 내 일관성 탐지

# Directory Tree

```
task2_context
.
├── configs
│   ├── BERT
│   │   ├── BERT-test.yaml
│   │   └── BERT-train.yaml
│   └── KoBERTSeg
├── dataset
│   ├── __init__.py
│   ├── bts.py
│   ├── build_dataset.py
│   ├── factory.py
│   └── kobertseg.py
├── demo
│   ├── demo.gif
│   ├── example.md
│   ├── inference.ipynb
│   └── README.md
├── models
│   ├── __init__.py
│   ├── bts.py
│   ├── factory.py
│   ├── kobertseg.py
│   ├── registry.py
│   └── utils.py
├── saved_model
│   ├── BTS
│   │   ├── best_model.pt
│   │   ├── best_score.json
│   │   ├── exp_results_train.csv
│   │   ├── exp_results_validation.csv
│   │   ├── latest_model.pt
│   │   ├── results_test.json
│   │   ├── results_train.json
│   │   └── results_validation.json
│   └── KoBERTSeg
├── log.py
├── main.py
├── run.sh
├── task2 - result analysis.ipynb
├── train.py
├── utils.py
└── README.md

```

# Datasets

<details>
<summary><strong>Train dataset (295,275건)</strong></summary>
<div markdown="1">

**Target**

| target            |   count |
|:------------------|--------:|
| Clickbait_Auto    |  109819 |
| Clickbait_Direct  |   40109 |
| NonClickbait_Auto |  145347 |

**Category**

낚시-직접생성(Clickbait_Direct)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |   15271 | 13.91%  |
| ET         |   14098 | 12.84%  |
| GB         |   17660 | 16.08%  |
| IS         |   15276 | 13.91%  |
| LC         |   10627 | 9.68%   |
| PO         |   14803 | 13.48%  |
| SO         |   22084 | 20.11%  |

낚시-자동생성(Clickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    5143 | 12.82%  |
| ET         |    4661 | 11.62%  |
| GB         |    6697 | 16.70%  |
| IS         |    5508 | 13.73%  |
| LC         |    3756 | 9.36%   |
| PO         |    5580 | 13.91%  |
| SO         |    8764 | 21.85%  |

비낚시-자동생성(NonClickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |   20668 | 14.22%  |
| ET         |   17548 | 12.07%  |
| GB         |   21381 | 14.71%  |
| IS         |   19900 | 13.69%  |
| LC         |   17508 | 12.05%  |
| PO         |   20058 | 13.80%  |
| SO         |   28284 | 19.46%  |

</div>
</details>

<details>
<summary><strong>Validation dataset (36,910건)</strong></summary>
<div markdown="1">

**Target**

| target            |   count |
|:------------------|--------:|
| Clickbait_Auto    |   13726 |
| Clickbait_Direct  |    5015 |
| NonClickbait_Auto |   18169 |

**Category**

낚시-직접생성(Clickbait_Direct)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    1909 | 13.91%  |
| ET         |    1762 | 12.84%  |
| GB         |    2207 | 16.08%  |
| IS         |    1909 | 13.91%  |
| LC         |    1328 | 9.68%   |
| PO         |    1850 | 13.48%  |
| SO         |    2761 | 20.12%  |

낚시-자동생성(Clickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |     643 | 12.82%  |
| ET         |     583 | 11.63%  |
| GB         |     837 | 16.69%  |
| IS         |     689 | 13.74%  |
| LC         |     469 | 9.35%   |
| PO         |     698 | 13.92%  |
| SO         |    1096 | 21.85%  |

비낚시-자동생성(NonClickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    2584 | 14.22%  |
| ET         |    2194 | 12.08%  |
| GB         |    2673 | 14.71%  |
| IS         |    2487 | 13.69%  |
| LC         |    2189 | 12.05%  |
| PO         |    2507 | 13.80%  |
| SO         |    3535 | 19.46%  |

</div>
</details>


<details>
<summary><strong>Test dataset (36,909건)</strong></summary>
<div markdown="1">

**Target**

| target            |   count |
|:------------------|--------:|
| Clickbait_Auto    |   13726 |
| Clickbait_Direct  |    5015 |
| NonClickbait_Auto |   18168 |

**Category**

낚시-직접생성(Clickbait_Direct)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    1909 | 13.91%  |
| ET         |    1762 | 12.84%  |
| GB         |    2207 | 16.08%  |
| IS         |    1909 | 13.91%  |
| LC         |    1328 | 9.68%   |
| PO         |    1850 | 13.48%  |
| SO         |    2761 | 20.12%  |

낚시-자동생성(Clickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |     643 | 12.82%  |
| ET         |     583 | 11.63%  |
| GB         |     837 | 16.69%  |
| IS         |     689 | 13.74%  |
| LC         |     470 | 9.37%   |
| PO         |     698 | 13.92%  |
| SO         |    1095 | 21.83%  |

비낚시-자동생성(NonClickbait_Auto)

| category   |   count | ratio   |
|:-----------|--------:|:--------|
| EC         |    2583 | 14.22%  |
| ET         |    2194 | 12.08%  |
| GB         |    2673 | 14.71%  |
| IS         |    2487 | 13.69%  |
| LC         |    2189 | 12.05%  |
| PO         |    2507 | 13.80%  |
| SO         |    3535 | 19.46%  |

</div>
</details>


# Models

- BERT(BTS)[^1] [ [description](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/assets/model_description/BTS.md) | [download](https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part2/BTS.zip) ]
- KoBERTSeg[^2] [ [description](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/assets/model_description/KoBERTSeg.md) | [download](https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part2/KoBERTSeg.zip) ]

# Configurations

학습을 위해 사용되는 arguments 구성

**BERT(BTS)**

- `./configs/BERT/BERT-train.yaml`
- `./configs/BERT/BERT-test.yaml`

**KoBERTSeg**

- `./configs/KoBERTSeg/KoBERTSeg-train.yaml`
- `./configs/KoBERTSeg/KoBERTSeg-test.yaml`

# Run

```bash
python main.py --yaml_config ./configs/${modelname}/${modelname}-train.yaml
```

**fine-tuning**

Fine-tuning을 수행하는 경우 `configs` 내 모델 yaml 파일에서 `checkpoint_path`에 학습이 완료된 모델 저장 경로를 설정하여 학습 진행

ex) `./configs/BERT/BERT-train.yaml`

```yaml
MODEL:
    modelname: bts
    PARAMETERS:
        finetune_bert: True
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
    <img width="1208" alt="image" src="https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/assets/figures/part2_history.png?raw=true">
</p>

**2. Model Performance**

Metrics	| AUROC	| F1	| RECALL	| PRECISION	| ACC	| ACC_PER_ARTICLE
---|:---:|:---:|:---:|:---:|:---:|:---:
BERT(BTS)	| 0.989	| 0.908	| 0.899	| 0.917	| 0.986	| 0.839
KoBERTSeg	| 0.987	| 0.881	| 0.871	| 0.892	| 0.982 | 0.796


**3. Misclassification Case**

category	| KoBERTSeg wrong / total (%)	| BERT(BTS) wrong / total (%)
---|---:|---:
NonClickbait_Auto	| 1229 / 18168 (6.76%)	| 1317 / 18168 (7.25%)
Clickbait_Auto	| 4643 / 13726 (33.83%)	| 3786 / 13726 (27.58%)
Clickbait_Direct	| 1643 / 5015 (32.76%)	| 846 / 5015 (16.87%)

# Reference

[^1]: 전재민, 최우용, 최수정, & 박세영. (2019). BTS: 한국어 BERT 를 사용한 텍스트 세그멘테이션. 한국정보과학회 학술발표논문집, 413-415.
[^2]: 소규성, 이윤승, 정의석, & 강필성. (2022). KoBERTSEG: 한국어 BERT 를 이용한 Local Context 기반 주제 분리 방법론. 대한산업공학회지, 48(2), 235-248. 



