# Task2

- **task**: 본문 내 일관성 탐지

# Directory Tree

```
task2_context
.
├── configs
│   ├── BTS
│   │   ├── BTS-test.yaml
│   │   └── BTS-train.yaml
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

# Models

- BTS[^1] [description]()
- KoBERTSeg[^2] [description]()

# Configurations

학습을 위해 사용되는 arguments 구성

**BTS**

- `./configs/BTS/BTS-train.yaml`
- `./configs/BTS/BTS-test.yaml`

**KoBERTSeg**

- `./configs/KoBERTSeg/KoBERTSeg-train.yaml`
- `./configs/KoBERTSeg/KoBERTSeg-test.yaml`

# Run

```bash
python main.py --yaml_config ${config_file_path}
```

# Results

**1. Model Performance**

Metrics	| AUROC	| F1	| RECALL	| PRECISION	| ACC	| ACC_PER_ARTICLE
---|:---:|:---:|:---:|:---:|:---:|:---:
BTS	| 0.989	| 0.908	| 0.899	| 0.917	| 0.986	| 0.839
KoBERTSeg	| 0.987	| 0.881	| 0.871	| 0.892	| 0.982 | 0.796


**2. Misclassification Case**

category	| KoBERTSeg wrong / total (%)	| BTS wrong / total (%)
---|---:|---:
NonClickbait_Auto	| 1229 / 18168 (6.76%)	| 1317 / 18168 (7.25%)
Clickbait_Auto	| 4643 / 13726 (33.83%)	| 3786 / 13726 (27.58%)
Clickbait_Direct	| 1643 / 5015 (32.76%)	| 846 / 5015 (16.87%)

# Reference

[^1]: 전재민, 최우용, 최수정, & 박세영. (2019). BTS: 한국어 BERT 를 사용한 텍스트 세그멘테이션. 한국정보과학회 학술발표논문집, 413-415.
[^2]: 소규성, 이윤승, 정의석, & 강필성. (2022). KoBERTSEG: 한국어 BERT 를 이용한 Local Context 기반 주제 분리 방법론. 대한산업공학회지, 48(2), 235-248. 



