# Task1 

- **task**: 제목과 내용 일치성

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

# Reference

[^1]: 전재민, 최우용, 최수정, & 박세영. (2019). BTS: 한국어 BERT 를 사용한 텍스트 세그멘테이션. 한국정보과학회 학술발표논문집, 413-415. [ [paper](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE09301605&mark=0&useDate=&ipRange=N&accessgl=Y&language=ko_KR&hasTopBanner=true) ] 
[^2]: 소규성, 이윤승, 정의석, & 강필성. (2022). KoBERTSEG: 한국어 BERT 를 이용한 Local Context 기반 주제 분리 방법론. 대한산업공학회지, 48(2), 235-248. [ [paper](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11056567&googleIPSandBox=false&mark=0&useDate=&ipRange=false&accessgl=Y&language=ko_KR&hasTopBanner=true) ]



