# Fake-News-Detection-Dataset

한국어 가짜 뉴스 탐지 데이터셋에 대한 baseline 실험 결과

<div align="center">

[📘 Documentation](https://tootouch.github.io/Fake-News-Detection-Dataset/) | [github](https://github.com/DSBA-Lab/Fake-News-Detection-Dataset)

</div>

# Enviroments

- python 3.6.10

```
torch==1.8.0a0+17f8c32
konlpy==0.6.0
einops
gluonnlp==0.10.0
wandb==0.12.18
transformers==4.18.0
git+https://git@github.com/SKTBrain/KoBERT.git@master
```


**Computer Resources**
- **CPU**: i7-9800X
- **GPU**: RTX 2080Ti x 2
- **RAM**: 64GB
- **SSD**: 2TB x 2
- **OS**: ubuntu 18.04

**1. docker image**

1. docker hub를 통해서 docker image pull 하는 방법

```bash
docker pull dsbalab/fake_news
```

2. Dockerfile을 통해서 docker image 설치 방법

docker image 생성 시 `word-embedding`와 Part1과 Part2에 대한 `checkpoints` 가 함께 생성

```bash
cd ./docker
docker build -t $image_name .
```


**2. Korean word-embeddings**

본 프로젝트에서는 한국어 word embedding 모델로 `Mecab`을 사용

- 한국어 임베딩 [ [github](https://github.com/ratsgo/embedding) ]
- word-embeddings [ [download](https://drive.google.com/file/d/1FeGIbSz2E1A63JZP_XIxnGaSRt7AhXFf/view) ]


# Directory Tree

```
Fake-News-Detection-Dataset
.
├── data
│   ├── Part1
│   │   ├── train
│   │   │   ├── Clickbait_Auto
│   │   │   │   ├── EC
│   │   │   │   ├── ET
│   │   │   │   ├── GB
│   │   │   │   ├── IS
│   │   │   │   ├── LC
│   │   │   │   ├── PO
│   │   │   │   └── SO
│   │   │   ├── Clickbait_Direct
│   │   │   └── NonClickbait_Auto
│   │   ├── validation
│   │   └── train
│   └── Part2
│   │   ├── train
│   │   ├── validation
│   │   └── train
├── docker
├── docs
├── LICENSE
├── part1_title
├── part2_context
├── README.md
└── requirements.txt

```

# Data

`./data`에는 다음과 같은 데이터 폴더 구조로 구성되어 있음

![image](https://user-images.githubusercontent.com/37654013/208360905-da4841f0-27d4-46f5-9e99-2179e9773cb5.png)


# Part 1: 제목 - 본문 일치성 [ [Part1](https://github.com/TooTouch/Fake-News-Detection-Dataset/tree/0bb478f18ad83cec2104a6ff8eebe3ff9f7b4e7a/part1_title) ]

## Baseline Models

- HAND[^1]
- FNDNet[^2]
- BERT[^3]

# Part 2: 주제 분리 탐지 [ [Part2](https://github.com/TooTouch/Fake-News-Detection-Dataset/tree/0bb478f18ad83cec2104a6ff8eebe3ff9f7b4e7a/part2_context) ]
## Baseline Models

- BERT[^4]
- KoBERTSeg[^5]


# Reference

[^1]: Jeong, H. (2021). Hierarchical Attention Networks for Fake News Detection (Doctoral dissertation, The Florida State University).
[^2]: Kaliyar, R. K., Goswami, A., Narang, P., & Sinha, S. (2020). FNDNet–a deep convolutional neural network for fake news detection. Cognitive Systems Research, 61, 32-44.
[^3]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT (1), 4171-4186
[^4]: 전재민, 최우용, 최수정, & 박세영. (2019). BTS: 한국어 BERT 를 사용한 텍스트 세그멘테이션. 한국정보과학회 학술발표논문집, 413-415.
[^5]: 소규성, 이윤승, 정의석, & 강필성. (2022). KoBERTSEG: 한국어 BERT 를 이용한 Local Context 기반 주제 분리 방법론. 대한산업공학회지, 48(2), 235-248. 
