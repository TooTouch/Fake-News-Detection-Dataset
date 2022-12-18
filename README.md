# Fake-News-Detection-Dataset

한국어 가짜 뉴스 탐지 데이터셋에 대한 baseline 실험 결과

<div align='center'>
[📘Documentation](https://tootouch.github.io/Fake-News-Detection-Dataset/)
</div>

# Enviroments

**Computer Resource**
- **CPU**: i7-7800K
- **GPU**: RTX 3080
- **RAM**: 64GB
- **SSD**: 512GB
- **OS**: ubuntu 18.04

**1. docker image**

```bash
bash ./docker/docker_build.sh $image_name
```


**2. Korean word-embeddings**

- 한국어 임베딩 [ [github](https://github.com/ratsgo/embedding) ]
- word-embeddings [ [download](https://drive.google.com/file/d/1FeGIbSz2E1A63JZP_XIxnGaSRt7AhXFf/view) ]


# Directory Tree

```
Fake-News-Detection-Dataset
.
├── data
│   ├── Part1
│   └── Part2
├── docker
├── docs
├── LICENSE
├── part1_title
├── part2_context
├── README.md
└── requirements.txt

```

# Data

`./data/Part1`에는 다음과 같은 데이터 폴더 구조로 구성되어 있음

![image](https://user-images.githubusercontent.com/37654013/207876089-16a0e88d-5fba-4aad-9654-521505a9e370.png)


# Part 1: 제목 - 본문 일치성 [ [Part1]() ]

## Baseline Models

- HAN[^1]
- FNDNet[^2]
- BERT[^3]

# Part 2: 주제 분리 탐지 [ [Part2]() ]
## Baseline Models

- BTS[^4]
- KoBERTSeg[^5]


# Reference

[^1]: Jeong, H. (2021). Hierarchical Attention Networks for Fake News Detection (Doctoral dissertation, The Florida State University).
[^2]: Kaliyar, R. K., Goswami, A., Narang, P., & Sinha, S. (2020). FNDNet–a deep convolutional neural network for fake news detection. Cognitive Systems Research, 61, 32-44.
[^3]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT (1), 4171-4186
[^4]: 전재민, 최우용, 최수정, & 박세영. (2019). BTS: 한국어 BERT 를 사용한 텍스트 세그멘테이션. 한국정보과학회 학술발표논문집, 413-415.
[^5]: 소규성, 이윤승, 정의석, & 강필성. (2022). KoBERTSEG: 한국어 BERT 를 이용한 Local Context 기반 주제 분리 방법론. 대한산업공학회지, 48(2), 235-248. 
