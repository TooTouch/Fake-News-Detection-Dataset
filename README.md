# Fake-News-Detection-Dataset
한국어 가짜 뉴스 탐지 데이터셋

# Enviroments

1. docker 

```bash
bash ./docker/docker_build.sh $image_name
```

2. requirements.txt

```bash
pip install -r requirements.txt
```

3. Korean word-embeddings

- 한국어 임베딩 [ [github](https://github.com/ratsgo/embedding) ]
- word-embeddings [ [download](https://drive.google.com/file/d/1FeGIbSz2E1A63JZP_XIxnGaSRt7AhXFf/view) ]

# Task 1: 제목 - 본문 일치성

## Baseline Models

- HAN
- FNDNet
- BERT
- MuSeM

# Task 2: 본문 내 일치성

## Baseline Models

- BTS
- KoBERTSeg
