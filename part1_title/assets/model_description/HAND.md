# HAND : Hierarchical Attention Networks for Document Classification

# Model description
* HAND 모델은 Word Attention Layer와 Sentence Attention Layer로 구성된 두 단계의 Hierarchical Attention 구조를 통해 서로 연속된 두 문장의 일관성을 파악하기 위한 용도로 활용되었다. 

# Model Architecture
HAND 구조는 크게 단어 level의 hidden representation을 구하는 부분과 문장 level의 hidden representation을 구하는 부분으로 나누어볼 수 있다. 가장 먼저 단어 임베딩을 입력으로 받아 GRU layer를 통과시켜 각 단어를 인코딩하고, 인코딩된 값으로 attention을 수행하여 단어별 hidden representation을 구한다. 이후 문장별로 각 문장에 속한 단어의 hidden representation을 가중합해 sentence vector를 만들고, 해당 값을 이용해 위와 동일한 인코딩, attention 과정을 거쳐 문장별 최종 hidden representation을 구한다.

<p align="center">
    <img width="600" src="https://user-images.githubusercontent.com/37654013/208854814-e6328428-84d2-4015-873a-ca288793aa5d.png">
</p>

# Input
* Shape: (Batch Size, Sentence Length, Word Length)  
* Batch size: Batch 내 샘플 개수 (default: 256)
* Sentence Length: 하나의 샘플을 구성할 최대 문장 개수 (default: 16)
* Word Length: 하나의 문장을 구성할 최대 단어 개수 (default: 64)

# Output
* Shape: (Batch size, Number of Classes)
* Number of classes: 2 (이진 분류 과업)

# Task
* 본 과업에서는 HAND을 이용하여 뉴스 기사의 제목과 본문의 일관성 여부 파악하는 것을 목표로 한다


# Training Dataset

[`HANDDataset`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/hanD.py#L5)

# Training Setup

[`HAND-train.yaml`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/configs/HAND/HAND-train.yaml)

```yaml
TRAIN:
    batch_size: 256
    num_training_steps: 30000
    accumulation_steps: 1
    num_workers: 12
    use_wandb: True

LOG:
    log_interval: 10
    eval_interval: 1000

OPTIMIZER: # AdamW
    lr: 0.003
    weight_decay: 0.0005
```

# Evaluation Metric

- Accuracy

# Reference

- Jeong, H. (2021). Hierarchical Attention Networks for Fake News Detection (Doctoral dissertation, The Florida State University).