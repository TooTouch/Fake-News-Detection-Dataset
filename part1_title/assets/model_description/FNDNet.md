# FNDNet – A deep convolutional neural network for fake news detection

# Model description
* FNDNet 모델은 세개의 병렬적인 Deep Convolutional neural network을 이용하여 
제목과 본문 간의 일관성을 파악하기 위한 용도로 활용되었다.

# Model Architecture

<p align="center">
    <img width="300" src="https://user-images.githubusercontent.com/37654013/208851157-97d95c95-dfc8-4850-97e3-bb026517919b.png">
</p>

# Input
* 기사와 본문을 GloVe를 통해 모두 토큰화 한 후 연결하여 하나의 샘플로 구성  
* Shape: (Batch size, Word length)  
* Batch size: Batch 내 샘플 개수(실험 환경 batch size=256)    
* Word length: 토큰의 최대 개수(실험 환경 vocab_len=50002(max_vocab_size + [UNK] + [PAD]))

# Output
* Shape: (Batch size, Number of Classes)
* Number of classes: 2 (이진 분류 과업)

# Task
* 본 과업에서는 FNDNet을 이용하여 뉴스 기사의 제목과 본문의 일관성 여부 파악하는 것을 목표로 한다.


# Training Dataset

[`FNDNetDataset`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/fndnet.py#L5)

# Training Setup

[`FNDNet-train.yaml`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/configs/FNDNet/FNDNet-train.yaml)

```yaml
TRAIN:
    batch_size: 256
    num_training_steps: 100000
    accumulation_steps: 1
    num_workers: 12
    use_wandb: True

LOG:
    log_interval: 10
    eval_interval: 1000

OPTIMIZER: # AdamW
    lr: 0.00003
    weight_decay: 0.0005

SCHEDULER:
    warmup_ratio: 0.1
    use_scheduler: True
```

# Evaluation Metric

- Accuracy

# Reference

- Kaliyar, R. K., Goswami, A., Narang, P., & Sinha, S. (2020). FNDNet–a deep convolutional neural network for fake news detection. Cognitive Systems Research, 61, 32-44.