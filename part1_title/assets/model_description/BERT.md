# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding


# Model description
* BERT는 Transformer의 Encoder를 사용하여 Masked Language Modeling(MIM)과 Next Sentence Prediction(NSP)를 통해 사전학습한 구조이다. 본 연구에서는 사전학습 된 BERT모델의 마지막 CLS token의 hidden representation을 통해 linear classifier를 학습하여 이진 분류를 수행하였다.

# Model Architecture

<p align="center">
    <img width="600" src="https://user-images.githubusercontent.com/37654013/208855490-8edf7f15-06f4-4c2e-b28e-f2d01b606107.png">
</p>

# Input
* 뉴스 기사 본문 내 연속된 두 문장을 [sep] 토큰으로 연결하여 하나의 샘플로 구성
* 구성요소
	* src: 각 토큰의 인코딩 id (Batch size, Max input length)
	* segs: 두 문장을 구분하는 id (Batch size, Max input length)
	* mask_src: padding이 된 부분과 실제 토큰을 구분해주는 id (Batch size, Max input length)
  
`Batch size`: Batch 내 샘플 개수(default: 8), `Max input length`: 토큰의 최대 개수(default: 512)  


# Output
* Shape: (Batch size, Number of Classes)  
* Number of classes: 클래스의 개수(default: 2 / 이진 분류 과업)  

# Task
* 본 과업에서는 BERT를 이용하여 뉴스 기사의 제목과 본문의 일관성 여부 파악하는 것을 목표로 한다


# Training Dataset

[`BERTDataset`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/bert.py#L5)

# Training Setup

[`BERT-train.yaml`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/configs/BERT/BERT-train.yaml)

```yaml
TRAIN:
    batch_size: 8
    num_training_steps: 10000
    accumulation_steps: 1
    num_workers: 12
    use_wandb: True

LOG:
    log_interval: 1
    eval_interval: 1000

OPTIMIZER: # AdamW
    lr: 0.00001
    weight_decay: 0.0005
```

# Evaluation Metric

- Accuracy

# Reference

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT (1), 4171-4186