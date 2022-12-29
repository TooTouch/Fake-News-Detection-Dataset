# BERT

BERT for Topic Segmentation

# Model description
* BERT for Topic Segmentation은 BERT(Devlin et al., 2019)을 이용하여 서로 연속된 두 문장의 연관성을 파악하고 연관성의 유무에 따라 문단을 구분하는 것이 목표인 방법론이다.

# Model Architecture

<p align="center">
    <img width="600" src="https://user-images.githubusercontent.com/37654013/208857336-281cc50f-e305-4810-9532-728f1a76c2c0.png">
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
* 본 과업에서는 BERT를 이용하여 뉴스 기사의 본문의 일관성 여부 파악하는 것을 목표로 한다.

# Training Dataset

[`BTSDataset`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/dataset/bts.py#L6)

# Training Setup

[`BERT-train.yaml`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/configs/BERT/BERT-train.yaml)

```yaml
TRAIN:
    batch_size: 8
    num_training_steps: 15000
    accumulation_steps: 1
    num_workers: 12
    use_wandb: True

LOG:
    log_interval: 1
    eval_interval: 2500

OPTIMIZER: # AdamW
    lr: 0.00001
    weight_decay: 0.0005

SCHEDULER:
    warmup_ratio: 0.1
    use_scheduler: True
```

# Evaluation Metric

- Accuracy (문서 단위 정확도)
- F1-Score

# Reference

- 전재민, 최우용, 최수정, & 박세영. (2019). BTS: 한국어 BERT 를 사용한 텍스트 세그멘테이션. 한국정보과학회 학술발표논문집, 413-415.