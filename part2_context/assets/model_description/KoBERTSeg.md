# KoBERTseg - 한국어 BERT를 이용한 Local Context 기반 주제 분리 방법론

# Model description
* KoBERTseg 모델은 KoBERT를 기반으로 하는 한국어 주제 분리 모델로, 입력되는 문장마다 추가한 [CLS] 토큰을 각 문장의 대표 표상이 되도록 학습하고 설정한 크기의 window 내에서 각 문장의 [CLS] 표상들을 합성곱 연산한 결과를 통해 주제 변화여부를 이진 분류한다.

# Model Architecture

# Input
* 기사 본문 전체를 입력받아 각 문장 마다 [CLS] 토큰을 추가하고, 모두 토큰화한 후 연결하여 하나의 샘플로 구성.
* window size 만큼 sliding window 방식으로 model에 input
* 특히 window 안에서 일관성을 평가하는 KoBERTseg가 기사의 시작과 끝 부분에서 주제가 변경되는 것을 탐지하지 못하는 것을 방지하기 위해 기사의 앞뒤에 (window size-1) 만큼 [PAD] 토큰을 추가함.  
* 구성요소:
	* input_ids: 실제 각 토큰의 인코딩 id (Batch size, Input length)
	* token_type_ids: 제목과 본문을 구분해주는 id (Batch size, Input length)
	* attention_mask: padding이 된 부분과 실제 토큰을 구분해주는 id (Batch size, Input length)
	* cls_ids: CLS 토큰의 인코딩 id (Batch size, cls token 개수)
	* mask_cls: classifier에서 CLS 토큰이 아닌 부분을 구분해주는 id (Batch size, cls token 개수)
  
`Batch size`: Batch 내 샘플 개수(default: 8), `Input length`: 샘플 내 토큰의 최대 개수(default: 512)  

# Output
* Shape: (Batch size, Number of Classes)
* Number of classes: 2 (이진 분류 과업)

# Task
* 본 과업에서는 KoBERTseg를 이용하여 뉴스 기사의 본문 내 일관성 여부 파악하는 것을 목표로 한다.


# Training Dataset

[`KoBERTSegDataset`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/dataset/kobertseg.py#L6)

# Training Setup

[`KoBERTSeg-train.yaml`](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/configs/KoBERTSeg/KoBERTSeg-train.yaml)

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

- 소규성, 이윤승, 정의석, & 강필성. (2022). KoBERTSEG: 한국어 BERT 를 이용한 Local Context 기반 주제 분리 방법론. 대한산업공학회지, 48(2), 235-248. 