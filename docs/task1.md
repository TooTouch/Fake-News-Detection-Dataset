---
layout: default
title: Task 1. 제목 - 본문 일치성
description: 제목과 본문 간 일치 여부에 따라 가짜 뉴스를 탐지하는 task 입니다.
---

# Task 1. 제목 - 본문 일치성

## Baseline Models

- HAN
- FNDNet
- BERT


## 실행

```bash
# HAN
python main.py --yaml_configs ./configs/HAN/HAN_w_freeze_w2e-train.yaml

# FNDNet
python main.py --yaml_configs ./configs/FNDNet/FNDNet_w_freeze_w2e-train.yaml

# BTS
python main.py --yaml_configs ./configs/BTS/BTS-train.yaml
```

# Load Pretrained Model

**model list**

`bts`, `fndnet`, `han` 이라고 적힌 모델은 학습된 모델이 아닙니다.

```python
from models import list_models

list_models('*')

['bts',
 'bts_task1',
 'fndnet',
 'fndnet_w_freeze_w2e_task1',
 'fndnet_wo_freeze_w2e_task1',
 'han',
 'han_w_freeze_w2e_task1',
 'han_wo_freeze_w2e_task1']
```

**load pretrained model**

```python
from models import create_model

model = create_model('bts_task1', pretrained=True)
```


## Result

- `freeze`는 word embedding weight 학습 여부입니다.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th align='left'>FNDNet - freeze</th>
      <td>0.924</td>
      <td>0.842</td>
      <td>0.807</td>
      <td>0.880</td>
      <td>0.848</td>
    </tr>
    <tr>
      <th align='left'>FNDNet + freeze</th>
      <td>0.924</td>
      <td>0.842</td>
      <td>0.807</td>
      <td>0.880</td>
      <td>0.848</td>
    </tr>
    <tr>
      <th align='left'>HAN - freeze</th>
      <td>0.951</td>
      <td>0.872</td>
      <td>0.817</td>
      <td>0.936</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th align='left'>HAN + freeze</th>
      <td>0.954</td>
      <td>0.874</td>
      <td>0.821</td>
      <td>0.935</td>
      <td>0.882</td>
    </tr>
    <tr>
      <th align='left'>BTS</th>
      <td>1.000</td>
      <td>0.998</td>
      <td>0.997</td>
      <td>1.000</td>
      <td>0.998</td>
    </tr>
  </tbody>
</table>