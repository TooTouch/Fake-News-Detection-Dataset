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
- MuSeM


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

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="5" halign="left">TRAIN</th>
      <th colspan="5" halign="left">VALID</th>
      <th colspan="5" halign="left">TEST</th>
    </tr>
    <tr>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FNDNet_wo_freeze_w2e</th>
      <td>0.990</td>
      <td>0.955</td>
      <td>0.938</td>
      <td>0.973</td>
      <td>0.956</td>
      <td>0.941</td>
      <td>0.868</td>
      <td>0.847</td>
      <td>0.891</td>
      <td>0.872</td>
      <td>0.924</td>
      <td>0.842</td>
      <td>0.807</td>
      <td>0.880</td>
      <td>0.848</td>
    </tr>
    <tr>
      <th>FNDNet_w_freeze_w2e</th>
      <td>0.990</td>
      <td>0.955</td>
      <td>0.938</td>
      <td>0.973</td>
      <td>0.956</td>
      <td>0.941</td>
      <td>0.868</td>
      <td>0.847</td>
      <td>0.891</td>
      <td>0.872</td>
      <td>0.924</td>
      <td>0.842</td>
      <td>0.807</td>
      <td>0.880</td>
      <td>0.848</td>
    </tr>
    <tr>
      <th>HAN_wo_freeze_w2e</th>
      <td>0.975</td>
      <td>0.919</td>
      <td>0.886</td>
      <td>0.954</td>
      <td>0.922</td>
      <td>0.965</td>
      <td>0.902</td>
      <td>0.868</td>
      <td>0.937</td>
      <td>0.905</td>
      <td>0.951</td>
      <td>0.872</td>
      <td>0.817</td>
      <td>0.936</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th>HAN_w_freeze_w2e</th>
      <td>0.978</td>
      <td>0.921</td>
      <td>0.887</td>
      <td>0.958</td>
      <td>0.924</td>
      <td>0.968</td>
      <td>0.905</td>
      <td>0.870</td>
      <td>0.943</td>
      <td>0.909</td>
      <td>0.954</td>
      <td>0.874</td>
      <td>0.821</td>
      <td>0.935</td>
      <td>0.882</td>
    </tr>
    <tr>
      <th>BTS</th>
      <td>1.000</td>
      <td>0.999</td>
      <td>0.998</td>
      <td>0.999</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>0.999</td>
      <td>0.998</td>
      <td>0.999</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>0.998</td>
      <td>0.997</td>
      <td>1.000</td>
      <td>0.998</td>
    </tr>
  </tbody>
</table>