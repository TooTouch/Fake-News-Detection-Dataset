---
layout: default
title: Task 2. 본문 내 일관성
description: 본문 내 내용 간 일관성 여부를 판단하여 가짜 뉴스를 탐지하는 task 입니다.
---

# Task 2. 본문 내 일관성

## Baseline Models

- BTS
- KoBERTSeg


## 실행

```bash
# BTS
python main.py --yaml_configs ./configs/BTS/BTS-train.yaml

# KoBERTSeg
python main.py --yaml_configs ./configs/KoBERTSeg/KoBERTSeg-train.yaml
```

# Load Pretrained Model

**model list**

```python
from models import list_models

list_models('*')

['bts',
 'kobertseg']
```

**load pretrained model**

ex) KoBERTSeg

**koBERTSeg-test.yaml**

```yaml
...

MODEL:
    modelname: kobertseg
    PARAMETERS:
        window_size: 3
        finetune_bert: True
    CHECKPOINT:
        checkpoint_path: ./saved_model/KoBERTSeg/best_model.pt

...

```

**create model**

```python
import yaml
from models import create_model

cfg = yaml.load(open(yaml_config_path,'r'), Loader=yaml.FullLoader)

model = create_model(
    modelname  = cfg['MODEL']['modelname'], 
    hparams    = cfg['MODEL']['PARAMETERS'],
    pretrained = cfg['MODEL']['CHECKPOINT']['pretrained'], 
    checkpoint_path = cfg['MODEL']['CHECKPOINT']['checkpoint_path']
)
```


## Result

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
      <th>ACC_PER_ARTICLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th align='left'>BTS</th>
      <td>0.992</td>
      <td>0.923</td>
      <td>0.897</td>
      <td>0.954</td>
      <td>0.990</td>
      <td>0.867</td>
    </tr>
    <tr>
      <th align='left'>KoBERTSeg</th>
      <td>0.994</td>
      <td>0.929</td>
      <td>0.911</td>
      <td>0.949</td>
      <td>0.990</td>
      <td>0.876</td>
    </tr>
  </tbody>
</table>