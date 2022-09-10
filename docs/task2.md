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


## Result

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="6" halign="left">TRAIN</th>
      <th colspan="6" halign="left">VALIDATION</th>
      <th colspan="6" halign="left">TEST</th>
    </tr>
    <tr>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
      <th>ACC_PER_ARTICLE</th>
      <th>AUROC</th>
      <th>F1</th>
      <th>RECALL</th>
      <th>PRECISION</th>
      <th>ACC</th>
      <th>ACC_PER_ARTICLE</th>
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
      <th>BTS</th>
      <td>0.991</td>
      <td>0.909</td>
      <td>0.875</td>
      <td>0.950</td>
      <td>0.988</td>
      <td>0.845</td>
      <td>0.991</td>
      <td>0.906</td>
      <td>0.872</td>
      <td>0.949</td>
      <td>0.988</td>
      <td>0.842</td>
      <td>0.992</td>
      <td>0.923</td>
      <td>0.897</td>
      <td>0.954</td>
      <td>0.99</td>
      <td>0.867</td>
    </tr>
    <tr>
      <th>KoBERTSeg</th>
      <td>0.991</td>
      <td>0.894</td>
      <td>0.860</td>
      <td>0.937</td>
      <td>0.986</td>
      <td>0.824</td>
      <td>0.991</td>
      <td>0.894</td>
      <td>0.860</td>
      <td>0.937</td>
      <td>0.986</td>
      <td>0.824</td>
      <td>0.994</td>
      <td>0.929</td>
      <td>0.911</td>
      <td>0.949</td>
      <td>0.99</td>
      <td>0.876</td>
    </tr>
  </tbody>
</table>