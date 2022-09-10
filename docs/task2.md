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
      <th>BTS</th>
      <td>0.992</td>
      <td>0.923</td>
      <td>0.897</td>
      <td>0.954</td>
      <td>0.990</td>
      <td>0.867</td>
    </tr>
    <tr>
      <th>KoBERTSeg</th>
      <td>0.994</td>
      <td>0.929</td>
      <td>0.911</td>
      <td>0.949</td>
      <td>0.990</td>
      <td>0.876</td>
    </tr>
  </tbody>
</table>