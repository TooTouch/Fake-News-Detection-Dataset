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
python main.py --yaml_configs ./configs/HAN/HAN_-train.yaml

# FNDNet
python main.py --yaml_configs ./configs/FNDNet/FNDNet-train.yaml

# BTS
python main.py --yaml_configs ./configs/BTS/BTS-train.yaml
```

# Load Pretrained Model

**model list**


```python
from models import list_models

list_models('*')

['bts',
 'fndnet',
 'han']
```

**load pretrained model**

ex) HAN

**HAN-test.yaml**

```yaml
...

TOKENIZER: 
    name: mecab
    vocab_path: ./word-embeddings/glove/glove.txt
    max_vocab_size: 50000  

MODEL:
    modelname: han
    freeze_word_embed: True
    use_pretrained_word_embed: True
    PARAMETERS:
        num_classes: 2
        vocab_len: 50002
        dropout: 0.1
        word_dims: 32
        sent_dims: 64
        embed_dims: 100
    CHECKPOINT:
        checkpoint_path: ./saved_model/HAN/best_model.pt

...

```

**create model**

```python
import yaml
from models import create_model

cfg = yaml.load(open(yaml_config_path,'r'), Loader=yaml.FullLoader)

tokenizer, word_embed = create_tokenizer(
    name            = cfg['TOKENIZER']['name'], 
    vocab_path      = cfg['TOKENIZER'].get('vocab_path', None), 
    max_vocab_size  = cfg['TOKENIZER'].get('max_vocab_size', None)
)

model = create_model(
    modelname                 = cfg['MODEL']['modelname'],
    hparams                   = cfg['MODEL']['PARAMETERS'],
    word_embed                = word_embed,
    tokenizer                 = tokenizer,
    freeze_word_embed         = cfg['MODEL'].get('freeze_word_embed',False),
    use_pretrained_word_embed = cfg['MODEL'].get('use_pretrained_word_embed',False),
    checkpoint_path           = cfg['MODEL']['CHECKPOINT']['checkpoint_path'],
)
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