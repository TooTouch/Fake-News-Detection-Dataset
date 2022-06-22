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

**HAN**

```bash
python main.py \
--do_train \
--exp_name HAN_wo_freeze_w2e \
--modelname HAN \
--num_training_steps 30000 \
--batch_size 256 \
--use_scheduler \
--lr 3e-3 \
--use_pretrained_word_embed \
--max_vocab_size 50000 \
--max_sent_len 16 \
--max_word_len 64 \
--use_saved_data \
--log_interval 10 \
--eval_interval 1000
```


**FNDNet**

```bash
python main.py \
--do_train \
--exp_name FNDNet_wo_freeze_w2e \
--modelname FNDNet \
--num_training_steps 100000 \
--batch_size 256 \
--use_scheduler \
--lr 3e-5 \
--use_pretrained_word_embed \
--use_saved_data \
--max_vocab_size 50000 \
--max_word_len 1000 \
--dims 128 \
--log_interval 10 \
--eval_interval 1000
```


**BTS**

```bash
python main.py \
--do_train \
--exp_name BTS \
--modelname BTS \
--pretrained_name 'klue/bert-base' \
--tokenizer 'bert' \
--num_training_steps 20 \
--use_saved_data \
--batch_size 8 \
--use_scheduler \
--lr 1e-5 \
--max_word_len 512 \
--log_interval 1 \
--eval_interval 5
```


## Result


       
 Model |  Test Acc |  Test Loss |  Train Acc |  Train Loss |  Valid Acc |  Valid Loss 
:---|---|---|---|---|---|---
BTS |    0.9982 |     0.1871 |     0.9986 |      0.1879 |     0.9986 |      0.1882 
FNDNet |    0.8484 |     0.3981 |     0.9562 |      0.1264 |     0.8717 |      0.3393 
HAN |    0.8805 |     0.2915 |     0.9215 |      0.1961 |     0.9052 |      0.2334 