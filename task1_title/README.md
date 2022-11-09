# Task1 

제목 내용 일치성


# Configurations

학습을 위해 사용되는 arguments 구성

**BTS**

- `./configs/BTS/BTS-train.yaml`
- `./configs/BTS/BTS-test-checkpoint.yaml`
- `./configs/BTS/BTS-test-pretrained.yaml`

**FNDNet**

- `./configs/FNDNet/FNDNet_save_dataloader.yaml`
- `./configs/FNDNet/FNDNet_w_freeze_w2e-train.yaml`
- `./configs/FNDNet/FNDNet_w_freeze_w2e-test-checkpoint.yaml`
- `./configs/FNDNet/FNDNet_w_freeze_w2e-test-pretrained.yaml`
- `./configs/FNDNet/FNDNet_wo_freeze_w2e-train.yaml`
- `./configs/FNDNet/FNDNet_wo_freeze_w2e-test-checkpoint.yaml`
- `./configs/FNDNet/FNDNet_wo_freeze_w2e-test-pretrained.yaml`


**HAN**

- `./configs/HAN/HAN_save_dataloader.yaml`
- `./configs/HAN/HAN_w_freeze_w2e-train.yaml`
- `./configs/HAN/HAN_w_freeze_w2e-test-checkpoint.yaml`
- `./configs/HAN/HAN_w_freeze_w2e-test-pretrained.yaml`
- `./configs/HAN/HAN_wo_freeze_w2e-train.yaml`
- `./configs/HAN/HAN_wo_freeze_w2e-test-checkpoint.yaml`
- `./configs/HAN/HAN_wo_freeze_w2e-test-pretrained.yaml`



# Save Datasets

학습 시 전처리 시간을 줄이기 위해 datasets을 사전에 저장하여 사용

yaml에서 data_path 또는 data_info_path를 확인 후 실행

**ex)** `./configs/BTS/BTS_save_dataloader.yaml`

```yaml
EXP_NAME: BTS
SEED: 223
    
DATASET:
    name: BTS
    data_path: ../data/direct_exp/Part1 # news article directory
    data_info_path: ../data/direct_exp/Part1/random_select # splitted csv file director
    saved_data_path: false
    PARAMETERS:
        max_word_len: 512

TOKENIZER:
    name: bert

TRAIN:
    batch_size: 256
    num_workers: 12

RESULT:
    savedir: ../data/direct_exp/Part1
    dataname: 'BTS_w512-direct_exp-random_select'

```


**run**

```bash
python save_dataloader.py --yaml_config ${config file path}
```


# Run


yaml에서 data_path 또는 data_info_path를 확인 후 실행


```
python main.py --yaml_config ${config file path}
```





