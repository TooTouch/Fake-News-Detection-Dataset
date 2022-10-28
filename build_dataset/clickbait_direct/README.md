# 직섭생성 실험 데이터 구성

난이도 높은 가짜 뉴스기사 제목을 생성을 위한 방법론 제안



# Configurations

생성 방법별 arguments 구성


- `./configs/random.yaml`
- `./configs/random_category.yaml`




ex) `./configs/random.yaml`

```yaml
BUILD:
  datadir: '../../data/origin_fake_news/Clickbait_Auto' 
  savedir: '../../data/direct_exp/Part1/Clickbait_Auto'
  METHOD:
    name: 'random_select' # 방법론 이름

SPLIT:
  datadir: '../../data/origin_fake_news'
  savedir: '../../data/direct_exp/Part1'
  ratio:
    - 6 # train ratio
    - 2 # validation ratio
    - 2 # test ratio

SEED: 42
```


# Methods

1. random_select

전체 중 무작위로 제목 교체

2. random_category_select

같은 카테고리 내에서 무작위로 제목 교체


# Run

가짜 뉴스시사 제목 생성 실행

```bash
python build.py --yaml_config ${config file path}
```


# Split dataset into train, validation, and test

학습을 위한 dataset 구성

```bash
python split_data.py --yaml_confg ${config file path}
```


