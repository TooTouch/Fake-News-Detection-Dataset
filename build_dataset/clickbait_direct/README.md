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

**random_select**

- 전체 중 무작위로 제목 교체

  
**random_category_select**

- 같은 카테고리 내에서 무작위로 제목 교체


# Proposed Methods

제안 방법 구현 시 아래 가이드를 따라 작성한 후 실행해주세요.

1. 방법별 title을 바꾸기 위한 function은 `./methods`에 scripts를 작성 후 추가.

- 추가한 후 `./methods/__init__.py`에 해당 functions import 작성

2. `build.py`에서 `make_fake_title` function의 comments 중 `extract fake title`에 추가한 방법에 대한 arguments를 if 문을 추가하여 작성


```python
def make_fake_title(file_list: list, savedir: str, cfg_method: dict) -> None:
    '''
    make fake title using selected method
    '''
    for file_path in tqdm(file_list):
        
        # source file name and category
        category_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)

        # load source file
        source_file = json.load(open(file_path, 'r'))
        
        # extract fake title
        if cfg_method['name'] == 'random_select':
            kwargs = {
                'file_path':file_path,
                'file_list':file_list
            }
        elif cfg_method['name'] == 'random_category_select':
            kwargs = {
                'file_path':file_path,
                'category':category_name,
                'file_list':file_list
            }

        fake_title = __import__('methods').__dict__[cfg_method['name']](**kwargs)
        
        # update label infomation
        source_file = update_label_info(file=source_file, new_title=fake_title)
        
        # save source file
        category_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        json.dump(
            obj          = source_file, 
            fp           = open(os.path.join(savedir, category_name, file_name), 'w', encoding='utf-8'), 
            indent       = '\t',
            ensure_ascii = False
        )

```

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


