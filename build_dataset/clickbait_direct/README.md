# 직섭생성 실험 데이터 구성

난이도 높은 가짜 뉴스기사 제목을 생성을 위한 방법론 제안



# Configurations

ex) `./configs/tfidf_title_category.yaml`

```yaml
datadir: '../../data/labeled_fake_news/Part1' 
savedir: '../../data/direct_exp/Part1'
METHOD:
  name: 'tfidf'
  target: 'title'
  select_name: 'tfidf_title_category_select'

SEED: 42
```


# Methods

- Cosine simliarity 사용
- Similarity matrix는 category 별로 수행

Method | Description 
---|---|---
TF-IDF | TF-IDF로 유사도를 계산하여 교체
BoW | Bag of Words로 유사도를 계산하여 교체
N-gram | Bi/Tri-gram을 통해 유사도를 계산하여 교체
Sentence Embedding | Sentence BERT(RoBERTa)를 통해 유사도를 계산하여 교체


# Run

가짜 뉴스시사 제목 생성 실행

```bash
python build.py --yaml_config ${config file path}
```


# Results

## Similarity Matrix Time

**제목 간 비교**

Method | Time
---|---
TF-IDF | 4분 21초
BoW | 4분 23초
N-gram | 1시간 16분 33초
Sentence Embedding |

**본문 간 비교**

Method | Time
---|---
TF-IDF | 20분 9초
BoW | 20분 4초
N-gram | 
Sentence Embedding |

