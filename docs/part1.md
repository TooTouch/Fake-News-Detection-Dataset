---
layout: default
title: Part 1. 제목 - 본문 일치성
description: 제목과 본문 간 일치 여부에 따라 가짜 뉴스를 탐지하는 task 입니다.
---

# Dataset


## FakeDataset


> class FakeDataset(*tokenizer*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/build_dataset.py#L13)]

낚시성 데이터 제목 본문 간 불일치 탐지를 위한 base Dataset

Parameters:

- **tokenizer** - SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)

> > load_dataset(*data_dir, split, saved_data_path=False)

뉴스 기사 데이터 불러오기

Parameters:

- **data_dir** (*str*) - 데이터 폴더 경로. ex) `../data/Part1`
- **split** (*str*) - 데이터셋 이름. ex) `train`, `validation`, 또는 `test`
- **saved_data_path** (*str*) - 전처리가 완료된 `.pt` 데이터 경로


---

## FNDNetDataset

> class FNDNetDataset(*tokenizer, max_word_len*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/fndnet.py#L5)]

`FNDNet` 모델을 위한 데이터셋

Parameters:

- **tokenizer** - SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **max_word_len** (*int*) - 입력 단어 기준 token 최대 개수


> > transform(*title, text*)

뉴스 기사 제목과 본문을 통해 입력 tokens 반환

Parameters:

- **title** (*str*) - 뉴스 기사 제목 텍스트
- **text** (*list*) - 뉴스 기사 본문 텍스트

Returns:

- **doc** (*dict*) - 전처리가 완료된 입력 `input_ids`

> > padding(*doc*)

`max_word_len` 보다 작은 입력 데이터에 대한 padding

Parameters:

- **doc** (*list*) - 뉴스 기사 제목과 본문이 포함된 list

Returns:

- **doc** (*list*) - padding 처리가 완료된 입력 token ids

---


## HANDDataset

> class HANDDataset(*tokenizer, max_word_len, max_sent_len*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/hand.py#L5)]

`HAND` 모델을 위한 데이터셋

Parameters:

- **tokenizer** - SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **max_word_len** (*int*) - 한 문장 내 단어 최대 token 개수 
- **max_sent_len** (*int*) - 전체 문장 최대 개수


> > transform(*title, text*)

뉴스 기사 제목과 본문을 통해 입력 tokens 반환

Parameters:

- **title** (*str*) - 뉴스 기사 제목 텍스트
- **text** (*list*) - 뉴스 기사 본문 텍스트

Returns:

- **doc** (*dict*) - 전처리가 완료된 입력 `input_ids`



> > padding(*doc*)

`max_word_len`와 `max_sent_len` 보다 작은 입력 데이터에 대한 padding

Parameters:

- **doc** (*list*) - 뉴스 기사 제목과 본문이 포함된 list

Returns:

- **doc** (*list*) - padding 처리가 완료된 입력 token ids



---

## BERTDataset

> class BERTDataset(*tokenizer, max_word_len*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/bert.py#L5)]

BERT 모델을 위한 Fake News Detection Dataset

Parameters:

- **tokenizer** - SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **max_word_len** (*int*) - 입력 단어 기준 token 최대 개수

> > transform(*title, text*)

Parameters:

뉴스 기사 제목과 본문을 통해 입력 tokens 반환

Parameters:

- **title** (*str*) - 뉴스 기사 제목 텍스트
- **text** (*list*) - 뉴스 기사 본문 텍스트

Returns:

- **doc** (*dict*) - 전처리가 완료된 입력 `input_ids`, `attention_mask`, 그리고 `token_type_ids`

> > tokenize(*src*)

입력 데이터 구성을 위한 데이터 전처리 함수

Parameters:

- **src** (*list*) - tokenizer를 거친 모든 뉴스기사 데이터에 대한 tokens

Returns:

- **input_ids** (*list*) - padding 처리된 token ids
- **token_type_ids** (*list*) - padding 처리된 segment ids
- **attention_mask** (*list*) - token ids 이외에 padding이 된 부분에 대한 mask


> > length_precessing(*src*)

입력 데이터의 token 개수가 `max_word_len`을 넘는 경우 뒤에서부터 자르는 것이 아닌 문장별 최대 길이를 제한하여 처리

Parameters:

- **src** (*list*) - `preprocessor`를 거친 모든 뉴스기사 데이터에 대한 tokens

Returns:

- **processed_src** (*list*) - 길이 제한 처리된 tokens

> > pad(*data, pad_idx*)

`max_word_len` 보다 작은 입력 데이터에 대한 padding

Parameters:

- **data** (*list*) - 입력 데이터 tokens list
- **pad_idx** (*int*) - vocab의 padding token id


Returns:

- **data** (*list*) - padding 처리된 입력 데이터 tokens list

> > padding_bert(*input_ids, token_type_ids*)

KoBERT의 모든 입력 데이터에 대한 padding 함수

Parameters:

- **input_ids** (*list*) - 입력 데이터로 사용된 문서에 대한 token ids
- **token_type_ids** (*list*) - 입력 데이터로 사용된 segment ids 


Returns:

- **input_ids** (*torch.Tensor*) - padding 처리된 token ids
- **token_type_ids** (*torch.Tensor*) - padding 처리된 segment ids
- **attention_mask** (*torch.Tensor*) - token ids 이외에 padding이 된 부분에 대한 mask

> > get_token_type_ids(*input_ids*)

입력 데이터 내 문장 별 segment ids를 계산하기 위한 함수

Parameters:

- **input_ids** (*list*) - 입력 데이터로 사용된 문서에 대한 token ids

Returns:

- **token_type_ids** (*list*) - 입력 `input_ids`에 대한 문장별 segment ids


---

## FNDTokenizer

> class FNDTokenizer(*vocab, tokenizer, special_tokens*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/tokenizer.py#L1)]

학습된 word embedding의 vocabulary와 Mecab 형태소 분석시를 활용하여 구성한 Fake News Tokenizer

Parameters:

- **vocab** (*list*) - word-embedding을 학습을 위해 사용된 vocab
- **tokenizer** - `KoNLPy`에서 제공하는 `Mecab` 사용
- **special_tokens** (*list*) - special tokens을 지정하는 경우 기존 vocab에 단어 추가


> > encode(*sentence*)

텍스트를 학습에 사용한 vocab의 id로 변환

Parameters:

- **sentence** (*list*) - token id로 변환할 텍스트

Returns:

- (*list*) - token id로 반환된 텍스트

> > batch_encode(*b_sentence*)

Batch 단위로 encode을 수행

Parameters:

- **b_sentence** (*list*) - token id로 변환할 batch size 단위 텍스트

Returns:

- (*list*) - token id로 반환된 batch size 단위 텍스트

> > decode(*input_ids*)

token ids를 vocab을 통해 다시 텍스트로 반환

Parameters:

- **input_ids** (*list) - 입력 token ids

Returns:

- (*list*) - vocab을 통해 단어로 반환된 token ids

> > batch_decode(*b_input_ids*)

Batch 단위로 decode을 수행

Parameters:

- **b_input_ids** (*list*) - batch size 단위 입력 token ids

Returns:

- (*list*) - vocab을 통해 단어로 반환된 batch size 단위 token ids

> > add_tokens(*name*)

special token을 추가하는 함수

Parameters:

- **name**: 새로운 단어




---

## factory

> extract_word_embedding(*vocab_path, max_vocab_size*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/factory.py#L17)]

학습된 word embedding의 vocabulary와 word embedding weights 추출

Parameters:

- **vocab_path** (*str*) - 학습된 word embeddnig의 경로
- **max_vocab_size** (*int*) - 최대 vocabulary 크기

Returns:

- **vocab** (*list*) - word embedding 학습에 사용된 vocabulary
- **word_embed** (*np.ndarray*) - 학습된 word embedding


> create_tokenizer(*name, vocab_path, max_vocab_size*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/factory.py#L34)]

사용할 모델에 맞는 Tokenizer 반환

Parameters:

- **name** (*list*) - 사용할 tokenizer 이름. ex) `mecab` 또는 `bert`
- **vocab_path** (*str*) - 학습된 word embeddnig의 경로
- **max_vocab_size** (*int*) - 최대 vocabulary 크기

Returns:

- **tokenizer** - `name`이 `mecab`인 경우 Mecab이 사용된 `FNDTokenizer` 또는 `name`이 `bert`인 경우 SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **word_embed** (*np.ndarray*) - 학습된 word embedding


> create_dataset(*name, data_path, split, tokenizer, saved_data_path, kwargs*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/factory.py#L46)]

사용할 모델에 맞는 Dataset 반환

Parameters:

- **name** (*str*) - 사용할 모델의 데이터셋 이름
- **data_path** (*str*) - 데이터 폴더 경로. ex) `../data/Part1`
- **split** (*str*) - 데이터셋 이름. ex) `train`, `validation`, 또는 `test`
- **saved_data_path** (*str*) - 전처리가 완료된 `.pt` 데이터 경로
- **tokenizer** - `FNDTokenizer` 또는 `BERTSPTokenizer`
- **kwargs** (*dict*) - 데이터셋 별 hyper-parameters

Returns:

`Dataset`

> create_dataloader(*dataset, batch_size, num_workers, shuffle*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/dataset/factory.py#L61)]

Parameters:

- **dataset** `create_dataset`을 통해 반환된 `Dataset`
- **batch_size** (*int*) - 입력 데이터의 batch size
- **num_workers** (*int*) - 사용할 worker 수
- **shuffle** (*bool*) - random하게 dataset의 index를 반환할지에 대한 여부


Returns:

`DataLoader`


---

# Models

## BERT

> bert(*hparams*)

Parameters:

- **hparams** (*dict*) - 모델 학습에 필요한 hyper-parameters. [BERT configuration](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/configs/BERT/BERT-train.yaml)


Returns:

- `BERT`

> class BERT(*pretrained_name, config, num_classes*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/models/bert.py#L9)]

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)에서 제안한 BERT 모델 사용. 모델 설명은 [여기](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/model_description/BERT.md)에서 확인할 수 있습니다.

Parameters:
- **pretrained_name** (*str*) - huggingface에 저장된 사전학습 BERT 이름
- **config** (*dict*) - huggingface에 저장된 사전학습 모델에 사용된 configuration
- **num_classes** (*int*) - 학습할 class 수
  
> > forward(*input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attention, output_hidden_states*)

Parameters:

- **input_ids** (*torch.Tensor, optional*) - 입력 텍스트를 변환한 token ids. Default = `None`
- **attention_mask** (*torch.Tensor, optional*) - padding 처리된 부분에 대한 mask 여부. Default = `None`
- **token_type_ids** (*torch.Tensor, optional*) - 입력 문장 간 구분을 위한 segment id. Default = `None`
- **output_attention** (*bool, optional*) - attention score 출력 여부. Default = `None`
  
Returns:

- **logits** (*torch.Tensor*) - $\hat{y} \in \mathbf{R}^{batch size \times 2}$

- `output_attentions`이 `True`인 경우 attention score도 함께 반환

---

## FNDNet


> fndnet(*hparams*)

Parameters:

- **hparams** (*dict*) - 모델 학습에 필요한 hyper-parameters. [FNDNet configuration](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/configs/FNDNet/FNDNet-train.yaml)

Returns:

`FNDNet`



> class FNDNet(*dims, num_classes, dropout, vocab_len, embed_dims*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/models/fndnet.py#L10)]

[FNDNet–a deep convolutional neural network for fake news detection](https://www.sciencedirect.com/science/article/pii/S1389041720300085)에서 제안한 FNDNet 모델 사용. 모델 설명은 [여기](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/model_description/FNDNet.md)에서 확인할 수 있습니다.


Parameters:

- **dims** (*int, optional*) - convolution에 사용할 dimension 크기. Default = 128
- **num_classes** (*int, optional*) - 학습할 class 수. Default = 2
- **dropout** (*float, optional*) - dropout에 사용할 확률. Default = 0.2
- **vocab_len** (*int, optional*) - embedding weight matrix를 위한 vocabulary 크기. Default = 58043
- **embed_dims** (*int, optional*) - embedding weight matrix에 사용할 embedding dimensions. Default = 100


> > init_w2e(*weights, nb_special_tokens*)

word embedding weights 초기화

Parameters:

- **weights** (*np.ndarray*) - 학습된 word embedding weights
- **nb_special_tokens** (*int*) - 추가한 special token 개수

> > freeze_w2e()

word embedding weights를 freeze 할지 여부

> > forward(*input_ids*)

Parameters:

- **input_ids** (*torch.Tensor*) - 입력 token ids

Returns:

- **out** (*torch.Tensor*) - $\hat{y} \in \mathbf{R}^{batch size \times 2}$

---

## HAND

> hand(*hparams*)

Parameters:

- **hparams** (*dict*) - 모델 학습에 필요한 hyper-parameters. [HAND configuration](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/configs/HAND/HAND-train.yaml)

Returns:

`HAND`


> class HierAttNet(*word_dims, sent_dims, dropout, num_classes, vocab_len, embed_dims) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/models/hand.py#L11)]

[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)에서 제안한 HAND 모델 사용. 모델 설명은 [여기](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/assets/model_description/HAND.md)에서 확인할 수 있습니다.


Parameters:

- **word_dims** (*int, optional*) - word attention network에서 사용할 GRU의 embedding 크기. Default = 64
- **sent_dims** (*int, optional*) - sentence attention network에서 사용할 GRU의 embedding 크기. Default = 128
- **num_classes** (*int, optional*) - 학습할 class 수. Default = 2
- **dropout** (*float, optional*) - dropout에 사용할 확률. Default = 0.2
- **vocab_len** (*int, optional*) - embedding weight matrix를 위한 vocabulary 크기. Default = 58043
- **embed_dims** (*int, optional*) - embedding weight matrix에 사용할 embedding dimensions. Default = 100


> > init_w2e(*weights, nb_special_tokens*)

word embedding weights 초기화

Parameters:

- **weights** (*np.ndarray*) - 학습된 word embedding weights
- **nb_special_tokens** (*int*) - 추가한 special token 개수

> > freeze_w2e()

word embedding weights를 freeze 할지 여부

> > forward(*input_ids, output_attentions*)

Parameters:

- **input_ids** (*torch.Tensor*) - 입력 token ids
- **output_attentions** (*bool*) - `True`인 경우 단어와 문장 기준 attention score 를 함께 반환

Returns:

- **out** (*torch.Tensor*) - $\hat{y} \in \mathbf{R}^{batch size \times 2}$
- `output_attentions`이 `True`인 경우 단어와 문장 기준 attention score 를 함께 반환
 


> class WordAttnNet(*vocab_len, embed_dims, word_dims, dropout*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/models/hand.py#L63)]


Parameters:

- **vocab_len** (*int*) - embedding weight matrix를 위한 vocabulary 크기
- **embed_dims** (*int*) - embedding weight matrix에 사용할 embedding dimensions
- **word_dims** (*int*) - word attention network에서 사용할 GRU의 embedding 크기
- **dropout** (*float*) - dropout에 사용할 확률


> > forward(*input_ids*)

Parameters:

- **input_ids** (*torch.Tensor*) - 입력 token ids

Returns:

- **words_embed** (*torch.Tensor*) - 단어 기준 embedding output
- **words_attn_score** (*torch.Tensor*) - 단어 기준 attention score


> class SentAttnNet(*word_dims, sent_dims, dropout*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/models/hand.py#L98)]

Parameters:

- **word_dims** (*int*) - word attention network에서 사용할 GRU의 embedding 크기
- **sent_dims** (*int*) - sentence attention network에서 사용할 GRU의 embedding 크기
- **dropout** (*float*) - dropout에 사용할 확률

> > forward(*words_embed*)
 
Parameters:

- **words_embed** (*torch.Tensor*) - 단어 기준 embedding output

Returns:

- **sents_embed** (*torch.Tensor*) - 문장 기준 embedding output
- **sents_attn_score** (*torch.Tensor*) - 문장 기준 attention score



---

## factory

> create_model(*modelname, hparams, word_embed, tokenizer, freeze_word_embed, use_pretrained_word_embed, checkpoint_path*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part1_title/models/factory.py#L8)]

Parameters:

- **modelname** (*str*) - 사용할 모델 이름. ex) `HAND`, `FNDNet`, 또는 `BERT`
- **hparams** (*dict*) - 모델에 필요한 hyper-parameters
- **word_embed** (*np.ndarray*) - 사전학습된 word embedding weights
- **tokenizer** - `FNDTokenizer` 또는 `BERTSPTokenizer`
- **freeze_word_embed** (*bool*) - word embedding weights를 freeze 할지 여부
- **use_pretrained_word_embed** (*bool*) - 사전학습된 word embedding weights를 사용할지 여부
- **checkpoint_path** (*str*) - 사전학습된 모델 경로
  
Returns:

`Model`