---
layout: default
title: Part 2. 본문 내 일관성
description: 본문 내 내용 간 일관성 여부를 판단하여 가짜 뉴스를 탐지하는 task 입니다.
---

# Dataset

## FakeDataset


> class FakeDataset(*tokenizer, vocab, window_size, max_word_len*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/dataset/build_dataset.py)]

낚시성 데이터 주제 분리 탐지를 위한 base Dataset

Parameters:

- **tokenizer** -  SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **vocab** - SKT에서 학습한 BERTSPTokenizer의 vocab
- **window_size** (*int*) - 주제 분리 탐지를 데이터셋 구성을 위한 문장 개수 (ex. `window_size`가 3인 경우 중간을 기준으로 3개씩 총 6문장이 하나의 데이터로 구성)
- **max_word_len** (*int, optional*) - 입력 token 최대 개수. Default: 512


> > load_dataset(*datadir, split*)

학습 전 데이터 경로와 파일을 미리 불러오는 함수

Parameters:

- **datadir** (*str*) - 데이터 폴더 경로. ex) `../data/Part2`
- **split** (*str*) - 데이터셋 이름. ex) `train`, `validation`, 또는 `test`

> > preprocessor()

학습을 위한 데이터 구조로 변환하는 함수. 기존 뉴스 문서를 사전에 정의한 `window_size`에 맞추어 문장 단위로 데이터를 구성한 후 labeling 수행


> > split_doc_into_sents(*doc, src, fake_label, window_size*)

모든 뉴스 본문을 받아온 후 각 문서를 `window_size`로 나누어 데이터 구성. Label 또한 구성된 데이터에 맞춰서 구성.

Parameters:

- **doc** (*list*) - 모든 뉴스 기사별 본문
- **src** (*list*) - tokenizer에 의해 처리된 모든 뉴스 기사별 본문의 tokens
- **fake_label** (*list*) - 모든 뉴스 기사별 본문에 대한 label. 본문 내 문장 마다 주제와 벗어나는 경우 1 아닌경우 0으로 labeling 되어 있음. ex. [[0,0,1,1], ...]
- **window_size** (*int*) - 주제 분리 탐지를 데이터셋 구성을 위한 문장 개수 (ex. `window_size`가 3인 경우 중간을 기준으로 3개씩 총 6문장이 하나의 데이터로 구성)


Returns:

- **datasets** (*list*) - 문서 마다 `window_size`에 따라 문장별로 구성된 tokens
- **targets** (*list*) - 문서 마다 `window_size`에 따라 문장별로 구성된 데이터에 대한 target label. 중간에 주제가 변하는 경우 1, 아닌경우 0. ex) [0,0,1,1] -> 1
- **docs** (*list*) - 문서 마다 `window_size`에 따라 구성된 문장을 합친 text
- **fake_labels** (*list*) - 문서 마다 `window_size`에 따라 문장별로 구성된 labels



> > length_processing(*src*)

입력 데이터의 token 개수가 `max_word_len`을 넘는 경우 뒤에서부터 자르는 것이 아닌 문장별 최대 길이를 제한하여 처리

Parameters:

- **src** (*list*) - `preprocessor`를 거친 모든 뉴스기사 데이터에 대한 tokens


Returns:

- **src** (*list*) - 길이 제한 처리된 tokens

> > pad(*data, pad_idx*)

`max_word_len` 보다 작은 입력 데이터에 대한 padding

Parameters:

- **data** (*list*) - 입력 데이터 tokens list
- **pad_idx** (*int*) - vocab의 padding token id


Returns:

- **data** (*list*) - padding 처리된 입력 데이터 tokens list

> > padding_bert(*src_token_ids, segments_ids, cls_ids*)

KoBERT의 모든 입력 데이터에 대한 padding 함수

Parameters:

- **src_token_ids** (*list*) - 입력 데이터로 사용된 문서에 대한 token ids
- **segments_ids** (*list*) - 입력 데이터로 사용된 segment ids 
- **cls_ids** (*list*) - 입력 데이터에 포함된 cls token에 대한 index 위치


Returns:

- **src** (*list*) - padding 처리된 token ids
- **segments_ids** (*list*) - padding 처리된 segment ids
- **cls_ids** (*list*) - padding 처리된 cls token에 대한 index 위치
- **mask_src** (*list*) - token ids 이외에 padding이 된 부분에 대한 mask
- **mask_cls** (*list*) - cls token에 대한 mask

> > get_token_type_ids(*src_token*)

입력 데이터 내 문장 별 segment ids를 계산하기 위한 함수

Parameters:

- **src_token** (*list*) - 입력 데이터로 사용된 문서에 대한 token ids

Returns:

- **seg** (*list*) - 입력 `src_token`에 대한 문장별 segment ids

> > get_cls_index(*src_doc*)

입력 데이터 내 cls token의 위치를 나타내는 cls index 함수

Parameters:

- **src_doc** (*list*) - 입력 데이터로 사용된 문서에 대한 token ids


Returns:

- **cls_index** (*list*) - 입력 `src_token`에 대한 문장별 cls token 위치를 나타내는 cls index

---

## BTSDataset

> class BTSDataset(*window_size, tokenizer, vocab, max_word_len) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/dataset/bts.py#L6)]

`BTS` 모델을 위한 데이터셋

Parameters:

- **tokenizer** -  SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **vocab** - SKT에서 학습한 BERTSPTokenizer의 vocab
- **window_size** (*int*) - 주제 분리 탐지를 데이터셋 구성을 위한 문장 개수 (ex. `window_size`가 3인 경우 중간을 기준으로 3개씩 총 6문장이 하나의 데이터로 구성)
- **max_word_len** (*int, optional*) - 입력 token 최대 개수. Default: 512

> > sing_preprocessor(*doc*)

하나의 문서에 대한 데이터 구성을 위한 전처리 함수

Parameters:

- **doc** (*list*) - 하나의 문서 내 문장 list


Returns:

- **inputs** (*dict*) - 전처리가 끝난 데이터의 token index를 나타내는 `src`, 문장 단위 구분을 위한 `segs`, 그리고 padding 된 부분을 고려하기 위한 `mask_src`가 포함된 입력 데이터

> > tokenize(*src*)

입력 데이터 구성을 위한 데이터 전처리 함수

Parameters:

- **src** (*list*) - `preprocessor`를 거친 모든 뉴스기사 데이터에 대한 tokens


Returns:

- **src** (*list*) - padding 처리된 token ids
- **segments_ids** (*list*) - padding 처리된 segment ids
- **mask_src** (*list*) - token ids 이외에 padding이 된 부분에 대한 mask


> > \_\_getitem\_\_(*i, return_txt, return_fake_label*)

Parameters:

- **i** (*int*) - 선택하고자 하는 index
- **return_txt** (*bool*) - 원본 text 반환 시 `True`
- **return_fake_label** (*bool*) - 문장 별 fake 여부를 나타내는 label 반환 시 `True`


Returns:

- **return_values** (*tuple*) - 전처리가 끝난 데이터의 token index를 나타내는 `src`, 문장 단위 구분을 위한 `segs`, 그리고 padding 된 부분을 고려하기 위한 `mask_src`가 포함된 입력 데이터. `return_txt` 또는 `return_fake_label`이 `True`인 경우 해당하는 결과값 함께 반환


---

## KoBERTSegDataset

> class KoBERTSegDataset(*window_size, tokenizer, vocab, max_word_len*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/5856b56f39ca104157550c8108435d2d2cc84f3f/part2_context/dataset/kobertseg.py#L6)]

`KoBERTSeg` 모델을 위한 데이터셋

Parameters:

- **tokenizer** -  SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **vocab** - SKT에서 학습한 BERTSPTokenizer의 vocab
- **window_size** (*int*) - 주제 분리 탐지를 데이터셋 구성을 위한 문장 개수 (ex. `window_size`가 3인 경우 중간을 기준으로 3개씩 총 6문장이 하나의 데이터로 구성)
- **max_word_len** (*int, optional*) - 입력 token 최대 개수. Default: 512

> > single_preprocessor(*doc*)

하나의 문서에 대한 데이터 구성을 위한 전처리 함수

Parameters:

- **doc** (*list*) - 하나의 문서 내 문장 list

Returns:

- **inputs** (*dict*) - 전처리가 끝난 데이터의 token index를 나타내는 `src`, 문장 단위 구분을 위한 `segs`, cls token 위치를 나타내는 `clss`, padding 된 부분을 고려하기 위한 `mask_src`, 그리고 cls 가 포함된 입력 데이터
  

> > tokenize(*src*)

입력 데이터 구성을 위한 데이터 전처리 함수

Parameters:

- **src** (*list*) - `preprocessor`를 거친 모든 뉴스기사 데이터에 대한 tokens

Returns:

- **src** (*list*) - padding 처리된 token ids
- **segments_ids** (*list*) - padding 처리된 segment ids
- **cls_ids** (*list*) - padding 처리된 cls token에 대한 index 위치
- **mask_src** (*list*) - token ids 이외에 padding이 된 부분에 대한 mask
- **mask_cls** (*list*) - cls token에 대한 mask

> > \_\_getitem\_\_(*i, return_txt, return_fake_label*)

Parameters:

- **i** (*int*) - 선택하고자 하는 index
- **return_txt** (*bool*) - 원본 text 반환 시 `True`
- **return_fake_label** (*bool*) - 문장 별 fake 여부를 나타내는 label 반환 시 `True`


Returns:

- **return_values** (*tuple*) - 전처리가 끝난 데이터의 token index를 나타내는 `src`, 문장 단위 구분을 위한 `segs`, 그리고 padding 된 부분을 고려하기 위한 `mask_src`가 포함된 입력 데이터. `return_txt` 또는 `return_fake_label`이 `True`인 경우 해당하는 결과값 함께 반환


---


## factory

> create_dataset(*name, data_path, split, tokenizer, vocab, kwargs*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/dataset/factory.py#L3)]

Parameters:

- **name** (*str*) - 사용할 데이터셋 이름. ex) `BTS` 또는 `KoBERTSeg`
- **data_path** (*str*) - 사용할 데이터 경로. ex) `../data/Part2`
- **split** (*str*) - 데이터셋 이름. ex) `train`, `validation`, 또는 `test`
- **tokenizer** -  SKT에서 학습한 [tokenizer](https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils/utils.py)를 통해 gluonnlp의 [BERTSPTokenizer](https://nlp.gluon.ai/_modules/gluonnlp/data/transforms.html#BERTSPTokenizer)
- **vocab** - SKT에서 학습한 BERTSPTokenizer의 vocab
- **kwargs** (*dict*) - 사용할 데이터셋에 대한 parameters


Returns:

`Dataset`


> create_dataloader(*dataset, batch_size, num_workers, shuffle*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/dataset/factory.py#L16)]

Parameters:

- **dataset** - `create_dataset`에서 반환한 Dataset
- **batch_size** (*int*) - 입력 데이터의 batch size
- **num_workers** (*int*) - 사용할 worker 수
- **shuffle** (*bool*) - random하게 dataset의 index를 반환할지에 대한 여부


Returns:

`DataLoader`

---



# Models


## BTS

> bts(*hparams*)

Parameters:

- **hparams** (*dict*) - 모델 학습에 필요한 hyper parameters. [BTS configuration](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/configs/BTS/BTS-train.yaml) 참고 

Returns:

`BTS`

> class BTS(*finetune_bert*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/models/bts.py#L11)]

[BTS: 한국어 BERT를 사용한 텍스트 세그멘테이션](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE09301605&mark=0&useDate=&ipRange=N&accessgl=Y&language=ko_KR&hasTopBanner=true)에서 제안한 BTS 모델 사용. 모델 설명은 [여기](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/assets/model_description/BTS.md)에서 확인할 수 있습니다.

Parameters:

- **finetung_bert** (*bool*) - BERT 모델을 finetuning 할지에 대한 여부 


> > forward(*src, segs, mask_src*)

Parameters:

- **src** (*torch.Tensor*) - padding 처리된 token ids
- **segs** (*torch.Tensor*) - padding 처리된 segment ids
- **mask_src** (*torch.Tensor*) - token ids 이외에 padding이 된 부분에 대한 mask

Returns:

- **output** (*torch.Tensor*) - $\hat{y} \in \mathbf{R}^{batch size \times 2}$

---

## KoBERTSeg

> kobertseg(*hparams*)

Parameters

- **hparams** (*dict*) - 모델 학습에 필요한 hyper parameters. [KoBERTSeg configuration](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/configs/KoBERTSeg/KoBERTSeg-train.yaml) 참고 
  
Returns:

`KoBERTSeg`


> class KoBERTSeg(*finetune_bert, window_size*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/models/kobertseg.py#L37)]

[KoBERTSEG: 한국어 BERT를 이용한 Local Context 기반 주제 분리 방법론](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002831197)에서 제안한 KoBERTSeg 사용. 모델 설명은 [여기](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/assets/model_description/KoBERTSeg.md)에서 확인할 수 있습니다.

Parameters:

- **finetune_bert** (*bool*) - BERT 모델을 finetuning 할지에 대한 여부 
- **window_size** (*int, optional*) - Conv1D를 위한 `kernal size`. Default = 3

> > forward(*src, segs, clss, mask_src, mask_cls*)

Parameters:

- **src** (*torch.Tensor*) - padding 처리된 token ids
- **segs** (*torch.Tensor*) - padding 처리된 segment ids
- **clss** (*torch.Tensor*) - padding 처리된 cls token에 대한 index 위치
- **mask_src** (*torch.Tensor*) - token ids 이외에 padding이 된 부분에 대한 mask
- **mask_cls** (*torch.Tensor*) - cls token에 대한 mask


Returns:

- **classified** (*torch.Tensor*) - $\hat{y} \in \mathbf{R}^{batch size \times 2}$


---


## factory

> create_model(*modelname, hparams, checkpoint_path*) [[SOURCE](https://github.com/TooTouch/Fake-News-Detection-Dataset/blob/master/part2_context/models/factory.py#L8)]

모델 생성을 위한 함수

Parameters:

- **modelname** (*str*) - 사용할 모델 이름. ex) `BTS` 또는 `KoBERTSeg`
- **hparams** (*dict*) - 사용할 모델의 hyper-parameters
- **checkpoint_path** (*str*) - 학습된 모델의 저장 경로


Returns:

`Model`
