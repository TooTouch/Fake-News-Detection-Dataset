---
layout: default
title: Part 1. 제목 - 본문 일치성
description: 제목과 본문 간 일치 여부에 따라 가짜 뉴스를 탐지하는 task 입니다.
---

# Dataset


## FakeDataset


> class FakeDataset(*tokenizer*) [[SOURCE]()]

Fake News Detection Base Dataset

Parameters:

- **tokenizer**:

> > load_dataset(*data_dir, split, saved_data_path=False)

데이터셋 불러오기

Parameters:

- **data_dir**:
- **split**:
- **saved_data_path**:


---

## FNDNetDataset

> class FNDNetDataset(*tokenizer, max_word_len*) [[SOURCE]()]

Parameters:

- **tokenizer**:
- **max_word_len**:


> > transform(*title, text*)

Parameters:

- **title**:
- **text**:

Returns:



> > padding(*doc*)

Parameters:

- **doc**:

Returns:


---


## HANDataset

> class HANDataset(*tokenizer, max_word_len, max_sent_len*) [[SOURCE]()]

Parameters:

- **tokenizer**:
- **max_word_len**:
- **max_sent_len**:


> > transform(*title, text*)

Parameters:

- **title**:
- **text**:

Returns:



> > padding(*doc*)

Parameters:

- **doc**:

Returns:



---

## BERTDataset

> class BERTDataset(*tokenizer, max_word_len*) [[SOURCE]()]

BERT 모델을 위한 Fake News Detection Dataset

Parameters:

- **tokenizer**:

> > transform(*title, text*)

Parameters:

- **title**:
- **text**:


Returns:

- **doc**

> > tokenize(*src*)

Parameters:

- **src**:

Returns:

- **input_ids**:
- **token_type_ids**:
- **attention_mask**:


> > length_precessing(*src*)

Parameters:

- **src**

Returns:

- **processed_src**

> > pad(*data, pad_idx*)

Parameters:

- **data**:
- **pad_idx**:

Returns:

- **data**

> > padding_bert(*input_ids, token_type_ids*)

Parameters:

- **input_ids**:
- **token_type_ids**:


Returns:


> > get_token_type_ids(*input_ids*)


Parameters:

- **input_ids**

Returns:


---

## FNDTokenizer

> FNDTokenizer(*vocab, tokenizer, special_tokens*)


Parameters:

- **vocab**:
- **tokenizer**:
- **special_tokens**:


> > encode(*sentence*)

Parameters:

- **sentence**:

Returns:


> > batch_encode(*b_sentence*)

Parameters:

- **b_sentence*

Returns:


> > decode(*input_ids*)

Parameters:

- **input_ids**:

Returns:



> > batch_decode(*b_input_ids*)

Parameters:

- **b_input_ids**:

Returns:


> > add_tokens(*name*)

Parameters:

- **name**:




---

## factory

> extract_word_embedding(*vocab_path, max_vocab_size*)

Parameters:

- **vocab_path**:
- **max_vocab_size**:

Returns:


> create_tokenizer(*name, vocab_path, max_vocab_size*)

Parameters:

- **name**:
- **vocab_path**:
- **max_vocab_size**:

Returns:



> create_dataset(*name, data_path, split, tokenizer, saved_data_path, kwargs*)

Parameters:

- **name**:
- **data_path**:
- **split**:
- **tokenizer**:
- **saved_data_path**:
- **kwargs**:

Returns:



> create_dataloader(*dataset, batch_size, num_workers, shuffle*)

Parameters:

- **dataset**:
- **batch_size**:
- **num_workers**:
- **shuffle**:

Returns:


---

# Models

## BERT

> bert(*hparams*)

Parameters:

- **hparams**:


Returns:

- [BERT]()

> class BERT(*pretrained_name, config, num_classes*)

Parameters:

- **pretrained_name**:
- **config**:
- **num_classes**:
  
> > forward(*input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attention, output_hidden_states*)

Parameters:

- **input_ids**:
- **attention_mask**:
- **token_type_ids**:
- **position_ids**:
- **head_mask**:
- **inputs_embeds**:
- **output_attention**:
  
Returns:

---

## FNDNet


> fndnet(*hparams*)

Parameters:

- **hparams**:

Returns:

[FNDNet]()



> class FNDNet(*dims, num_classes, dropout, vocab_len, embed_dims*)

Parameters:

- **dims**:
- **num_classes**:
- **dropout**:
- **vocab_len**:
- **embed_dims**:

Returns:


> > init_w2e(*weights, nb_special_tokens*)

Parameters:

- **weights**:
- **nb_special_tokens**:


> > freeze_w2e()


> > forward(*input_ids*)

Parameters:

- **input_ids**:

Returns:


---

## HAN

> han(*hparams*)

Parameters:

- **hparams**:

Returns:

[HAN]()


> class HierAttNet(*word_dims, sent_dims, dropout, num_classes, vocab_len, embed_dims)

Parameters:

- **word_dims**:
- **sent_dims**:
- **dropout**:
- **num_classes**:
- **vocab_len**:
- **embed_dims**:


> > init_w2e(*weights, nb_special_tokens*)

Parameters:

- **weights**:
- **nb_special_tokens**:
  

> > freeze_w2e()


> > forward(*input_ids, output_attentions*)

Parameters:

- **input_ids**:
- **output_attentions**:


Returns:




> class WordAttnNet(*vocab_len, embed_dims, word_dims, dropout*)

Parameters:

- **vocab_len**:
- **embed_dims**:
- **word_dims**:
- **dropout**:


> > forward(*input_ids*)

Parameters:

- **input_ids**:

Returns:




> class SentAttnNet(*word_dims, sent_dims, dropout*)

Parameters:

- **word_dims**:
- **sent_dims**:
- **dropout**:

> > forward(*words_embed*)
 
Parameters:

- **words_embed** 


Returns:




---

## factory

> create_model(*modelname, hparams, word_embed, tokenizer, freeze_word_embed, use_pretrained_word_embed, checkpoint_path*)

Parameters:

- **modelname**:
- **hparams**:
- **word_embed**:
- **tokenizer**:
- **freeze_word_embed**:
- **use_pretrained_word_embed**:
- **checkpoint_path**:
  
Returns:
