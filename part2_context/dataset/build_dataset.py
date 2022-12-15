from torch.utils.data import Dataset
import json 
import numpy as np
import torch
import os
from glob import glob

from tqdm.auto import tqdm
from typing import List

class FakeDataset(Dataset):
    def __init__(self, tokenizer, vocab, window_size: int, max_word_len: int = 512):
        
        # data parameters
        self.window_size = window_size
        self.max_word_len = max_word_len
        
        # vocab and tokenizer
        self.vocab = vocab
        self.tokenizer = tokenizer
        
        # special token index
        self.pad_idx = self.vocab[self.vocab.padding_token]
        self.cls_idx = self.vocab[self.vocab.cls_token]
        
    def load_dataset(self, datadir: str, split: str):        
        data_info = glob(os.path.join(datadir, split, '*/*/*'))

        data = {}
        for filename in data_info:
            f = json.load(open(filename,'r'))
            data[filename] = f

        setattr(self, 'data', data)
        setattr(self, 'data_info', data_info)

    
    def _single_preprocessor(self, doc: list) -> list:
        # tokenizing
        src = [self.tokenizer(d_i) for d_i in doc]
        pad_sents = [[self.vocab.padding_token]] * (self.window_size-1)
        src = pad_sents + src + pad_sents

        return self._split_doc_into_sents(src, self.window_size)

    def _split_doc_into_sents(self, src: list, window_size: int) -> list:
        datasets = []

        for i in range(0, len(src)-(window_size*2)+1):
            # slicing source sentences and label by total window size
            total_window = window_size*2

            src_i = src[i:i+total_window]

            # inputs
            datasets.append(src_i)
            
        return datasets

    def preprocessor(self):
        datasets = [] 
        targets = []
        docs = []
        fake_labels = []
        news_ids = []

        for file_path in tqdm(self.data_info, total=len(self.data_info), desc='Preprocessing'):
            # extract news contents
            news_info = self.data[file_path]
            doc = [sent_info['sentenceContent'] for sent_info in news_info['labeledDataInfo']['processSentenceInfo']]

            # define fake label
            fake_label = np.zeros(len(doc) + (self.window_size-1)*2)

            if 'NonClickbait_Auto' not in file_path:
                fake_idx = [ 
                    d['sentenceNo'] - 1
                    for d in news_info['labeledDataInfo']['processSentenceInfo'] 
                    if d['subjectConsistencyYn'] == 'N'
                ]
                fake_label[fake_idx[0] + (self.window_size-1):] = 1

            # doc to save
            doc_i = ['blank.'] * (self.window_size-1) + doc + ['blank.'] * (self.window_size-1)

            # tokenizing
            src = [self.tokenizer(d_i) for d_i in doc]

            pad_sents = [[self.vocab.padding_token]] * (self.window_size-1)
            src = pad_sents + src + pad_sents

            # split doc into sents by windwo size
            dataset_i, target_i, doc_i, fake_label_i = self.split_doc_into_sents(
                doc         = doc_i,
                src         = src, 
                fake_label  = fake_label, 
                window_size = self.window_size
            )

            # stack
            datasets.extend(dataset_i)
            targets.extend(target_i)
            docs.extend(doc_i)
            fake_labels.extend(fake_label_i)
            news_ids.extend([file_path]*len(dataset_i))

        setattr(self, 'datasets', datasets)
        setattr(self, 'targets', targets)
        setattr(self, 'docs', docs)
        setattr(self, 'fake_labels', fake_labels)
        setattr(self, 'news_ids', news_ids)

    def split_doc_into_sents(self, doc: list, src: list, fake_label: list, window_size: int) -> List[list]:
        datasets = []
        targets = []
        docs = []
        fake_labels = []

        for i in range(0, len(src)-(window_size*2)+1):
            # slicing source sentences and label by total window size
            total_window = window_size*2

            src_i = src[i:i+total_window]
            doc_i = doc[i:i+total_window]

            # fake
            # if [0,0,0,1,1,1], then 1 else 0
            fake_label_i = fake_label[i:i+total_window]
            target_i = 1 if fake_label_i[:window_size].mean() == 0 and fake_label_i[window_size:].mean() == 1 else 0

            # inputs
            datasets.append(src_i)
            targets.append(target_i)

            # ground truth
            docs.append('\n'.join(doc_i))
            fake_labels.append(fake_label_i)
            
        return datasets, targets, docs, fake_labels
    
    def tokenize(self):
        raise NotImplementedError
    
    def length_processing(self, src: list) -> list:
        # 문장별 최대 길이
        # 최대 길이가 512일 경우 각 문장이 window size에 따라 평균 길이를 넘지 않도록 함 (뒤에 잘리지 않도록)
        
        special_token_num = (self.window_size * 2) * 2 # [cls], [sep] for each sentence
        avg_length = (self.max_word_len - special_token_num) // (self.window_size * 2)
        
        src = [sent[:avg_length] for sent in src]
        return src

    def pad(self, data: list, pad_idx: int) -> list:
        data = data + [pad_idx] * max(0, (self.max_word_len - len(data)))
        return data
    
    def padding_bert(self, src_token_ids: list, segments_ids: list, cls_ids: list) -> List[torch.Tensor]:
        # padding using bert models (bts, kobertseg)        
        src = torch.tensor(self.pad(src_token_ids, self.pad_idx))
        seg_ids = torch.tensor(self.pad(segments_ids, self.pad_idx))
        clss = torch.tensor(cls_ids)

        mask_src = ~(src == self.pad_idx)
        mask_cls = ~(clss == -1)

        return src, seg_ids, clss, mask_src, mask_cls

    def get_token_type_ids(self, src_token: list) -> list:
        # for segment token
        seg = []
        for i, v in enumerate(src_token):
            if i % 2 == 0:
                seg.append([0] * len(v))
            else:
                seg.append([1] * len(v))
        return seg

    def get_cls_index(self, src_doc: list) -> list:
        # for cls token
        cls_index = [index for index, value in enumerate(src_doc) if value == self.cls_idx]
        return cls_index
    
    def __getitem__(self, i):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError