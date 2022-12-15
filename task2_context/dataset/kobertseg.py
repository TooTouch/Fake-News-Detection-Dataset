from .build_dataset import FakeDataset
import torch

from typing import List

class KoBERTSegDataset(FakeDataset):
    def __init__(self, window_size: int, tokenizer, vocab, max_word_len: int = 512):
        super(KoBERTSegDataset, self).__init__(
            tokenizer    = tokenizer, 
            vocab        = vocab, 
            window_size  = window_size,
            max_word_len = max_word_len
        )
    
    def single_preprocessor(self, doc: list) -> dict:
        datasets = self._single_preprocessor(doc)

        inputs = {
            'src': [],
            'segs': [],
            'clss': [],
            'mask_src': [],
            'mask_cls': [],
        }

        # tokenizer
        for dataset in datasets:
            src_subtoken_idxs, segments_ids, cls_ids, mask_src, mask_cls = self.tokenize(dataset)

            inputs['src'].append(src_subtoken_idxs)
            inputs['segs'].append(segments_ids)
            inputs['clss'].append(cls_ids)
            inputs['mask_src'].append(mask_src)
            inputs['mask_cls'].append(mask_cls)

        for k, v in inputs.items():
            inputs[k] = torch.stack(v)
        
        return inputs

    def tokenize(self, src: list) -> List[torch.Tensor]:
        # length
        src = self.length_processing(src)

        src_subtokens = [[self.vocab.cls_token] + sent + [self.vocab.sep_token] for sent in src]
        src_token_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in src_subtokens]
        
        segments_ids = self.get_token_type_ids(src_token_ids)
        segments_ids = sum(segments_ids,[])
        
        src_token_ids = [x for sublist in src_token_ids for x in sublist]
        cls_ids = self.get_cls_index(src_token_ids)
        
        src_token_ids, segments_ids, cls_ids, mask_src, mask_cls = self.padding_bert(
            src_token_ids = src_token_ids,
            segments_ids  = segments_ids,
            cls_ids       = cls_ids
        )
        
        return src_token_ids, segments_ids, cls_ids, mask_src, mask_cls

    
    def __getitem__(self, i: int, return_txt: bool = False, return_fake_label: bool = False):
        
        doc, target, news_id = self.datasets[i], self.targets[i], self.news_ids[i]
        doc_txt = doc
        
        # tokenizer
        src_subtoken_idxs, segments_ids, cls_ids, mask_src, mask_cls = self.tokenize(doc_txt)

        inputs = {
            'src': src_subtoken_idxs,
            'segs': segments_ids,
            'clss': cls_ids,
            'mask_src': mask_src,
            'mask_cls': mask_cls,
        }

        return_values = (inputs, target, news_id)

        if return_txt:
            src_txt = self.docs[i]
            return_values = return_values + (src_txt,)

        if return_fake_label:
            src_fake_label = self.fake_labels[i]
            return_values = return_values + (src_fake_label,)
        
        return return_values
    
    def __len__(self):
        return len(self.datasets)