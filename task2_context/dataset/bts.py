from .build_dataset import FakeDataset
import torch

class BTSDataset(FakeDataset):
    def __init__(self, window_size, tokenizer, vocab, max_word_len=512):
        super(BTSDataset, self).__init__(
            tokenizer    = tokenizer, 
            vocab        = vocab, 
            window_size  = window_size,
            max_word_len = max_word_len
        )
    
    def single_preprocessor(self, doc):
        datasets = self._single_preprocessor(doc)

        inputs = {
            'src': [],
            'segs': [],
            'mask_src': []
        }

        # tokenizer
        for dataset in datasets:
            src_subtoken_idxs, segments_ids, _, mask_src, _ = self.tokenize(dataset)

            inputs['src'].append(src_subtoken_idxs)
            inputs['segs'].append(segments_ids)
            inputs['mask_src'].append(mask_src)

        for k, v in inputs.items():
            inputs[k] = torch.stack(v)
        
        return inputs

    def tokenize(self, src):
        # length
        src = self.length_processing(src)

        src_subtokens = [[self.vocab.cls_token] + src[0] + [self.vocab.sep_token]] + [src[1] + [self.vocab.sep_token]]
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

    
    def __getitem__(self, i, return_txt=False, return_fake_label=False):
        
        doc, target, news_id = self.datasets[i], self.targets[i], self.news_ids[i]
        
        # tokenizer
        src_subtoken_idxs, segments_ids, _, mask_src, _ = self.tokenize(doc)

        inputs = {
            'src': src_subtoken_idxs,
            'segs': segments_ids,
            'mask_src': mask_src,
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