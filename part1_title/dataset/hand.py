from .build_dataset import FakeDataset

import torch

class HANDDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len: int, max_sent_len: int):
        super(HANDDataset, self).__init__(tokenizer=tokenizer)

        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len

    def transform(self, title: str, text: list) -> dict:
        sent_list = [title] + text

        sent_list = sent_list[:self.max_sent_len]
        doc = [self.tokenizer.encode(sent)[:self.max_word_len] for sent in sent_list] 
        
        doc = self.padding(doc)

        doc = {'input_ids':torch.tensor(doc)}

        return doc
    
    def padding(self, doc: list) -> list:
        num_pad_doc = self.max_sent_len - len(doc)
        num_pad_sent = [max(0, self.max_word_len - len(sent)) for sent in doc]

        doc = [sent + [self.tokenizer.pad_token_id] * num_pad_sent[idx] for idx, sent in enumerate(doc)]
        doc = doc + [[self.tokenizer.pad_token_id] * self.max_word_len for i in range(num_pad_doc)]
            
        return doc


    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)

