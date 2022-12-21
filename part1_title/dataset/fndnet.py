from .build_dataset import FakeDataset

import torch

class FNDNetDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len: int):
        super(FNDNetDataset, self).__init__(tokenizer=tokenizer)
        self.max_word_len = max_word_len

    def transform(self, title: str, text: list) -> dict:
        sent_list = [title] + text

        doc = sum([self.tokenizer.encode(sent) for sent in sent_list], [])[:self.max_word_len]

        doc = self.padding(doc)

        doc = {'input_ids':torch.tensor(doc)}

        return doc 

    def padding(self, doc: list) -> list:
        num_pad_word = max(0, self.max_word_len - len(doc))
        doc = doc + [self.tokenizer.pad_token_id] * num_pad_word

        return doc


    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)

