from .build_dataset import FakeDataset

import torch
import os


class FNDNetDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len, saved_data_path=False):
        super(FNDNetDataset, self).__init__(tokenizer=tokenizer)
        self.max_word_len = max_word_len

        # load data
        self.saved_data_path = saved_data_path

    def transform(self, title, text):
        sent_list = [title] + text

        doc = sum([self.tokenizer.encode(sent) for sent in sent_list], [])[:self.max_word_len]

        doc = self.padding(doc)

        doc = {'input_ids':torch.tensor(doc)}

        return doc 

    def padding(self, doc):
        num_pad_word = max(0, self.max_word_len - len(doc))
        doc = doc + [self.tokenizer.pad_token_id] * num_pad_word

        return doc

    def __getitem__(self, i):

        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]

            return doc, label
        
        else:
            news_idx = self.data_info.iloc[i]
            news_info = self.data[str(news_idx['id'])]
        
            # label
            label = 1 if news_idx['label']=='fake' else 0
            
            # transform and padding
            doc = self.transform(
                title = news_info['title'], 
                text  = news_info['text']
            )

            
            return doc, label


    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)

