from .build_dataset import FakeDataset
 
import torch
import os


class BTSDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len, saved_data_path=False):
        super(BTSDataset, self).__init__(tokenizer=tokenizer)

        self.max_word_len = max_word_len

        # load data
        self.saved_data_path = saved_data_path

    def transform(self, title, text):
        doc = self.tokenizer(
            title,
            ' '.join(text),
            return_tensors     = 'pt',
            max_length         = self.max_word_len,
            padding            = 'max_length',
            truncation         = True,
            add_special_tokens = True
        )

        doc['input_ids'] = doc['input_ids'][0]
        doc['attention_mask'] = doc['attention_mask'][0]
        doc['token_type_ids'] = doc['token_type_ids'][0]

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
    



