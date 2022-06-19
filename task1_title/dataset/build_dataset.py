from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os


class FNDDataset(Dataset):
    def __init__(self, modelname, datadir, split, tokenizer, max_word_len, max_sent_len, use_saved_data=False):
        self.modelname = modelname
        self.split = split
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len

        # load data
        self.use_saved_data = use_saved_data
        if self.use_saved_data:
            self.data = torch.load(os.path.join(datadir, f'{modelname}_s{max_sent_len}_w{max_word_len}', f'{split}.pt'))
        else:
            self.data = json.load(open(os.path.join(datadir, f'{split}.json'),'r'))
            self.data_info = pd.read_csv(os.path.join(datadir, f'{split}_info.csv'))
        
        # tokenizer
        self.tokenizer = tokenizer

    def transform(self, sent_list):
        sent_list = sent_list[:self.max_sent_len]
        doc = [self.tokenizer.encode(sent)[:self.max_word_len] for sent in sent_list] 
        
        return doc
    
    def padding(self, doc):
        num_pad_doc = self.max_sent_len - len(doc)
        num_pad_sent = [max(0, self.max_word_len - len(sent)) for sent in doc]

        doc = [sent + [self.tokenizer.pad_token_id] * num_pad_sent[idx] for idx, sent in enumerate(doc)]
        doc = doc + [[self.tokenizer.pad_token_id] * self.max_word_len for i in range(num_pad_doc)]
            
        return doc

    def transform_fndnet(self, sent_list):
        doc = sum([self.tokenizer.encode(sent) for sent in sent_list], [])[:self.max_word_len]

        return doc 

    def padding_fndnet(self, doc):
        num_pad_word = max(0, self.max_word_len - len(doc))
        doc = doc + [self.tokenizer.pad_token_id] * num_pad_word

        return doc


    def __getitem__(self, i):

        if self.use_saved_data:
            doc = {'input_ids':self.data['doc'][i]}
            label = self.data['label'][i]

            return doc, label
        
        else:
            news_idx = self.data_info.iloc[i]
            news_info = self.data[str(news_idx['id'])]
        
            # label
            label = 1 if news_idx['label']=='fake' else 0

            if self.modelname != 'BERT':
                # input
                sent_list = [news_info['title']] + news_info['text']
                
                # HAN
                if self.modelname == 'HAN':
                    doc = self.transform(sent_list)
                    doc = self.padding(doc)
                elif self.modelname == 'FNDNet':
                    doc = self.transform_fndnet(sent_list)
                    doc = self.padding_fndnet(sent_list)

                doc = {'input_ids':torch.tensor(doc)}

            elif self.modelname == 'BERT':
                doc = self.tokenizer(
                    news_info['title'],
                    ' '.join(news_info['text']),
                    return_tensors='pt',
                    max_length = self.max_word_len,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True
                )

                doc['input_ids'] = doc['input_ids'][0]
                doc['attention_mask'] = doc['attention_mask'][0]
                doc['token_type_ids'] = doc['token_type_ids'][0]

            return doc, label


    def __len__(self):
        if self.use_saved_data:
            return len(self.data['doc'])
        else:
            return len(self.data)
    
    @property
    def num_classes(self):
        return 2


class FNDTokenizer:
    def __init__(self, vocab, tokenizer, special_tokens: list = []):    
        self.vocab = vocab        
        self.tokenizer = tokenizer

        # add token ids
        self.special_tokens = {}

        special_tokens = ['UNK','PAD'] + special_tokens
        for special_token in special_tokens:
            self.add_tokens(special_token)
        
        # set unknown and pad 
        self.unk_token_id = self.special_tokens['UNK']
        self.pad_token_id = self.special_tokens['PAD']
        
    def encode(self, sentence):
        return [
            self.vocab.index(word) if word in self.vocab else self.unk_token_id 
            for word in self.tokenizer.morphs(sentence)
        ]
    
    def batch_encode(self, b_sentence):
        return [self.encode(sentence) for sentence in b_sentence]
    
    def decode(self, input_ids):
        return [self.vocab[i] for i in input_ids]
    
    def batch_decode(self, b_input_ids):
        return [self.decode(input_ids) for input_ids in b_input_ids]
    
    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.vocab)

    def add_tokens(self, name):
        self.special_tokens.update({name: self.vocab_size})
        self.vocab += [name]

