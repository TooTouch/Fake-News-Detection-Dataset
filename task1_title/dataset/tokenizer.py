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