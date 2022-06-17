import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv

from einops import rearrange

class HierAttNet(nn.Module):
    def __init__(self, word_dim=64, sent_dim=128, dropout=0.1, num_classes=2, 
                 vocab_len=358043, embed_dim=100):
        super(HierAttNet, self).__init__()

        # word to embeding
        self.w2e = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim)

        # word attention
        self.word_attention = WordAttnNet(
            vocab_len = vocab_len, 
            embed_dim = embed_dim,
            word_dim  = word_dim,
            dropout   = dropout
        )

        # sentence attention
        self.sent_attention = SentAttnNet(
            word_dim = word_dim, 
            sent_dim = sent_dim, 
            dropout  = dropout
        )

        # classifier
        self.fc = nn.Linear(2 * sent_dim, num_classes)

    def init_w2e(self, weights, nb_special_tokens=0):
        assert isinstance(weights, np.ndarray)

        weights = torch.from_numpy(
            np.concatenate([
                weights, 
                np.random.randn(nb_special_tokens, weights.shape[1])
            ]).astype(np.float)
        )
        self.word_attention.w2e = self.word_attention.w2e.from_pretrained(weights)

    def freeze_w2e(self):
        self.word_attention.w2e.weight.requires_grad = False

    def forward(self, input_ids, output_attentions=False):
        # word attention
        words_embed, words_attn_score = self.word_attention(input_ids) 

        # sentence attention
        sents_embed, sents_attn_score = self.sent_attention(words_embed)

        # classification
        out = self.fc(sents_embed)

        if output_attentions:
            return out, words_attn_score, sents_attn_score
        else:
            return out


class WordAttnNet(nn.Module):
    def __init__(self, vocab_len, embed_dim, word_dim, dropout):
        super(WordAttnNet, self).__init__()
        # word to embeding
        self.w2e = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim)

        # word attention
        self.gru = nn.GRU(embed_dim, word_dim, bidirectional=True, dropout=dropout)
        self.attention = Attention(2 * word_dim, word_dim)

        # layer norm and dropout
        self.layer_norm = nn.LayerNorm(2 * word_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # docs: B x sents x words
        b, s, _ = input_ids.size()
        input_ids = rearrange(input_ids, 'b s w -> (b s) w')

        # input_embed: (B x sents) x words x dims
        input_embed = self.w2e(input_ids)
        input_embed = self.dropout(input_embed)

        # word attention
        words_embed, _ = self.gru(input_embed.float())
        words_embed = self.layer_norm(words_embed)

        words_embed, words_attn_score = self.attention(words_embed)

        words_embed = rearrange(words_embed, '(b s) d -> b s d', b=b, s=s)
        words_embed = self.dropout(words_embed)

        return words_embed, words_attn_score


class SentAttnNet(nn.Module):
    def __init__(self, word_dim, sent_dim, dropout):
        super(SentAttnNet, self).__init__()
        # sentence attention
        self.gru = nn.GRU(2 * word_dim, sent_dim, bidirectional=True, dropout=dropout)
        self.attention = Attention(2 * sent_dim, sent_dim)

        # layer norm and dropout
        self.layer_norm = nn.LayerNorm(2 * sent_dim)

    def forward(self, words_embed):
        # sentence attention
        sents_embed, _ = self.gru(words_embed)
        sents_embed = self.layer_norm(sents_embed)

        sents_embed, sents_attn_score = self.attention(sents_embed)
        
        return sents_embed, sents_attn_score


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(in_dim, out_dim)
        self.context = nn.Linear(out_dim, 1)

    def forward(self, x):
        attn = torch.tanh(self.attention(x))
        attn = self.context(attn).squeeze()
        attn_score = torch.softmax(attn, dim=1)

        out = torch.einsum('b n d, b n -> b d', x, attn_score) 

        return out, attn_score

