import torch
import torch.nn as nn
import numpy as np

from .registry import register_model

import logging
_logger = logging.getLogger('train')

class FNDNet(nn.Module):
    def __init__(self, dims: int = 128, num_classes: int = 2, dropout: float = 0.2,
                 vocab_len: int = 58043, embed_dims: int = 100):
        super(FNDNet, self).__init__()

        # word to embeding
        self.w2e = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dims)

        # Conv 1D
        self.conv1_1 = nn.Conv1d(embed_dims, dims, 3)
        self.conv1_2 = nn.Conv1d(embed_dims, dims, 4)
        self.conv1_3 = nn.Conv1d(embed_dims, dims, 5)
        self.conv2 = nn.Conv1d(dims, dims, 5)
        self.conv3 = nn.Conv1d(dims, dims, 5)

        # Max Pooling
        self.max_pool1 = nn.MaxPool1d(5)
        self.max_pool2 = nn.MaxPool1d(30)

        # Lienar 
        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(3 * dims, dims),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.fc = nn.Linear(dims, num_classes)


    def init_w2e(self, weights: np.ndarray, nb_special_tokens: int = 0):

        weights = torch.from_numpy(
            np.concatenate([
                weights, 
                np.random.randn(nb_special_tokens, weights.shape[1])
            ]).astype(np.float)
        )
        self.w2e = self.w2e.from_pretrained(weights)

    def freeze_w2e(self):
        self.w2e.weight.requires_grad = False


    def forward(self, input_ids):
        # B x words
        inputs_embed = self.w2e(input_ids)

        # B x words x dims -> B x dims x words
        inputs_embed = inputs_embed.float().permute(0,2,1)

        # feature extraction
        out1 = self.max_pool1(self.conv1_1(inputs_embed))
        out2 = self.max_pool1(self.conv1_2(inputs_embed))
        out3 = self.max_pool1(self.conv1_3(inputs_embed))
        out = torch.cat([out1, out2, out3], dim=-1)

        out = self.max_pool1(self.conv2(out))
        out = self.max_pool2(self.conv3(out))

        out = torch.flatten(out, start_dim=1)

        # linear classifier
        out = self.linear(out)
        out = self.fc(out)

        return out


@register_model
def fndnet(hparams: dict, **kwargs):
    model = FNDNet(
        dims        = hparams['dims'],
        embed_dims  = hparams['embed_dims'],
        dropout     = hparams['dropout'],
        vocab_len   = hparams['vocab_len'],
        num_classes = hparams['num_classes'], 
    )

    return model