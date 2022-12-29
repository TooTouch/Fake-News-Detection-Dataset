import torch
import torch.nn as nn

from kobert.pytorch_kobert import get_pytorch_kobert_model

from log import logging
from .registry import register_model

_logger = logging.getLogger('train')

class BTS(nn.Module):
    """
    BERT for Topic Segmentation
    """
    def __init__(self, finetune_bert: bool = False):
        super(BTS, self).__init__()
        self.model, vocab = get_pytorch_kobert_model(cachedir=".cache")
        # add [BOS], [EOS]
        self.model.resize_token_embeddings(len(vocab)) 

        # whether finetune the backbone
        self.finetune = finetune_bert
        if self.finetune:
            _logger.info(f"Finetuning BERT backbone")
        else:
            for p in self.model.parameters():
                p.requires_grad = False
            _logger.info(f"Not finetuning BERT backbone")

        self.classifier = nn.Linear(768, 2)

    def forward(self, src, segs, mask_src):
        if self.finetune:
            top_vec, _ = self.model(src, token_type_ids=segs, attention_mask=mask_src)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(src, token_type_ids=segs, attention_mask=mask_src)

        cls_hidden_states = top_vec[:,0]
        outputs = self.classifier(cls_hidden_states)
        
        return outputs

    def __len__(self):
        return len(self.datasets)
    
@register_model
def bts(hparams, **kwargs):
    model = BTS(
        finetune_bert = hparams['finetune_bert']
    )
    return model
