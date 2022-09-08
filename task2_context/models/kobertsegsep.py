import torch
import torch.nn as nn

from kobert.utils import get_tokenizer
from kobert.utils import download as _download
from kobert.pytorch_kobert import get_pytorch_kobert_model

from log import logging
from .registry import register_model

#from .utils import download_weights

_logger = logging.getLogger('train')

class Bert(nn.Module):
    def __init__(self, finetune_bert=False):
        super(Bert, self).__init__()
        self.model, vocab = get_pytorch_kobert_model(cachedir=".cache")
        # add [BOS], [EOS]
        self.model.resize_token_embeddings(len(vocab)) 

        # whether finetune the backbone (bert or bertsum)
        self.finetune = finetune_bert
        if self.finetune:
            _logger.info(f"Finetuning BERT backbone")
        else:
            for p in self.model.parameters():
                p.requires_grad = False
            _logger.info(f"Not finetuning BERT backbone")

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class KoBERTSegSep(nn.Module):
    def __init__(self, finetune_bert=False):
        super(KoBERTSegSep, self).__init__()

        self.bert = Bert(finetune_bert=finetune_bert)
        self.classifier = nn.Linear(768, 2)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        classified = self.classifier(sents_vec)
        return classified
    
    
@register_model
def kobertsegsep(hparams, **kwargs):
    model = KoBERTSegSep(
        finetune_bert = hparams['finetune_bert']
    )
    return model