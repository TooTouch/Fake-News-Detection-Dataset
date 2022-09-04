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
    def __init__(self, finetune_bert=False, do_train=True):
        super(Bert, self).__init__()
        self.model, vocab = get_pytorch_kobert_model(cachedir=".cache")
        # add [BOS], [EOS]
        self.model.resize_token_embeddings(len(vocab)) 

        # whether finetune the backbone (bert or bertsum)
        self.finetune = True if (finetune_bert and do_train) else False
        if do_train:
            if self.finetune:
                _logger.info(f"Finetuning BERT backbone")
            else:
                _logger.info(f"Not finetuning BERT backbone")

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec

class KoBERTSEG(nn.Module):
    def __init__(self, finetune_bert=False, do_train=True, window_size=3):
        super(KoBERTSEG, self).__init__()

        self.bert = Bert(finetune_bert=finetune_bert, do_train=do_train)
        self.classifier = Classifier(window_size=window_size)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        classified = self.classifier(sents_vec)
        return classified
    
    
class Classifier(nn.Module):
    def __init__(self, window_size):
        super(Classifier, self).__init__()
        if window_size == 1:
            conv_kernel_size = 2
            flat_size = 256
        else:
            conv_kernel_size = window_size*2-2
            flat_size = 256 * 3

        ln_size = 1 if window_size == 1 else 3
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=256, kernel_size=conv_kernel_size),
            nn.LayerNorm([256, ln_size]),
            nn.ReLU(),
        )
        
        self.block2 = nn.Sequential(nn.Linear(flat_size, 2))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.transpose(1, 2).contiguous() # B * N * C --> B * C * N
        out = self.block1(x)
        out = out.view(batch_size, -1)
        out = self.block2(out)
        return out
    
    
@register_model
def kobertseg(**kwargs):
    args = kwargs['args']
    model = KoBERTSEG(
        finetune_bert = args.finetune_bert, 
        do_train      = args.do_train, 
        window_size   = args.window_size
    )
    return model