from models import HierAttNet, FNDNet, BTS
from transformers import AutoConfig

import logging

_logger = logging.getLogger('train')

def create_model(args, word_embed, tokenizer):
    if args.modelname == 'HAN':
        model = HierAttNet(
            word_dims   = args.word_dims, 
            sent_dims   = args.sent_dims, 
            num_classes = args.num_classes, 
        )
    elif args.modelname == 'FNDNet':
        model = FNDNet(
            dims        = args.dims,
            num_classes = args.num_classes, 
        )
    elif args.modelname == 'BTS':
        model_config = AutoConfig.from_pretrained(args.pretrained_name)
        model = BTS.from_pretrained(args.pretrained_name, config=model_config)

    if args.use_pretrained_word_embed:
        _logger.info('load pretrained word embedding')
        model.init_w2e(word_embed, len(tokenizer.special_tokens))
        
    if args.freeze_word_embed:
        _logger.info('freeze pretrained word embedding')
        model.freeze_w2e()

    return model