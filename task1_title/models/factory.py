import logging
import torch

from .registry import is_model, model_entrypoint

_logger = logging.getLogger('train')

def create_model(
        modelname,
        pretrained = False,
        word_embed = None,
        tokenizer = None,
        args = None, 
    ):
 
    if not is_model(modelname):
        raise RuntimeError('Unknown model (%s)' % modelname)

    create_fn = model_entrypoint(modelname)
    
    model = create_fn(
        pretrained = pretrained, 
        args = args
    )

    # word embedding
    use_pretrained_word_embed = args.use_pretrained_word_embed if args else None
    if use_pretrained_word_embed:
        _logger.info('load pretrained word embedding')
        model.init_w2e(word_embed, len(tokenizer.special_tokens))
        
    # freeze word embedding
    freeze_word_embed = args.freeze_word_embed if args else None    
    if freeze_word_embed:
        _logger.info('freeze pretrained word embedding')
        model.freeze_w2e()

    # load checkpoint weights
    checkpoint_path = args.checkpoint_path if args else None
    if checkpoint_path:
        _logger.info('load a trained model weights from {}'.format(args.checkpoint_path))
        model.load_state_dict(torch.load(args.checkpoint_path))

    return model
