import logging
import torch

from .registry import is_model, model_entrypoint

_logger = logging.getLogger('train')

def create_model(
        modelname: str,
        hparams: dict,
        word_embed = None,
        tokenizer = None,
        freeze_word_embed: bool = False,
        use_pretrained_word_embed: bool = False,
        checkpoint_path: str = None,
        **kwargs
    ):
 
    if not is_model(modelname):
        raise RuntimeError('Unknown model (%s)' % modelname)

    create_fn = model_entrypoint(modelname)
    
    model = create_fn(
        hparams    = hparams
    )

    # word embedding
    if use_pretrained_word_embed:
        _logger.info('load pretrained word embedding')
        model.init_w2e(word_embed, len(tokenizer.special_tokens))
        
    # freeze word embedding
    if freeze_word_embed:
        _logger.info('freeze pretrained word embedding')
        model.freeze_w2e()

    # load checkpoint weights
    if checkpoint_path:
        _logger.info('load a trained model weights from {}'.format(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path))

    return model
