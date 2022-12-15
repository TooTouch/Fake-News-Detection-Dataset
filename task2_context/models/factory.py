import logging
import torch

from .registry import is_model, model_entrypoint

_logger = logging.getLogger('train')

def create_model(
        modelname: str,
        hparams: dict,
        checkpoint_path: str = None,
        **kwargs
    ):

    if not is_model(modelname):
        raise RuntimeError('Unknown model (%s)' % modelname)

    create_fn = model_entrypoint(modelname)
    
    model = create_fn(
        hparams    = hparams,
    )

    # load checkpoint weights
    if checkpoint_path:
        _logger.info('load a trained model weights from {}'.format(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path))

    return model