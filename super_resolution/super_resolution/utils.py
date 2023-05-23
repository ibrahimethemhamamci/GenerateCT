import torch
from torch import nn
from functools import reduce
from pathlib import Path

from super_resolution.configs import SuperresConfig, ElucidatedSuperresConfig
from ema_pytorch import EMA

def exists(val):
    return val is not None

def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

def load_superres_from_checkpoint(
    checkpoint_path,
    load_weights = True,
    load_ema_if_available = False
):
    model_path = Path(checkpoint_path)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'checkpoint not found at {full_model_path}'
    loaded = torch.load(str(model_path), map_location='cpu')

    superres_params = safeget(loaded, 'superres_params')
    superres_type = safeget(loaded, 'superres_type')

    if superres_type == 'original':
        superres_klass = SuperresConfig
    elif superres_type == 'elucidated':
        superres_klass = ElucidatedSuperresConfig
    else:
        raise ValueError(f'unknown superres type {superres_type} - you need to instantiate your superres with configurations, using classes SuperresConfig or ElucidatedSuperresConfig')

    assert exists(superres_params) and exists(superres_type), 'superres type and configuration not saved in this checkpoint'

    superres = superres_klass(**superres_params).create()

    if not load_weights:
        return superres

    has_ema = 'ema' in loaded
    should_load_ema = has_ema and load_ema_if_available

    superres.load_state_dict(loaded['model'])

    if not should_load_ema:
        print('loading non-EMA version of unets')
        return superres

    ema_unets = nn.ModuleList([])
    for unet in superres.unets:
        ema_unets.append(EMA(unet))

    ema_unets.load_state_dict(loaded['ema'])

    for unet, ema_unet in zip(superres.unets, ema_unets):
        unet.load_state_dict(ema_unet.ema_model.state_dict())

    print('loaded EMA version of unets')
    return superres
