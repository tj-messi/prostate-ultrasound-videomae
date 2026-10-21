"""CPU-first checkpoint loading with explicit legacy-pickle consent."""
import inspect
from collections.abc import Mapping
from pathlib import Path

import torch


def load_checkpoint(path, trust_checkpoint=False):
    path = Path(path)
    if path.suffix.lower() == '.safetensors':
        from safetensors.torch import load_file
        return load_file(str(path), device='cpu')
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = not trust_checkpoint
    elif not trust_checkpoint:
        raise RuntimeError('This PyTorch version lacks restricted loading. Upgrade PyTorch or explicitly trust the checkpoint.')
    return torch.load(str(path), **kwargs)


def state_dict_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must contain a state dictionary')
    for key in ('state_dict', 'model', 'module'):
        if isinstance(checkpoint.get(key), Mapping):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                 for k, v in checkpoint.items()):
        raise ValueError('Expected a nonempty string-to-tensor state dictionary')
    state = {}
    for key, value in checkpoint.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key in state:
            raise ValueError('Duplicate normalized checkpoint key: ' + key)
        state[key] = value
    return state


def load_model_weights(model, path, strict=True, prefix=None, trust_checkpoint=False):
    state = state_dict_from_checkpoint(load_checkpoint(path, trust_checkpoint))
    if prefix:
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    expected = model.state_dict()
    matched = set(expected).intersection(state)
    if not matched:
        raise ValueError('No checkpoint parameters match the model; check checkpoint type and prefix')
    for key in matched:
        if expected[key].shape != state[key].shape:
            raise ValueError('Checkpoint shape mismatch for ' + key)
    return model.load_state_dict(state, strict=strict)
