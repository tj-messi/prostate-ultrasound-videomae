import argparse
import pickle

import pytest
import torch

from arch.ViT.model_finetune import VisionTransformer
from util.checkpoint import load_checkpoint, load_model_weights


@pytest.mark.parametrize('container', [None, 'model', 'state_dict', 'module'])
def test_checkpoint_containers(tmp_path, container):
    model = torch.nn.Linear(3, 2)
    state = {'module.' + k: v.clone() for k, v in model.state_dict().items()}
    checkpoint = state if container is None else {container: state}
    path = tmp_path / 'synthetic.pth'
    torch.save(checkpoint, path)
    restored = torch.nn.Linear(3, 2)
    load_model_weights(restored, path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_checkpoint_zero_match_and_shape_mismatch(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / 'synthetic.pth'
    torch.save({'unrelated': torch.ones(1)}, path)
    with pytest.raises(ValueError, match='No checkpoint parameters'):
        load_model_weights(model, path, strict=False)
    torch.save({'weight': torch.ones(1)}, path)
    with pytest.raises(ValueError, match='shape mismatch'):
        load_model_weights(model, path, strict=False)


def test_encoder_prefix_loading(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / 'synthetic.pth'
    torch.save({'model': {'encoder.' + k: v for k, v in model.state_dict().items()}}, path)
    result = load_model_weights(torch.nn.Linear(3, 2), path, strict=False, prefix='encoder.')
    assert not result.missing_keys


def test_legacy_pickle_requires_opt_in(tmp_path):
    path = tmp_path / 'synthetic.pth'
    torch.save({'args': argparse.Namespace(synthetic=True)}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert load_checkpoint(path, trust_checkpoint=True)['args'].synthetic


def test_small_classifier_forward_backward_and_checkpoint(tmp_path):
    model = VisionTransformer(img_size=16, patch_size=8, all_frames=4, tubelet_size=2,
                              embed_dim=24, depth=1, num_heads=3, num_classes=2, init_scale=0.001)
    frames = torch.randn(2, 3, 4, 16, 16)
    output = model(frames)
    assert output.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()
    path = tmp_path / 'synthetic.pth'
    torch.save(model.state_dict(), path)
    load_model_weights(model, path)
