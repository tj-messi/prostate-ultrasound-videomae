from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from pretrain.dataset.pretrain_datasets import DataAugmentationForVideoMAEv2, HybridVideoMAE
from pretrain.models.modeling_pretrain import PretrainVisionTransformer


def test_pretraining_raw_frames_without_decord(tmp_path):
    video = tmp_path / 'synthetic_video'
    video.mkdir()
    for i in range(1, 5):
        Image.new('RGB', (32, 32), (10 * i, 60, 100)).save(video / f'img_{i:05}.jpg')
    manifest = tmp_path / 'synthetic.txt'
    manifest.write_text('synthetic_video 1 4 0\n', encoding='utf-8')
    args = SimpleNamespace(input_size=32, window_size=(2, 2, 2), mask_type='tube',
                           mask_ratio=0.5, decoder_mask_ratio=0)
    dataset = HybridVideoMAE(str(tmp_path), str(manifest), new_length=4, new_step=1,
                             transform=DataAugmentationForVideoMAEv2(args))
    frames, mask, decoder_mask = dataset[0]
    assert frames.shape == (3, 4, 32, 32)
    assert mask.shape == (2, 4)
    assert torch.as_tensor(mask).unsqueeze(0).flatten(1).shape == (1, 8)
    assert mask.sum() == 4
    assert (decoder_mask == 1 - mask).all()
    assert torch.isfinite(frames).all()


def test_invalid_pretraining_frames_fail_without_resampling(tmp_path):
    video = tmp_path / 'synthetic_video'
    video.mkdir()
    manifest = tmp_path / 'synthetic.txt'
    manifest.write_text('synthetic_video   1   4   0\n\n', encoding='utf-8')
    dataset = HybridVideoMAE(str(tmp_path), str(manifest), new_length=4, new_step=1)
    with pytest.raises(RuntimeError, match='Failed to load video'):
        dataset[0]


def test_tiny_mae_reconstructs_all_tokens_and_backpropagates():
    model = PretrainVisionTransformer(
        img_size=16, patch_size=8, all_frames=4, tubelet_size=2,
        encoder_embed_dim=24, encoder_depth=1, encoder_num_heads=3,
        decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=3,
        decoder_num_classes=2 * 8 * 8 * 3)
    frames = torch.randn(2, 3, 4, 16, 16)
    mask = torch.tensor([[False, True, False, True, False, True, False, True]] * 2)
    output = model(frames, mask, ~mask)
    assert output.shape == (2, 8, 384)
    output[:, 4:].square().mean().backward()
    assert model.mask_token.grad is not None
    assert torch.isfinite(model.mask_token.grad).all()
