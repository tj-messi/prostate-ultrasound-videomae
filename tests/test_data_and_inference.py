from unittest.mock import patch

import pytest
import torch
from PIL import Image

from data.USVideo import USVideoDataset
from util.video_inference import aggregate_logits, predict_video, window_starts


def make_dataset(tmp_path, count=3):
    for label in ('T0', 'T1'):
        video = tmp_path / label / 'synthetic_video'
        video.mkdir(parents=True)
        for i in range(count):
            image = Image.new('RGB', (12, 10), (20 + i * 5, 60, 100))
            image.save(video / f'{i:04}.png')
        (video / 'notes.txt').touch()
        (video / 'subdirectory').mkdir()
    return tmp_path


def test_short_training_video_padding_and_labels(tmp_path):
    root = make_dataset(tmp_path)
    ds = USVideoDataset(root, patch_size=8, frame_count=16, full_seq=False)
    sample = ds[0]
    assert sample['frames'].shape == (3, 16, 8, 8)
    assert sample['start_index'] == 0
    torch.testing.assert_close(sample['frames'][:, 2], sample['frames'][:, -1])
    assert ds[0]['labels'].item() == 0
    assert ds[1]['labels'].item() == 1


def test_augmentation_is_consistent_across_the_clip(tmp_path):
    root = make_dataset(tmp_path)
    plain = USVideoDataset(root, patch_size=8, full_seq=False)[0]['frames']
    augmented = USVideoDataset(root, patch_size=8, full_seq=False, data_augmentation=True)
    with patch('data.USVideo.random.choice', return_value=True):
        frames = augmented[0]['frames']
    torch.testing.assert_close(frames, plain.flip((1, 2, 3)))


def test_grayscale_preserves_normalize_then_channel_average(tmp_path):
    root = make_dataset(tmp_path)
    rgb = USVideoDataset(root, patch_size=8)[0]['frames']
    gray = USVideoDataset(root, patch_size=8, gray_scale=True)[0]['frames']
    torch.testing.assert_close(gray, rgb.mean(0, keepdim=True).repeat(3, 1, 1, 1))


def test_full_sequence_keeps_long_videos(tmp_path):
    root = make_dataset(tmp_path, 19)
    assert USVideoDataset(root, patch_size=8)[0]['frames'].shape[1] == 19


def test_empty_video_and_empty_dataset_fail(tmp_path):
    root = make_dataset(tmp_path, 0)
    with pytest.raises(ValueError, match='no supported image'):
        USVideoDataset(root)[0]
    for label in ('T0', 'T1'):
        (root / label / 'synthetic_video' / 'notes.txt').unlink()
        (root / label / 'synthetic_video' / 'subdirectory').rmdir()
        (root / label / 'synthetic_video').rmdir()
    with pytest.raises(ValueError, match='no video directories'):
        USVideoDataset(root)


@pytest.mark.parametrize('total,expected', [(16, [0]), (17, [0, 1]), (32, [0, 16]), (33, [0, 16, 17])])
def test_temporal_windows(total, expected):
    assert window_starts(total) == expected


def test_aggregation_disagreement_and_ties():
    logits = [torch.tensor([-1., 1.]), torch.tensor([10., 0.])]
    assert aggregate_logits(logits) == 1
    assert aggregate_logits(logits, 'mean_logits') == 0
    assert aggregate_logits([torch.zeros(2)]) == 0
    with pytest.raises(ValueError, match='Non-finite'):
        aggregate_logits([torch.tensor([float('nan'), 0.])])
    with pytest.raises(ValueError):
        window_starts(0)


def test_predict_video_covers_last_frame():
    calls = []

    def classifier(clip):
        calls.append(clip.clone())
        return torch.tensor([[1., 0.]])

    frames = torch.arange(17.).reshape(1, 1, 17, 1, 1)
    assert predict_video(classifier, frames) == 0
    assert len(calls) == 2
    assert calls[-1][0, 0, -1, 0, 0].item() == 16
