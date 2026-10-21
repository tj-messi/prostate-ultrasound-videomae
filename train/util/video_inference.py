"""Shared temporal windowing and video-level binary classification."""
import torch


def window_starts(total_frames, clip_size=16):
    if clip_size <= 0 or total_frames < clip_size:
        raise ValueError('Pad videos to at least clip_size frames before inference')
    return [min(start, total_frames - clip_size)
            for start in range(0, total_frames, clip_size)]


def aggregate_logits(logits, mode='any_positive'):
    if mode not in ('any_positive', 'mean_logits'):
        raise ValueError('Unsupported aggregation: ' + mode)
    if not logits:
        raise ValueError('At least one clip is required')
    values = torch.stack(logits)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('Expected one pair of logits per clip')
    if not torch.isfinite(values).all():
        raise ValueError('Non-finite classifier logits')
    if mode == 'any_positive':
        return int((values[:, 1] > values[:, 0]).any().item())
    return int(values.mean(dim=0).argmax().item())


@torch.no_grad()
def predict_video(model, frames, mode='any_positive', clip_size=16):
    if frames.ndim != 5 or frames.shape[0] != 1:
        raise ValueError('Expected video tensor [1, C, T, H, W]')
    logits = []
    for start in window_starts(frames.shape[2], clip_size):
        output = model(frames[:, :, start:start + clip_size])
        if not isinstance(output, torch.Tensor) or output.shape != (1, 2):
            raise ValueError('Classifier must return a tensor of shape [1, 2]')
        logits.append(output[0].float())
    return aggregate_logits(logits, mode)
