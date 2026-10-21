#!/usr/bin/env python3
"""Inference-only evaluation for the prostate ultrasound VideoMAE classifier."""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))

from arch.ViT.model_finetune import VisionTransformer
from data.USVideo import USVideoDataset
from util.video_inference import predict_video
from util.checkpoint import load_model_weights


def build_model(checkpoint: Path, device: torch.device, grayscale: bool, trust_checkpoint=False) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, img_size=224, num_classes=2, all_frames=16,
        tubelet_size=2, drop_rate=0.0, drop_path_rate=0.1,
        attn_drop_rate=0.0, head_drop_rate=0.0, use_mean_pooling=True,
        init_scale=0.001, with_cp=False,
    )
    load_model_weights(model, checkpoint, strict=True, trust_checkpoint=trust_checkpoint)
    model.to(device).eval()
    return model


def evaluate(model, root: Path, device: torch.device, grayscale: bool, aggregation='any_positive'):
    ds = USVideoDataset(str(root), patch_size=224, frame_count=-1,
                        full_seq=True, data_augmentation=False,
                        gray_scale=grayscale)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows, correct = [], 0
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device)
            label = int(batch["labels"].item())
            pred = predict_video(model, frames, mode=aggregation)
            correct += pred == label
            rows.append({"video_path": batch["video_path"][0],
                         "label": label, "prediction": pred,
                         "correct": int(pred == label)})
    return rows, correct / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Directory containing T0/ and T1/")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument('--trust-checkpoint', action='store_true', help='Allow pickle loading ONLY for a verified trusted checkpoint')
    ap.add_argument('--aggregation', choices=['any_positive', 'mean_logits'], default='any_positive')
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    model = build_model(Path(args.checkpoint), device, args.grayscale, args.trust_checkpoint)
    rows, acc = evaluate(model, Path(args.data_root), device, args.grayscale, args.aggregation)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label", "prediction", "correct"])
        writer.writeheader(); writer.writerows(rows)
    print(f"samples={len(rows)} accuracy={acc:.6f}")
    print(f"predictions saved to {args.output}")


if __name__ == "__main__":
    main()
