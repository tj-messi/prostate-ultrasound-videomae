# MAE Pretraining

Adapted from [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), whose MIT license is retained in [LICENSE](LICENSE). Source headers also acknowledge BEiT, timm, DeiT, and DINO. This directory contains the ultrasound reconstruction adaptations and legacy action-classification utilities, not the complete upstream repository.

## Install

Install the root requirements first, with a matching CUDA PyTorch build for training, then:

```bash
python -m pip install -r pretrain/requirements.txt
```

Raw image-frame datasets do not need decord. To read encoded videos, install `decord==0.6.0` separately on a supported platform. The legacy `run_class_finetuning.py` action-classification utility imports DeepSpeed and is outside the tested ultrasound workflow; its optional dependencies are not installed by these requirements. Use `train/run.py` for the downstream ultrasound classifier.

## Raw Frames

Create a local private manifest with one entry per video directory:

```text
video_0001 1 100 0
video_0002 1 80 0
```

Fields are `relative_path start_index frame_count unused_label`. The example assumes frames named `img_00001.jpg`, starting at 1, with exactly the stated count. Paths cannot contain spaces. Do not commit real manifests or patient identifiers. For encoded videos use a negative frame count; the loader then uses decord. The label is unused for reconstruction.

From the repository root, an example single-GPU launch is:

```bash
cd pretrain
python run_mae_pretraining.py --model pretrain_videomae_base_patch16_224 --data_root ../datasets/pretrain --data_path ../datasets/pretrain_manifest.txt --fname_tmpl "img_{:05}.jpg" --num_frames 16 --sampling_rate 1 --input_size 224 --mask_type tube --mask_ratio 0.9 --decoder_mask_ratio 0 --batch_size 4 --num_workers 4 --epochs 100 --warmup_epochs 10 --output_dir ../experiments/pretrain --log_dir ../experiments/pretrain/tensorboard --no_auto_resume
```

This command illustrates the interface; it is not an authenticated reconstruction of the original training run. The dataset must contain at least one complete global batch. Set batch size, schedule, and worker count for your hardware and protocol. The manifest format is inherited from the upstream frame loader, and pretraining augmentations differ from downstream resizing/grayscale preprocessing.

No initialization checkpoint is loaded implicitly. To initialize from a verified MAE checkpoint, add `--finetune /local/path/checkpoint.pth`. Checkpoint containers and DDP/compile prefixes are normalized, zero-match loads fail, and missing/unexpected keys are printed. Legacy pickled objects require explicit `--trust_checkpoint`. Only use that option for trusted files. `--resume` restores training state; new checkpoints serialize arguments as ordinary dictionaries for restricted loading. `--compile` explicitly opts into torch.compile where supported; it is disabled by default.

## Reconstruction Objective

The current entry point sets `loss_scaler=None`. Its active objective is masked-patch MSE with weight 1 and visible-patch MSE with weight 0, with target patch normalization enabled by default. The decoder returns all tokens in visible-then-masked order. Preserve this ordering when modifying the engine.

The dormant scaler branch uses visible MSE weight 1 and masked MSE weight 10. Enabling that branch changes the scientific objective; it is not merely a precision setting. SSIM/perceptual helpers exist, but the SSIM contribution to training loss is commented out. This release does not claim SSIM-trained reproduction.

Decoder masking is unsupported by the custom all-token loss path: nonzero `--decoder_mask_ratio` now fails explicitly. Encoder masking must be strictly between zero and one. Training-time image/reconstruction exports are disabled by default. The retained diagnostic blocks can export sensitive images if manually enabled; keep generated outputs private.

## Verification Limits

The pretraining CLI/import path, frame dataset transforms, and a tiny MAE reconstruction forward/backward pass are checked with synthetic inputs where documented in the release notes. Full pretraining, CUDA kernels, distributed training, legacy DeepSpeed, external weights, and reported scientific results have not been reproduced for this release. Keep patient-level splits and institution approvals outside this code-only repository.
