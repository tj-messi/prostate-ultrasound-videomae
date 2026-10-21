# Prostate Ultrasound VideoMAE

Research code for binary classification of prostate ultrasound videos using masked-autoencoder pretraining and a ViT classifier. This model produces video labels, not lesion boxes or segmentation masks.

**Code-only release:** patient data, private manifests, weights, predictions, and experiment outputs are excluded. Download weights separately from [Google Drive](https://drive.google.com/drive/folders/1slvRRm54JkY9pPxWY2bpTi8NQ-6JqIx4). No weights are attached to GitHub Releases.

## Install

Use an isolated Python environment. Release smoke tests used Python 3.12 on Windows with PyTorch 2.6.0 CPU, torchvision 0.21.0, and timm 0.4.12. For evaluation and tests:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Training requires CUDA. Install a matching CUDA build of the pinned PyTorch/torchvision pair following the [official instructions](https://pytorch.org/get-started/previous-versions/), then install the requirements. Full GPU experiments have not been rerun for this release. See [pretraining instructions](pretrain/README.md) for additional dependencies.

## Layout

- `train/`: downstream dataset, ViT/VideoMAE fine-tuning model, trainer, losses, metrics, and `run.py`.
- `test.py`: inference-only evaluation. It writes per-video predictions to CSV and reports accuracy.
- `pretrain/`: VideoMAE pretraining and fine-tuning implementation.
- `configs/`: training configuration. Update data and checkpoint paths for your machine.
- `tests/`: generated-image regression tests and tiny-model smoke tests.
- [Release notes](docs/RELEASE_NOTES.md): changes and verification boundaries.

## Data format

The dataset root must contain two directories named `T0` and `T1`. Each video is represented by a directory of ordered image frames:

```
data_root/
  T0/video_0001/*.png
  T1/video_0002/*.png
```

Labels are **T0 = 0, T1 = 1**. Medical label definitions are not established by the code; verify them against your study protocol. Split by patient, keeping all visits, clips, and frames of a patient together. No patient-level leakage audit is claimed.

Use zero-padded frame names: sorting is lexicographic. Supported extensions are PNG, JPEG, BMP, TIFF. Hidden files and nonimage files are ignored; empty videos fail explicitly. Both class directories must exist. Training samples a contiguous 16-frame clip and repeats the final frame if short. Evaluation keeps the full sequence and pads it to at least 16 frames.

Preprocessing converts to RGB, resizes to 224 x 224, and applies ImageNet normalization. Grayscale mode averages the three **normalized** channels and repeats them three times, preserving the original ordering. This differs from grayscale conversion before normalization. Match preprocessing to the checkpoint.

## Evaluation

From the package root:

```bash
python test.py --data-root datasets/val --checkpoint weights/classifier.pth --grayscale --device cpu --output predictions.csv
```

The evaluation script uses the same 224x224 preprocessing and 16-frame temporal windows as the training pipeline. It does not train or modify model weights.

Download a fine-tuned two-class classifier to your ignored local `weights/` directory. A MAE pretraining checkpoint cannot substitute for a classifier. The Drive folder is maintained externally; exact file hashes, accessibility, and training provenance are not verified by this release.

The default `--aggregation any_positive` matches training validation: any clip with `logit_T1 > logit_T0` makes the video T1; ties are T0. `--aggregation mean_logits` retains the earlier standalone evaluation rule. The final window aligns to the end and may overlap. The two aggregation rules can disagree; neither implies clinical calibration. Prediction CSVs contain local video paths and should remain private.

Checkpoints load on CPU with restricted loading by default. Raw state dictionaries and `model`, `state_dict`, or `module` containers are supported, with DDP/compile prefixes normalized. Legacy Python objects may require `--trust-checkpoint`, which permits pickle code execution: use it only for verified trusted files. There is no automatic unsafe fallback.

## Training

Update `configs/Finetune_ViT_4090_USVideo_baseline.json` so that its dataset and initialization paths point to local paths, then run:

Paths are relative to the working directory (`train/` below). Set the generic `ExternalA/B/C` validation paths or remove those entries. Adjust batch size and workers to your hardware. `init_weights_path_loose: null` means random initialization; set it to your MAE checkpoint to extract `encoder.` parameters. For a full classifier use `init_weights_path_strict` instead. Enable only one initialization path. Set `trust_checkpoint: true` only for a verified legacy pickle. Missing/unexpected initialization keys are reported. A differently numbered shared checkpoint is not asserted to equal the historical initialization checkpoint.

```bash
cd train
python run.py --config ../configs/Finetune_ViT_4090_USVideo_baseline.json --phase train
```

The original experiments used a 12-layer ViT/VideoMAE-style encoder, 2 classes, 224x224 frames, 16-frame clips, and cross-entropy loss. The configuration is provided for reproducibility; no training is performed as part of this release.

The iteration schedule is an example, not a guarantee of paper reproduction. The configuration permits nondeterministic execution. CUDA training and distributed execution have not been validated for this release. Resume uses a prefix via `path.resume_state`; trusted legacy RNG state may require `path.trust_resume_state: true`.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests use generated images, tiny models, and synthetic checkpoints. They check padding, augmentation, grayscale ordering, temporal coverage, aggregation, checkpoint matching/trust, and classifier forward/backward execution. Passing tests does not demonstrate clinical validity, full-checkpoint inference, GPU stability, or reproduced study metrics.

## Attribution

Pretraining derives from [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2). Existing BEiT, timm, DeiT, DINO, Facebook, and Ross Wightman notices are retained. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [Apache-2.0](licenses/Apache-2.0.txt). The code license grants no rights to patient data or externally hosted weights. No project-specific paper citation is asserted.

Research use only. This repository is not a validated medical device or clinical decision system.
