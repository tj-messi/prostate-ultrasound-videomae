# v0.1.0 - Code-Only Research Release

Prostate ultrasound video classification with adapted VideoMAE pretraining and a two-class ViT classifier. This is an initial source release, not a claim of reproduced clinical results.

## Included

- MAE pretraining source, downstream fine-tuning, labeled video evaluation, and portable example configuration.
- Installation instructions, data layout, preprocessing and label conventions, upstream notices and licenses.
- Synthetic regression tests with no medical images or external checkpoints.

## Release Preparation Fixes

- Fix short-video random sampling and consistent flip augmentation; ignore nonimage files and reject empty videos.
- Preserve resize/RGB/ImageNet normalization and normalize-then-average grayscale behavior.
- Share temporal windowing and default any-positive aggregation between validation and standalone evaluation; retain mean-logit evaluation as an explicit option. Predictions may differ from the previous standalone evaluator.
- Load checkpoint containers on CPU, normalize DDP/compile prefixes, reject zero matching parameters and shape mismatches, and require explicit opt-in for legacy pickle loading in the main ultrasound workflow.
- Use compatible CUDA AMP imports, propagate training errors with nonzero exit status, and parse configuration as standard JSON.
- Remove private server/institution paths and unconditional private pretraining initialization. Compile is opt-in; unsupported decoder masking is rejected; reconstruction image exports are disabled by default.
- Resolve the pretraining profiler import, permit raw-frame use without decord, and serialize new pretraining arguments as dictionaries.
- Make corrupt pretraining samples fail explicitly instead of recursively replacing samples; accept whitespace-separated manifests.

## External Weights

Download from [Google Drive](https://drive.google.com/drive/folders/1slvRRm54JkY9pPxWY2bpTi8NQ-6JqIx4). **No weights or patient data are included in this repository or attached as Release assets.** Fine-tuned classifier and MAE initialization checkpoints have different roles. The folder's exact hashes and training provenance are not established by this release.

## Verification And Limits

Release checks use Python 3.12 on Windows, PyTorch 2.6.0 CPU, torchvision 0.21.0, and timm 0.4.12. Synthetic tests cover short/full videos, augmentation, grayscale ordering, temporal windows and aggregation, checkpoint containers/trust/matching, a tiny classifier forward/backward pass, raw-frame pretraining transforms, and tiny all-token MAE reconstruction forward/backward.

All 27 synthetic tests passed. This includes CLI/import and Python syntax checks; the default base MAE model also constructs successfully. No real patient data, full-checkpoint inference, complete GPU training, distributed/DeepSpeed execution, clinical accuracy, patient leakage audit, or paper metric reproduction is claimed. T0/T1 medical meanings must be confirmed against the study protocol. The example configuration defaults to random initialization until a local MAE path is supplied.

See the README and pretrain/README.md for exact commands and known scientific differences. Preserve applicable upstream licenses. Code licensing does not grant rights to medical data or external weights.
