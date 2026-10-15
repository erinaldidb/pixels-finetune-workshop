# MONAI bundle fine-tuning contract — NV-Segment-CT / NV-Segment-CTMR

Read this before changing a segmentation label mapping, workflow, or the
launcher `assets/train_segment_finetune.py`. Facts were checked against
[NVIDIA-Medtech/NV-Segment-CTMR@cb921f5](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/cb921f5c58837c0f42a713855d68b32af88e1cdd)
(`docs/finetune.md`, `docs/softmax_finetune.md`) and NVIDIA's
[`nv-segment-ct-finetune`](https://github.com/NVIDIA-Medtech/medical-AI-skills/tree/dev/skills/nv-segment-ct-finetune) skill.

## Pins

| Model | Bundle dir in the repo | Weights (Hugging Face) | Label IDs |
|---|---|---|---|
| `nv-segment-ctmr` | `NV-Segment-CTMR/` | `nvidia/NV-Segment-CTMR@4fb8b4a6…` `vista3d_pretrained_model/model.pt` | 345 (CT + MRI), `assets/label_ids/nv-segment-ctmr.json` |
| `nv-segment-ct` | `NV-Segment-CT/` | `nvidia/NV-Segment-CT@afb51518…` `vista3d_pretrained_model/model.pt` | 124 (CT), `assets/label_ids/nv-segment-ct.json` |

Both use commit `cb921f5c58837c0f42a713855d68b32af88e1cdd`, `monai==1.4.0`
(MONAI 1.5.x can crash the finetune loss on boolean labels) and `numpy<2`.

## Choosing the workflow

| | `softmax` | `standard` |
|---|---|---|
| Config stack | `configs/train_continual_softmax.json` (self-contained) | `configs/train.json` + `configs/train_continual.json` |
| Classes | Fixed, mutually exclusive | Promptable; classes chosen at inference |
| Class IDs | Must exist in the bundle's `label_dict.json` | Any ID `< 255` (`num_classes`); unknown IDs are learned from scratch |
| Checkpoint | `checkpoints/model_softmax.pt` | `checkpoints/model_finetune.pt` |
| Inference | upstream `configs/inference_softmax.json` with the same mapping order | upstream `configs/inference.json` |

ImageCHD cardiac IDs (149, 151–154, 171) exist only in NV-Segment-CTMR. With
NV-Segment-CT, use `standard`: aorta (6) starts from pretrained weights, and
the chambers become new classes.

## Launch

The launcher runs the upstream entry point from the bundle directory, with the
bundle on `PYTHONPATH` (its configs do `$import scripts`):

```bash
python -m monai.bundle run --meta_file configs/metadata.json --logging_file configs/logging.conf \
  --config_file "['<workflow configs>','override.json']"
torchrun --standalone --nproc_per_node=N -m monai.bundle run ... \
  --config_file "['<workflow configs>','configs/multi_gpu_train.json','override.json']"
```

## Override keys (top-level)

| Key | Notes |
|---|---|
| `dataset_dir`, `data_list_file_path` | MSD datalist `{"training": [{"image","label","fold"}]}`; `fold == 0` is validation. **Not** `data_file_base_dir`. |
| `label_mappings` | Set the whole dict `{"default": [[dataset_label, class_id], ...]}`. |
| `epochs` | **Not** `train#trainer#max_epochs`: the LR schedule `t_total` and class sampling read `@epochs`. |
| `learning_rate`, `patch_size`, `patch_size_valid` | Upstream patch defaults: CT 128³, CTMR `[192,198,128]` (standard) / 128³ (softmax). |
| `train_dataset_cache_rate`, `val_dataset_cache_rate` | Lower if host RAM is short. |
| `finetune`, `finetune_model_path` | `<bundle>/models/model.pt`. |
| `ckpt_dir`, `output_dir` | Checkpoints and TensorBoard events (replayed into MLflow). |
| `use_mlflow: false`, `use_tensorboard: true` | The launcher owns MLflow logging. |
| `lr_schedule#activate: false` | Standard workflow only, as in `train_continual.json`. |
| `drop_label_prob: 0`, `drop_point_prob: 1` | Standard workflow, `training.auto_seg: true`: class-prompt-only training. |

## Label rules

- Foreground entries only, positive and unique. Never pad with `[v, 0]`. In
  softmax mode each entry becomes an output channel.
- `Relabeld` (`MapLabelValue`) leaves unmapped values untouched. Stray values
  break the softmax one-hot loss, or are silently trained as global class IDs
  in the standard workflow. `check_labels: true` zeroes them during staging.
- Keep the mapping order unchanged for inference. The launcher logs it as
  `configs/label_mapping.json`.

## Before / after evaluation

After training, the launcher scores the first `evaluation.max_cases` fold-0
(validation) cases with both the pretrained and the fine-tuned weights. Both
runs go through upstream configs only:

| Workflow | Before (pretrained) | After (fine-tuned) |
|---|---|---|
| `softmax` | `configs/inference.json`, `input_dict: {image, label_prompt: <class IDs>}` (zero-shot VISTA3D) | `configs/inference_softmax.json`, `checkpoint_path: model_softmax.pt`, same `label_mappings` |
| `standard` | `train.json` + `train_continual.json` + `configs/evaluate.json`, `ckpt_path: models/model.pt` | same stack, `ckpt_path: model_finetune.pt` |

- Predictions come back in the original image space (`Invertd`). Class-ID
  outputs are mapped back to dataset labels, and Dice is computed per class in
  canonical (RAS) orientation against the staged label.
- `inference.json` only accepts prompts the pretrained bundle knows (CT:
  1–132 minus a few; CTMR: 1–345 minus a few). An unknown class has no
  "before" mask.
- Outputs: `evaluation/dice.json` (per case and mean, per class) and
  `evaluation/slices_<case>.npz`. The npz holds axial and coronal slices
  through the largest ground-truth cross-section: CT clipped to
  [-200, 600] HU, plus ground-truth, before, and after masks. These are
  small enough to plot on CPU.
- Metrics: `eval_before_mean_dice`, `eval_after_mean_dice`. An evaluation
  failure adds a warning; it never fails a finished training run.

## MLflow model and Unity Catalog

`segment_pyfunc.py` (models-from-code) is logged as `model/` with artifacts:
`bundle/` (upstream `configs/` + `scripts/` at the pinned commit),
`checkpoint`, and `label_mapping`.
- Signature: input `image_path` (string); outputs `label_path`,
  `class_voxels` (JSON); param `output_dir`.
- `predict()` runs `configs/inference_softmax.json` (softmax) or
  `configs/inference.json` with class prompts (standard) in a subprocess.
  It needs a CUDA GPU for practical runtimes.
- The model URI is stored in the run tag `segment.model_uri`.
  `scripts/register_model.py` registers it as `<catalog>.ws_<user>.<model_id>_finetuned`,
  sets alias `workshop`, and tags the version with the pins and Dice.
  The practitioner needs `USE CATALOG` and `CREATE SCHEMA` on the catalog.

## Evidence

- `run_summary.json` records `best_val_mean_dice`, `best_val_epoch`, versions, and warnings.
- `val_at_start: true` scores the initial weights at epoch 0. If epoch 0 stays
  best, training did not improve on them.
- Validation dice is a training-time trace on fold 0, not a held-out evaluation.
