---
name: nv-segment-finetune
description: NVIDIA NV-Segment-CT and NV-Segment-CTMR (VISTA3D, MONAI bundle) fine-tuning on Databricks AI Runtime serverless GPU. Scaffolds and validates the 3-file databricks air project (train.yaml, finetune_config.yml, train_segment_finetune.py) for the softmax or standard continual-learning workflow on NIfTI CT/MRI image/label pairs in a Unity Catalog Volume, with MLflow metrics and checkpoints. Load when the user mentions VISTA3D, NV-Segment-CT, NV-Segment-CTMR, MONAI bundle fine-tuning, 3D segmentation fine-tuning, label mappings, ImageCHD, or the AI Runtime MONAI workshop. For NV-Reason-CXR / NV-Reason-CT use nv-reason-finetune; for UC registration and Model Serving use the Databricks ML training and Model Serving skills.
category: Others
summary: Fine-tune NVIDIA VISTA3D segmentation models (NV-Segment-CT, NV-Segment-CTMR) on AI Runtime, compare before/after, and register in Unity Catalog.
version: 0.3.0
author: Emanuele Rinaldi
license: Databricks License
---

# NV-Segment Fine-Tuning on AI Runtime

## Overview

This skill creates and checks an AI Runtime CLI project that fine-tunes an
NVIDIA VISTA3D segmentation model on the user's 3D NIfTI dataset. The
launcher in `assets/` stages the data, pins the upstream bundle and weights,
runs the upstream entry point (`python -m monai.bundle run`, or `torchrun`
for multi-GPU), and logs per-epoch dice, configs, and the checkpoint to MLflow.
It then scores the pretrained model ("before") against the fine-tuned one ("after")
on validation cases, and logs an MLflow model that `scripts/register_model.py`
registers in Unity Catalog. Everything is validated on CPU first, so no GPU time
goes to config errors. Visualisation and registration run on CPU in the notebook.

| Model (`--model`) | Classes | Workflows | Validated on AI Runtime |
|---|---|---|---|
| `nv-segment-ctmr` | 345 (CT + MRI) | `softmax` (default), `standard` | Not yet |
| `nv-segment-ct` | 124 (CT) | `standard` (default), `softmax` | Not yet |

**Validated** = a real `databricks air run` produced a checkpoint. CPU tests
cover scaffolding, config resolution, label checks, and data staging. This
does not imply clinical validation.

## When to Use

- A workshop notebook or user needs the training project for NV-Segment-CT or NV-Segment-CTMR, and nothing exists yet.
- The user wants to fine-tune VISTA3D on CT or MRI segmentation data stored in a UC Volume.
- The user must choose between a fixed softmax head and promptable VISTA3D, or map dataset labels to VISTA3D class IDs.
- A dataset has extra or leftover label values, or class names that are not in the bundle's label dictionary.
- An existing project fails, or ignores epochs or the LR schedule, and needs repair against the upstream bundle contract.
- The user wants a short smoke run before the full job.
- The user wants to see what fine-tuning changed: before/after Dice per class and overlay images.
- The user wants the fine-tuned model registered in their own Unity Catalog schema (`<catalog>.ws_<user>`).

## Prerequisites

- **Tools**: Python with `PyYAML` for `scripts/` (any Databricks compute; no GPU, MONAI, or network). The Databricks CLI (`databricks air`) only to submit.
- **Inputs**: a UC Volume with NIfTI image/label pairs (default naming `ct_XXXX_image.nii.gz` / `ct_XXXX_label.nii.gz`, configurable) or an MSD datalist JSON; an MLflow experiment path; a workspace `PROJECT_DIR`.
- **Environment**: AI Runtime access. `train.yaml` uses env `'5'` (torch 2.9, CUDA 12.9) plus `monai==1.4.0` and `numpy<2`. `torch` is never listed.
- **Skill location**: `/Workspace/Users/<user>/.assistant/skills/nv-segment-finetune/` (or `/Workspace/.assistant/skills/...`); `SKILL_DIR` below.

## Quick Start

```python
import json, subprocess, sys
SKILL_DIR = f"/Workspace/Users/{USERNAME}/.assistant/skills/nv-segment-finetune"
res = subprocess.run([sys.executable, f"{SKILL_DIR}/scripts/scaffold_project.py", PROJECT_DIR,
                      "--model", "nv-segment-ctmr", "--dataset-path", DATASET_PATH,
                      "--experiment-name", EXPERIMENT_NAME, "--check-dataset"],
                     capture_output=True, text=True)
report = json.loads(res.stdout)
print(report["ok"], report["workflow"], {k: v["action"] for k, v in report["files"].items()})
print(report["validation"]["warnings"], report["next_step"])
```

## Workflow

### Step 1: Resolve inputs and pick the model

Ask for `DATASET_PATH` if it is unknown. Run
`scripts/scaffold_project.py --list-models` to show the options. Default
`GPU_1xH100`; use `GPU_8xH100` only for large datasets.

```python
USERNAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
```

### Step 2: Choose the workflow and label mapping

- **`softmax`**: a fixed, mutually exclusive class set. Every class ID must
  exist in the bundle (`assets/label_ids/<model>.json`). Produces `model_softmax.pt`.
- **`standard`**: promptable VISTA3D. IDs absent from the bundle (`< 255`)
  are learned from scratch. Produces `model_finetune.pt`.
- Mapping format: `{"<dataset label>": [<class id>, "<name>"]}`. Foreground
  only, in channel order. The default is the 7-class ImageCHD mapping (CTMR IDs).

Read `references/monai-bundle-contract.md` before choosing mappings for a new dataset.

### Step 3: Scaffold the project

Run `scripts/scaffold_project.py` (see Quick Start). Add `--workflow` and
`--label-mapping '<json>'` if needed. Existing files are **kept**. Never
re-type or shorten `train_segment_finetune.py`. Pass `--force` only when the
user asks to overwrite.

```python
subprocess.run([sys.executable, f"{SKILL_DIR}/scripts/scaffold_project.py", PROJECT_DIR, "--model", "nv-segment-ct",
                "--workflow", "standard", "--dataset-path", DATASET_PATH, "--experiment-name", EXPERIMENT_NAME],
               capture_output=True, text=True)
```

### Step 4: Gate on validation

`report["ok"]` must be `True`. Fix every failed `validation.checks[]` entry
and show `validation.warnings` to the user (e.g. "class IDs … are learned
from scratch"). Re-check with:

```python
subprocess.run([sys.executable, f"{SKILL_DIR}/scripts/validate_project.py", PROJECT_DIR, "--check-dataset"],
               capture_output=True, text=True)
```

### Step 5: Dry-run the resolved config (CPU)

```python
subprocess.run([sys.executable, "train_segment_finetune.py", "--config", "finetune_config.yml", "--print-config"],
               cwd=PROJECT_DIR, capture_output=True, text=True)
```

### Step 6: Submit (only when the user asks)

```bash
cd PROJECT_DIR && databricks air run --file train.yaml           # add --dry-run first to validate the schema
```

From a notebook, install `databricks-air` (`%pip install databricks-air`,
then `%restart_python`) and run `subprocess.run(["air", "run", "--file", "train.yaml"], cwd=PROJECT_DIR, env=os.environ.copy())`.
Export auth for the subprocess first: `DATABRICKS_HOST` = `ctx.apiUrl().get()`,
`DATABRICKS_TOKEN` = `ctx.apiToken().get()`. Never use `--watch` from a notebook cell.

### Step 7: Look at the data and the before/after results (CPU)

`scripts/visualize.py` renders everything on CPU from files or MLflow
artifacts. Use it before training (`show_case`) and after the job finishes:

```python
import sys; sys.path.insert(0, f"{SKILL_DIR}/scripts")
import visualize as viz
viz.show_case(f"{DATASET_PATH}/ct_1001_image.nii.gz", f"{DATASET_PATH}/ct_1001_label.nii.gz")
eval_dir = viz.load_evaluation(RUN_ID)          # MLflow artifacts: evaluation/dice.json + slices_*.npz
display(viz.dice_table(eval_dir)); viz.plot_dice(eval_dir); viz.plot_before_after(eval_dir)
```

"Before" is the pretrained model on the same validation cases. In the softmax
workflow it is zero-shot VISTA3D (`configs/inference.json` with class prompts),
so a class unknown to the pretrained model shows no "before" mask.

### Step 8: Register in Unity Catalog (CPU)

Each practitioner registers into their own schema, `<catalog>.ws_<user>`. The
script creates the schema if needed, registers the run's MLflow model, sets
the alias `workshop`, and tags the version with the pins and Dice.

```python
from register_model import register_run          # in-process: uses the notebook's credentials
register_run(RUN_ID, CATALOG, dry_run=True)       # preview <catalog>.ws_<user>.<model> and the model URI
register_run(RUN_ID, CATALOG)                     # create the schema if needed, register, alias "workshop"
```

### Step 9: Report

Report the model, workflow, validator result and warnings, the run URL, the
before/after Dice, and the registered model name and version. Stop there.
Serving is a separate request.

## Key Parameters

| Parameter (`finetune_config.yml`) | Default | Range / Options | Effect |
|-----------|---------|-----------------|--------|
| `model.id` | `nv-segment-ctmr` | `nv-segment-ctmr`, `nv-segment-ct` | Bundle, weights, label dictionary |
| `model.workflow` | per model | `softmax`, `standard` | Config stack, class-ID rules, checkpoint name |
| `dataset.label_mapping` | ImageCHD 7-class | dict / JSON file | Ordered output classes |
| `dataset.image_glob`, `label_from`, `label_to` | `ct_*_image.nii.gz`, `_image.nii.gz`, `_label.nii.gz` | any | How image/label pairs are found |
| `dataset.datalist` | `null` | MSD JSON | Use your own folds (fold 0 = validation) |
| `dataset.check_labels` | `true` | bool | Zeroes label values not in the mapping |
| `dataset.max_cases` | `null` | int ≥ 4 | Smoke runs |
| `training.epochs` / `learning_rate` | `50` / `5e-5` | `1`-`200` / `1e-5`-`1e-4` | Also drive the warmup-cosine schedule |
| `training.patch_size` | `[96,96,96]` | 64³-160³ | Context vs GPU memory |
| `training.cache_rate` | `1.0` | `0`-`1` | Host RAM used for cached volumes |
| `training.auto_seg` | `false` | bool | Standard only: class-prompt-only training |
| `training.num_gpus` | `auto` | int | > 1 launches `torchrun` with `multi_gpu_train.json` |
| `evaluation.max_cases` | `3` | `0`-`10` | Validation cases scored before/after; `0` skips |
| `mlflow.log_model` | `true` | bool | Log the pyfunc model that Unity Catalog registers |

## Common Recipes

### Recipe: 3-minute smoke job

When to use: check the pipeline end to end before the full run.

```yaml
dataset:
  max_cases: 6
training:
  epochs: 2
```

### Recipe: Your own folds or file naming

When to use: an MSD-style dataset (`imagesTr/`, `labelsTr/`) or a patient-level split.

```yaml
dataset:
  src_dir: /Volumes/cat/sch/vol/Task09_Spleen
  datalist: /Volumes/cat/sch/vol/Task09_Spleen/datalist.json   # {"training": [{"image","label","fold"}]}
  label_mapping: {"1": [3, "spleen"]}
```

### Recipe: In-notebook smoke test on GPU compute

When to use: the notebook runs on GPU compute and MLflow should use notebook auth.

```python
import os, sys, runpy
os.chdir(PROJECT_DIR)
sys.argv = ["train_segment_finetune.py", "--config", "finetune_config.yml", "--max-cases", "4", "--epochs", "1"]
runpy.run_path("train_segment_finetune.py", run_name="__main__")
```

## Expected Outputs

- `PROJECT_DIR/` holding `train.yaml`, `finetune_config.yml`, `train_segment_finetune.py` (~40 KB), and `segment_pyfunc.py`.
- Scaffold/validator JSON: `{"ok", "model_id", "workflow", "files": {...}, "validation": {"checks", "warnings"}, "next_step"}`.
- After the job, the MLflow run holds:
  - params: model, workflow, pins, label mapping
  - metrics: `val_mean_dice` and `train_loss` per epoch, `best_val_mean_dice`, `best_val_epoch`
  - artifacts: `checkpoints/model_{softmax|finetune}.pt`, `configs/{override,label_mapping,datalist}.json`, `run_summary.json`, `train.log`
  - before/after: metrics `eval_before_mean_dice` / `eval_after_mean_dice`; artifacts `evaluation/dice.json` and `evaluation/slices_<case>.npz`
  - model: `model/` (MLflow pyfunc: input `image_path`, outputs `label_path` + `class_voxels`), tagged `segment.model_uri`
- After registration: `<catalog>.ws_<user>.<model_id>_finetuned`, alias `workshop`.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `softmax needs existing bundle class IDs` | Mapping uses IDs absent from this bundle | Use `standard`, pick IDs from `assets/label_ids/<model>.json`, or switch to `nv-segment-ctmr` |
| `standard workflow needs class IDs < 255` | CTMR IDs ≥ 255 | Use `softmax` |
| `label mapping must contain positive ...` | Padding such as `[8, 0]` | Remove it; `check_labels: true` zeroes stray values |
| `dataset image/label pairs` check fails | Wrong path or naming | Fix `src_dir`, `image_glob`, `label_from`/`label_to`, or use `datalist` |
| Check `torch is not listed` fails | `torch` in dependencies | Remove it |
| `INTERNAL_ERROR` at provisioning | GPU capacity | Retry, or try `GPU_1xA100` |
| CUDA OOM | Patch or cache too large | Smaller `patch_size`; lower `cache_rate` |
| MLflow `PERMISSION_DENIED` from a notebook | Subprocess lacks notebook auth | Export `DATABRICKS_HOST`/`TOKEN`, or use `runpy` |
| `run_summary.json`: "best checkpoint is from epoch 0" | No improvement over the initial weights | Check the mapping, labels, and LR before trusting the checkpoint |
| No "before" mask for a class | The pretrained model does not know that class ID | Expected for new classes; compare "after" against ground truth |
| `before/after evaluation failed` warning | Inference config error on the node | Read `evaluation_runs/*.log` in `train.log`; training results are unaffected |
| `Cannot create <catalog>.ws_<user>` | Missing `USE CATALOG` / `CREATE SCHEMA` | Ask the instructor to grant them on the workshop catalog |
| `run ... has status 'FAILED'` on registration | Training or checkpoint failed | Register only a successful run |

## Guardrails

1. Never hand-write or shorten the launcher. Always produce the project with `scripts/scaffold_project.py`.
2. Never patch upstream bundle files. Change behaviour through `finetune_config.yml` only.
3. Do not overwrite existing project files without an explicit request (`--force`).
4. Do not submit `databricks air run` or start GPU training without confirmation. Do not retry a failing submission more than 3 times.
5. Never print full tokens. Mask them, e.g. `****abcd`.
6. Register only into the practitioner's own schema (`ws_<user>`) unless told otherwise; never overwrite another user's model.
7. Engineering use only: no clinical, diagnostic, or treatment-planning claims. Tell the user to check each model's license on its Hugging Face card.

## Bundled Resources

- `scripts/scaffold_project.py` — `--list-models`; writes the 3 project files from `assets/` and validates them; JSON on stdout.
- `scripts/validate_project.py` — CPU-only checks of `train.yaml` pins, the config, class IDs against `assets/label_ids/`, and (optionally) the dataset.
- `scripts/visualize.py` — CPU plots: `show_case`, `load_evaluation`, `dice_table`, `plot_dice`, `plot_before_after`.
- `scripts/register_model.py` — creates `<catalog>.ws_<user>` and registers the run's model in Unity Catalog; JSON on stdout.
- `assets/train_segment_finetune.py` — GPU-node launcher with the `MODELS` registry, before/after evaluation and model logging.
- `assets/segment_pyfunc.py` — MLflow pyfunc wrapper around the upstream inference configs (models-from-code).
- `assets/train.yaml`, `assets/finetune_config.yml` — templates; `assets/label_ids/*.json` — bundle class IDs.
- `references/monai-bundle-contract.md` — workflows, override keys, label rules, outputs.
- `references/adding-a-model.md` — how to register another MONAI bundle.

## References

- [NVIDIA-Medtech/NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR) — bundles, `docs/finetune.md`, `docs/softmax_finetune.md`
- [nvidia/NV-Segment-CTMR](https://huggingface.co/nvidia/NV-Segment-CTMR) and [nvidia/NV-Segment-CT](https://huggingface.co/nvidia/NV-Segment-CT) — weights and licenses
- [NVIDIA-Medtech/medical-AI-skills](https://github.com/NVIDIA-Medtech/medical-AI-skills) — `nv-segment-ct-finetune` reference wrapper
- [MONAI bundle docs](https://docs.monai.io/en/stable/bundle_intro.html) — `monai.bundle run`
- [AI Runtime CLI YAML reference](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/yaml-config) — `train.yaml` fields
- [Genie Code skills](https://docs.databricks.com/aws/en/genie-code/skills) — skill installation
