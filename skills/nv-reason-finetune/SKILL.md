---
name: nv-reason-finetune
description: NVIDIA NV-Reason-CXR and NV-Reason-CT (reasoning vision-language models) SFT and GRPO fine-tuning on Databricks AI Runtime serverless GPU. Scaffolds and validates the 3-file databricks air project (train.yaml, finetune_config.yml, train_reason_finetune.py) that runs the upstream TRL scripts through accelerate on a JSONL messages manifest with chest X-ray images or CT NIfTI volumes in a Unity Catalog Volume, with MLflow metrics and the final model. Load when the user mentions NV-Reason-CXR, NV-Reason-CT, radiology report VLM fine-tuning, chest X-ray or CT reasoning, SFT or GRPO for medical VLMs, or TRL on AI Runtime. For NV-Segment (VISTA3D) segmentation use nv-segment-finetune; for UC registration and Model Serving use the Databricks ML training and Model Serving skills.
category: Others
summary: Fine-tune NVIDIA reasoning VLMs (NV-Reason-CXR, NV-Reason-CT) with SFT or GRPO on AI Runtime and compare answers before/after.
version: 0.2.0
author: Emanuele Rinaldi
license: Databricks License
---

# NV-Reason Fine-Tuning on AI Runtime

## Overview

This skill creates and checks an AI Runtime CLI project that fine-tunes an
NVIDIA reasoning VLM with the upstream TRL scripts: `train/vlm_sft_train.py`
or `train/vlm_grpo_train.py` via `accelerate launch`. The launcher in
`assets/` validates the JSONL manifest, pins the upstream repo and Hugging
Face weights, applies the AI Runtime-specific overrides, and logs step
metrics plus the final model to MLflow. Held-out rows are answered by the base and
fine-tuned model for a before/after comparison. Everything is validated on CPU first.

| Model (`--model`) | Input | Base | Workflows | License | Validated on AI Runtime |
|---|---|---|---|---|---|
| `nv-reason-cxr` | 2D chest X-ray (JPG/PNG) | Qwen2.5-VL-3B | `sft` (default), `grpo` | NVIDIA OneWay **Noncommercial** | Not yet |
| `nv-reason-ct` | 3D CT NIfTI (HU) | Qwen3.5-4B + 3D encoder | `sft` (default), `grpo` | OpenMDW-1.1 | Not yet |

**Validated** = a real `databricks air run` produced a model. CPU tests cover
scaffolding, config resolution, manifest checks, and the upstream command
line. This does not imply clinical validation.

## When to Use

- The user wants to adapt NV-Reason-CXR or NV-Reason-CT to their own reports, findings, or prompts.
- A workshop or user needs an AI Runtime project for SFT (reference answers) or GRPO (verified finding labels).
- The user has a JSONL manifest plus images or volumes in a UC Volume, and wants it checked before spending GPU hours.
- The user must pick a GPU shape (1 vs 8 H100), or freeze model parts to fit memory.
- An existing TRL/accelerate job fails on AI Runtime (wandb, streaming, vLLM, DeepSpeed install, pinned versions).
- The user wants a short `max_steps` pipeline check before a full run.
- The user wants to compare generated reports before and after fine-tuning.

## Prerequisites

- **Tools**: Python with `PyYAML` for `scripts/` (any Databricks compute; no GPU, torch, or network). The Databricks CLI (`databricks air`) only to submit.
- **Inputs**: a JSONL manifest in the upstream `datalists/{sft,grpo}.jsonl` format (see `references/reason-vlm-contract.md`); an image root (`--image-dir`) that the relative paths resolve against; an MLflow experiment path; a workspace `PROJECT_DIR`.
- **Data access**: the upstream examples use MIMIC-CXR-JPG and CheXpert (credentialed access) and CT-RATE (gated on Hugging Face). The user must hold the rights to any data used.
- **Environment**: AI Runtime env `'5'` (torch 2.9, CUDA 12.9, flash-attn 2.8.3) plus the per-model pins from the registry. On multiple GPUs, DeepSpeed is installed in `command`.
- **Skill location**: `/Workspace/Users/<user>/.assistant/skills/nv-reason-finetune/` (or `/Workspace/.assistant/skills/...`); `SKILL_DIR` below.

## Quick Start

```python
import json, subprocess, sys
SKILL_DIR = f"/Workspace/Users/{USERNAME}/.assistant/skills/nv-reason-finetune"
res = subprocess.run([sys.executable, f"{SKILL_DIR}/scripts/scaffold_project.py", PROJECT_DIR,
                      "--model", "nv-reason-cxr", "--manifest", MANIFEST, "--image-dir", IMAGE_DIR,
                      "--experiment-name", EXPERIMENT_NAME, "--check-dataset"],
                     capture_output=True, text=True)
report = json.loads(res.stdout)
print(report["ok"], report["gpu_type"], report["license"])
print(report["validation"]["warnings"], report["next_step"])
```

## Workflow

### Step 1: Resolve inputs and pick the model

Run `scripts/scaffold_project.py --list-models`. Ask for the manifest and
image root if they are unknown. Tell the user the model's license; NV-Reason-CXR is
non-commercial.

```python
USERNAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
```

### Step 2: Choose workflow and GPU shape

- **`sft`**: each row ends with the assistant answer to imitate.
- **`grpo`**: rows hold the user turn only, plus a `solution` finding set
  used by the reward. For NV-Reason-CT, labels must come from upstream
  `train/vlm_labels.py`.
- **GPU**: the default is `GPU_8xH100` (upstream recipes use 8 GPUs, DeepSpeed ZeRO-2).
  `GPU_1xH100` works for NV-Reason-CXR smoke runs. For NV-Reason-CT on one
  GPU, expect OOM unless you freeze parts: `training.extra_args: {freeze_llm: true}`.

### Step 3: Scaffold the project

Run `scripts/scaffold_project.py` (see Quick Start), adding `--workflow` and
`--gpu-type` when needed. Existing files are **kept**. Never write a custom
training loop or edit the upstream configs. Pass `--force` only when the user
asks to overwrite.

```python
subprocess.run([sys.executable, f"{SKILL_DIR}/scripts/scaffold_project.py", PROJECT_DIR, "--model", "nv-reason-ct",
                "--workflow", "grpo", "--manifest", MANIFEST, "--image-dir", IMAGE_DIR,
                "--experiment-name", EXPERIMENT_NAME, "--gpu-type", "GPU_8xH100", "--check-dataset"],
               capture_output=True, text=True)
```

### Step 4: Gate on validation

`report["ok"]` must be `True`. Common failures: the manifest schema doesn't
match the workflow, or images are missing under `image_dir`. Show
`validation.warnings` to the user. If an SFT manifest is given for GRPO,
offer to derive GRPO rows (drop the assistant turn, keep `solution`); do not
bypass the check.

```python
subprocess.run([sys.executable, f"{SKILL_DIR}/scripts/validate_project.py", PROJECT_DIR, "--check-dataset"],
               capture_output=True, text=True)
```

### Step 5: Dry-run the resolved config (CPU)

```python
subprocess.run([sys.executable, "train_reason_finetune.py", "--config", "finetune_config.yml", "--print-config"],
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

### Step 7: Compare answers before and after (CPU)

```python
import sys; sys.path.insert(0, f"{SKILL_DIR}/scripts")
import show_eval
eval_dir = show_eval.load_evaluation(RUN_ID)     # evaluation/generations.jsonl
display(show_eval.f1_table(eval_dir))
show_eval.show_case(eval_dir, 0, image_dir=IMAGE_DIR)
```

Findings F1 compares the `<answer>` label set with `solution`. It is a rough
signal: read the texts too.

### Step 8: Report

Report the model, workflow, GPU shape, license, validator result and
warnings, the run URL, and the before/after findings F1. Stop there. Unity
Catalog registration of the final model (`model/` artifacts) and serving are
separate requests.

## Key Parameters

| Parameter (`finetune_config.yml`) | Default | Range / Options | Effect |
|-----------|---------|-----------------|--------|
| `model.id` | — | `nv-reason-cxr`, `nv-reason-ct` | Upstream repo, weights, pins |
| `model.workflow` | `sft` | `sft`, `grpo` | Upstream script and config |
| `dataset.manifest` / `image_dir` | — | absolute paths | Rows and image root |
| `dataset.max_cases` | `null` | int | Subset rows for a pipeline check |
| `training.max_steps` | `null` | e.g. `8` | Overrides epochs; smoke runs |
| `training.epochs` / `learning_rate` | `null` (upstream: 1–2 epochs, `2e-5`) | | `--num_train_epochs` / `--learning_rate` |
| `training.per_device_train_batch_size`, `gradient_accumulation_steps` | `null` (upstream) | | For GRPO, keep the generation batch divisible by `num_generations` |
| `training.extra_args` | `{}` | any upstream flag | e.g. `{freeze_llm: true}`, `{attn_implementation: sdpa}` |
| `training.num_gpus` | `auto` | int | 1 → `ddp.yaml`; > 1 → `zero2.yaml` (DeepSpeed) |
| `evaluation.max_cases` | `4` | `0`-`32` | Rows held out of training for before/after; `0` skips |
| `mlflow.log_model` | `true` | bool | Upload the final model and processor (several GB) |

## Common Recipes

### Recipe: 8-step pipeline check

When to use: confirm data loading, the model download, and MLflow before a full run.

```yaml
dataset:
  max_cases: 32
training:
  max_steps: 8
```

### Recipe: Continue GRPO from an SFT run

When to use: the upstream two-stage recipe. Point `extra_args` at the SFT model directory you copied to a Volume, and use a new output directory.

```yaml
model: {id: nv-reason-ct, workflow: grpo}
training:
  extra_args: {model_name_or_path: /Volumes/cat/sch/vol/runs/nv_reason_ct_sft/model}
```

### Recipe: Parameter-efficient adaptation on one GPU (NV-Reason-CT)

When to use: only a single H100 is available.

```yaml
training:
  extra_args: {freeze_llm: true}   # trains the 3D encoder + projector only
```

## Expected Outputs

- `PROJECT_DIR/` holding `train.yaml` (per-model pins; DeepSpeed install in `command` on 8 GPUs), `finetune_config.yml`, and `train_reason_finetune.py` (~22 KB).
- Scaffold/validator JSON: `{"ok", "model_id", "workflow", "gpu_type", "license", "files", "validation": {"checks", "warnings"}, "next_step"}`.
- After the job, the MLflow run holds:
  - from the transformers MLflowCallback: per-step `loss`, `learning_rate`, `grad_norm` (GRPO: `reward`, `reward_std`) and TrainingArguments params
  - from the launcher: `reason.*` params, `training_time_s`, `run_summary.json`, `manifest.jsonl`, `train.log`, and `model/` (final weights and processor)
  - before/after: `eval_before_findings_f1` / `eval_after_findings_f1` and `evaluation/generations.jsonl`

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `manifest rows match the sft/grpo schema` fails | Wrong row shape for the workflow | SFT ends with an assistant turn; GRPO has the user turn only plus `solution` |
| `referenced images exist` fails | Paths not relative to `image_dir` | Set `--image-dir` to the root the manifest paths assume |
| CUDA OOM | Full fine-tuning too large for the GPU shape | `GPU_8xH100`, `extra_args {freeze_llm: true}`, smaller batch |
| `flash_attention_2` import error | Pinned transformers vs the env's flash-attn build | `extra_args {attn_implementation: sdpa}` |
| DeepSpeed install fails | Built in an isolated env without torch | Keep the `pip install --no-build-isolation` line in `command`; never add it to `dependencies` |
| GRPO very slow (CXR) | vLLM disabled (it pins torch 2.7.1) | Expected on AI Runtime; reduce `num_generations` or `max_completion_length` via `extra_args` |
| `wandb` login prompt | Upstream configs default to W&B | The launcher forces `--report_to mlflow`; do not remove it |
| MLflow param conflict error | Same param logged with two values | Keep the launcher's `reason.` prefix; do not log TrainingArguments by hand |

## Guardrails

1. Never write a custom training loop or edit upstream scripts or configs. Always go through `scripts/scaffold_project.py` and `finetune_config.yml`.
2. Surface the model license (NV-Reason-CXR-3B is non-commercial) and the dataset access terms before scaffolding.
3. Do not overwrite existing project files without an explicit request (`--force`).
4. Do not submit `databricks air run` or start GPU training without confirmation. Do not retry a failing submission more than 3 times.
5. Never print full tokens. Mask them, e.g. `****abcd`.
6. Engineering use only: no clinical, diagnostic, or treatment-planning claims about generated reports.

## Bundled Resources

- `scripts/scaffold_project.py` — `--list-models`; writes the 3 project files from `assets/` and validates them; JSON on stdout.
- `scripts/validate_project.py` — CPU-only checks of `train.yaml` pins, DeepSpeed setup, the config, manifest rows, and image paths.
- `scripts/show_eval.py` — CPU helpers: `load_evaluation`, `f1_table`, `show_case`.
- `assets/train_reason_finetune.py` — GPU-node launcher with the `MODELS` registry and the before/after evaluation.
- `assets/train.yaml`, `assets/finetune_config.yml` — templates.
- `references/reason-vlm-contract.md` — upstream CLI, manifest schema, AI Runtime overrides, outputs.
- `references/adding-a-model.md` — how to register another TRL/accelerate model.

## References

- [NVIDIA-Medtech/NV-Reason-CXR](https://github.com/NVIDIA-Medtech/NV-Reason-CXR) — training scripts and configs
- [NVIDIA-Medtech/NV-Reason-CT](https://github.com/NVIDIA-Medtech/NV-Reason-CT) — training scripts, configs, `vlm_labels.py`
- [nvidia/NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B) and [nvidia/NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT) — weights and licenses
- [TRL documentation](https://huggingface.co/docs/trl) — SFT and GRPO trainers
- [AI Runtime CLI YAML reference](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/yaml-config) — `train.yaml` fields
- [Genie Code skills](https://docs.databricks.com/aws/en/genie-code/skills) — skill installation
