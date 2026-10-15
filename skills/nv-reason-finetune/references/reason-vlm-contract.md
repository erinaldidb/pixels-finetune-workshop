# Reasoning VLM fine-tuning contract — NV-Reason-CXR / NV-Reason-CT

Read this before preparing a manifest, changing hyperparameters, or editing the
launcher `assets/train_reason_finetune.py`. Sources:
[NVIDIA-Medtech/NV-Reason-CXR@83a4d51](https://github.com/NVIDIA-Medtech/NV-Reason-CXR/tree/83a4d51c9fbbff68156a5f01796f04e26519b6ad)
and [NVIDIA-Medtech/NV-Reason-CT@b4abd35](https://github.com/NVIDIA-Medtech/NV-Reason-CT/tree/b4abd354fef90819ec40dd637c3bd14521b8023c).
These are Hugging Face TRL models, not MONAI bundles. NV-Reason-CT uses MONAI
inside its 3D image processor.

## Pins and environment

| | `nv-reason-cxr` | `nv-reason-ct` |
|---|---|---|
| Weights | `nvidia/NV-Reason-CXR-3B@056bd03…` (Qwen2.5-VL-3B) | `nvidia/NV-Reason-CT@386b93e…` (Qwen3.5-4B + 3D encoder, `trust_remote_code`) |
| Input | 2D JPG/PNG chest X-rays | 3D CT NIfTI in Hounsfield units |
| Pins | `transformers==4.56.1`, `trl==0.22.2` | `transformers==5.6.2`, `trl==1.2.0`, `accelerate==1.13.0`, `datasets==4.8.4`, `liger-kernel==0.7.0`, `monai>=1.5.1` |
| DeepSpeed (multi-GPU) | `deepspeed` | `deepspeed==0.18.9` |
| License | NVIDIA OneWay **Noncommercial** | OpenMDW-1.1 |
| Upstream hardware | 8× A100 80 GB | 8 GPUs |

AI Runtime environment `'5'` provides torch 2.9 + CUDA 12.9 and flash-attn
2.8.3; the pins above go on top. DeepSpeed builds against the installed torch,
so `train.yaml` installs it in `command` with `--no-build-isolation` rather
than in `environment.dependencies`.

## Launch

The launcher downloads the weights at the pinned revision (the upstream
scripts do not pass a revision to `from_pretrained`). It then runs, from the
repo checkout:

```bash
accelerate launch --config_file accelerate/{ddp|zero2}.yaml --num_processes N --num_machines 1 \
  train/vlm_{sft|grpo}_train.py --config configs/{sft|grpo}_config.yaml \
  --model_name_or_path <local weights> --output_dir <out>/model --image_dir <image_dir> \
  --report_to mlflow --run_name <run> <data args> <overrides>
```

- 1 GPU → `ddp.yaml`; more than 1 → `zero2.yaml`.
- Data args: CXR `--dataset_root DIR --dataset_path FILE`; CT `--dataset_path FILE`.
- Fixed overrides: CXR `--dataset_streaming false` (streaming needs `max_steps`); CXR GRPO `--use_vllm false` (vLLM 0.10.1.1 pins torch 2.7.1; generation falls back to transformers, which is slower).
- `training.epochs/learning_rate/max_steps/per_device_train_batch_size/gradient_accumulation_steps` map to the TRL flags. `training.extra_args` passes any other upstream flag, e.g. `{freeze_llm: true}` (CT), `{attn_implementation: sdpa}`.
- Note: the CXR README cites `configs/vlm_sft_config.yaml`, but the repo file is `configs/sft_config.yaml`. The launcher uses the real file.

## Manifest (JSONL)

SFT rows end with the assistant answer. GRPO rows hold the user turn only, plus `solution`.

```json
{"id": "case-001", "image": "case-001.jpg", "solution": "Lung Opacity",
 "messages": [{"role": "user", "content": [{"type": "image", "image": "case-001.jpg"},
                                            {"type": "text", "text": "Find abnormalities and support devices."}]},
              {"role": "assistant", "content": [{"type": "text", "text": "<think>...</think><answer>Lung Opacity</answer>"}]}]}
```

- Image paths are relative to `dataset.image_dir`.
- CXR: the upstream loader uses a fixed `datasets.Features` schema (`id`,
  `image`, `subject_id`, `study_id`, `solution`, `messages`). The launcher fills
  missing keys with `""` and casts them to strings.
- CT: add `"anatomy_region": "chest" | "abdomen"`. GRPO `solution` labels must
  come from upstream `train/vlm_labels.py`. No-finding cases use exactly
  `No Chest Finding` / `No Abdominal Finding`.
- Upstream example data: MIMIC-CXR-JPG and CheXpert (credentialed access);
  CT-RATE (gated on Hugging Face). Comply with each dataset's terms.

## Before / after evaluation

`evaluation.max_cases` rows (default 4) are held out from the end of the
manifest. They are excluded from training only when the manifest has more
than twice that many rows. After training, each held-out row is answered by
the pretrained weights ("before") and the fine-tuned `model/` ("after"):

- NV-Reason-CT: upstream `inference.py IMAGE --model DIR --region <anatomy_region> --prompt <user prompt>`, one subprocess per case.
- NV-Reason-CXR: the README generation snippet (`AutoModelForImageTextToText`
  + `AutoProcessor`, greedy, bf16). The repo ships no inference CLI.

Scoring: findings F1 between the last `<answer>...</answer>` set and the
row's `solution` (case-insensitive, comma-separated). This is a rough
engineering signal, not a report-quality metric.
- Outputs: `evaluation/generations.jsonl` (prompt, reference, before, after, per-case F1).
- Metrics: `eval_before_findings_f1`, `eval_after_findings_f1`.
- `scripts/show_eval.py` renders the table and per-case texts on CPU.

## Outputs and tracking

- `<output_dir>/model/` holds the final model and processor, loadable with the
  upstream `inference.py --model`. `checkpoint-*` dirs hold optimizer state
  and are not uploaded.
- The launcher creates the MLflow run and passes `MLFLOW_RUN_ID` to the
  transformers `MLflowCallback` (rank 0), so per-step loss, learning rate, and
  reward metrics land in the same run.
- Launcher params use a `medtech.` prefix to avoid clashing with the
  callback's TrainingArguments params.
- With `mlflow.log_model: true`, the final model files are uploaded under
  `model/`. That is several GB.
- Short `max_steps` runs check the pipeline only. Loss need not fall at every
  step. For GRPO, watch `reward` and `reward_std`.
