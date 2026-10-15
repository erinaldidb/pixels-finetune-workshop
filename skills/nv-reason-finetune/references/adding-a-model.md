# Adding a TRL vision-language model

`MODELS` in `assets/train_reason_finetune.py` is the single registry. The
scaffold, the validator, and the GPU-node launcher all read it. This skill
covers upstream repos laid out like NV-Reason-CXR / NV-Reason-CT:
`train/vlm_{sft,grpo}_train.py` scripts parsed by `TrlParser`,
`configs/*_config.yaml`, `accelerate/{ddp,zero2}.yaml`, and a JSONL
`messages` manifest.

1. **Pin everything.** Use a full 40-character `repo_commit` and `hf_revision`.
   Copy version pins from the upstream `requirements*.txt` / README. Never add
   `torch`; check the pins are compatible with AI Runtime env `'5'`
   (torch 2.9, CUDA 12.9).
2. **Add the entry**, mirroring `nv-reason-ct`. Required keys: `family: hf_vlm`,
   `title`, `repo_url`, `repo_commit`, `hf_repo_id`, `hf_revision`,
   `workflows` (`{"sft"|"grpo": {"script", "config", "fixed_args"}}`),
   `data_args`, `manifest_keys`, `deepspeed`, `default_workflow`,
   `default_gpu`, `environment_version`, `dependencies`, `license`.
   - `data_args`: `"path"` for `--dataset_path FILE`, or `"root_and_name"`
     for `--dataset_root DIR --dataset_path FILE`.
   - `manifest_keys`: string keys the upstream `datasets.Features` schema
     requires, with fill values for missing ones.
   - `fixed_args`: upstream flags that must change on AI Runtime (e.g. streaming, vLLM).
3. **Test:** add a tiny synthetic fixture under `tests/fixtures/` and the
   model to `MODEL_ARGS` in `tests/test_reason_finetune.py`. Then run
   `python -m pytest tests -q`.
4. **Document:** add a row to the model table in `SKILL.md`. Mark the model
   *not validated on AI Runtime* until a real GPU run has succeeded.
