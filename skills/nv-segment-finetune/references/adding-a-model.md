# Adding a MONAI bundle model

`MODELS` in `assets/train_segment_finetune.py` is the single registry. The
scaffold, the validator, and the GPU-node launcher all read it. This skill
covers models trained through `python -m monai.bundle run` configs with the
VISTA3D continual-learning layout (`train.json`, `train_continual.json`,
`train_continual_softmax.json`, `multi_gpu_train.json`). A bundle with a
different training layout (e.g. NV-Generate-CTMR's MAISI diffusion and VAE
fine-tuning) needs its own skill.

1. **Pin everything.** Use a full 40-character `repo_commit` and `hf_revision`.
   Keep `monai==1.4.0` unless the upstream bundle's `metadata.json` says otherwise.
   Never add `torch` (the AI Runtime env provides it).
2. **Add the entry**, mirroring `nv-segment-ct`. Required keys: `family:
   monai_bundle`, `title`, `repo_url`, `repo_commit`, `bundle_subdir`,
   `hf_repo_id`, `hf_revision`, `hf_weights` (path of `model.pt` inside the
   Hugging Face repo), `workflows`, `default_workflow`, `default_gpu`,
   `environment_version`, `dependencies`, `license`.
3. **Workflows:** reuse `_SEGMENT_WORKFLOWS` when the bundle ships the same
   configs. Otherwise define `{"configs": [...], "checkpoint": "<file>.pt",
   "defaults": {...}}` per workflow.
4. **Label IDs:** export the bundle's `configs/label_dict.json` to
   `assets/label_ids/<model-id>.json` as `{"source": ..., "labels": {"<id>": "<name>"}}`.
   The validator then checks class IDs offline.
5. **Test:** add the model to `MODEL_ARGS` in `tests/test_segment_finetune.py`,
   then run `python -m pytest tests -q`.
6. **Document:** add a row to the model table in `SKILL.md`. Mark the model
   *not validated on AI Runtime* until a real GPU run has succeeded.
