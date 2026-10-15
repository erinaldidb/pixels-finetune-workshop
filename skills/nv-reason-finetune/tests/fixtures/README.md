# Synthetic manifests for tests

No patient data. Images are random noise; texts are placeholders.

- `reason_cxr_micro/`: 3 × 64×64 JPGs in `images/`. `sft.jsonl` and
  `grpo.jsonl` follow the NV-Reason-CXR `datalists/` row format.
  `subject_id`/`study_id` are integers, which exercises string casting.
- `reason_ct_micro/`: 3 × 16³ NIfTI volumes in `volumes/`. `sft.jsonl` and
  `grpo.jsonl` follow the NV-Reason-CT format, with `anatomy_region` and
  `No Chest Finding` labels.

Used by `tests/` for CPU checks of manifest validation and command building
only. Not valid training data.
