# imagechd_micro

Synthetic 16×16×16 CT-like volumes in the ImageCHD naming layout
(`ct_XXXX_image.nii.gz` / `ct_XXXX_label.nii.gz`). No patient data.

- Labels 1–7 are cubes, one per ImageCHD class.
- `ct_1001_label.nii.gz` has one voxel with the unmapped value `9`. This
  exercises the training script's `check_labels` staging.

Used by `tests/` for CPU checks of the scaffold and the data staging only. It is not
valid training data.
