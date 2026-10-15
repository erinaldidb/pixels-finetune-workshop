#!/usr/bin/env python3
"""Generate synthetic CT-like NIfTI pairs for the nv-segment-finetune eval (task seg-003).

Writes ct_XXXX_image.nii.gz / ct_XXXX_label.nii.gz with the four heart chambers
(labels 1-4). Some label files also get leftover values 5-9 that the skill must
zero, not map. No patient data.

Usage (Databricks notebook or local):
    python generate_data.py /Volumes/<catalog>/<schema>/eval/nv_segment/chambers_stray_labels
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

SHAPE = (64, 64, 48)          # small enough to write quickly, big enough for 32³ patches
SPACING = (1.5, 1.5, 1.5)     # matches the bundle's resample_to_spacing
N_CASES = 12
STRAY_EVERY = 3               # every 3rd case carries leftover labels 5-9
SEED = 0


def make_case(rng: np.random.Generator, with_stray: bool):
    img = rng.normal(-200, 150, SHAPE).astype(np.int16)  # HU-like background
    lbl = np.zeros(SHAPE, np.uint8)
    x, y, z = (s // 2 for s in SHAPE)
    for c, (dx, dy) in enumerate([(-8, -8), (8, -8), (-8, 8), (8, 8)], start=1):  # 4 chambers
        lbl[x + dx - 5:x + dx + 5, y + dy - 5:y + dy + 5, z - 6:z + 6] = c
        img[lbl == c] = 300 + 20 * c                                              # contrast-filled blood pool
    if with_stray:
        for v in range(5, 10):
            lbl[2 + v, 2:4, 2:4] = v
    return img, lbl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    affine = np.diag([*SPACING, 1.0])
    for i in range(N_CASES):
        img, lbl = make_case(rng, with_stray=(i % STRAY_EVERY == 0))
        nib.save(nib.Nifti1Image(img, affine), str(out / f"ct_{2001 + i}_image.nii.gz"))
        nib.save(nib.Nifti1Image(lbl, affine), str(out / f"ct_{2001 + i}_label.nii.gz"))
    print(f"Wrote {N_CASES} pairs to {out}")


if __name__ == "__main__":
    main()
