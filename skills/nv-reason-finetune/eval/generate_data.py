#!/usr/bin/env python3
"""Generate synthetic NV-Reason manifests + images for the nv-reason-finetune eval.

Writes, under OUT_DIR:
  cxr/images/*.jpg + cxr/sft.jsonl + cxr/grpo.jsonl   (NV-Reason-CXR row format)
  ct/volumes/*.nii.gz + ct/sft.jsonl + ct/grpo.jsonl  (NV-Reason-CT row format)
Images are noise and answers are placeholders: the data exercises the
pipeline, not model quality. No patient data.

Usage (Databricks notebook or local; needs numpy, nibabel, Pillow):
    python generate_data.py /Volumes/<catalog>/<schema>/eval/nv_reason
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

N_CASES = 32              # enough for an 8-step smoke run at 1 example per GPU on 8 GPUs (half an epoch)
CXR_SIZE = 512            # matches the upstream MIMIC-CXR images_512 example
CT_SHAPE = (96, 96, 64)   # small, HU-range volumes; the processor resamples and crops
CXR_LABELS = ["No Finding", "Lung Opacity", "Pleural Effusion", "Cardiomegaly"]
CT_LABELS = ["No Chest Finding", "Atelectasis", "Pleural effusion"]  # must exist in upstream train/vlm_labels.py
SEED = 0


def rows(image: str, prompt: str, answer: str, extra: dict) -> tuple:
    user = {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    sft = {**extra, "image": image, "solution": answer,
           "messages": [user, {"role": "assistant", "content": [
               {"type": "text", "text": f"<think>Synthetic reasoning.</think>\n<answer>{answer}</answer>"}]}]}
    grpo = {**extra, "image": image, "solution": answer, "messages": [user]}
    return sft, grpo


def write_jsonl(path: Path, items: list):
    path.write_text("".join(json.dumps(r) + "\n" for r in items))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    out = Path(ap.parse_args().out_dir)
    rng = np.random.default_rng(SEED)

    cxr = out / "cxr"
    (cxr / "images").mkdir(parents=True, exist_ok=True)
    sft, grpo = [], []
    for i in range(N_CASES):
        name = f"cxr_{i:04d}.jpg"
        Image.fromarray((rng.random((CXR_SIZE, CXR_SIZE)) * 255).astype(np.uint8)).save(cxr / "images" / name)
        s, g = rows(name, "Find abnormalities and support devices.", CXR_LABELS[i % len(CXR_LABELS)],
                    {"id": f"cxr-{i:04d}", "subject_id": str(10000 + i), "study_id": str(50000 + i)})
        sft.append(s)
        grpo.append(g)
    write_jsonl(cxr / "sft.jsonl", sft)
    write_jsonl(cxr / "grpo.jsonl", grpo)

    ct = out / "ct"
    (ct / "volumes").mkdir(parents=True, exist_ok=True)
    sft, grpo = [], []
    affine = np.diag([1.5, 1.5, 3.0, 1.0])
    for i in range(N_CASES):
        name = f"volumes/ct_{i:04d}.nii.gz"
        vol = rng.normal(-700, 150, CT_SHAPE).astype(np.int16)  # air/lung-like background in HU
        nib.save(nib.Nifti1Image(vol, affine), str(ct / name))
        s, g = rows(name, "Write a structured chest CT report.", CT_LABELS[i % len(CT_LABELS)],
                    {"id": f"ct-{i:04d}", "anatomy_region": "chest"})
        sft.append(s)
        grpo.append(g)
    write_jsonl(ct / "sft.jsonl", sft)
    write_jsonl(ct / "grpo.jsonl", grpo)
    print(f"Wrote {N_CASES} CXR and {N_CASES} CT cases under {out}")


if __name__ == "__main__":
    main()
