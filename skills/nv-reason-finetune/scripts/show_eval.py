#!/usr/bin/env python3
"""CPU-only helpers to compare NV-Reason answers before and after fine-tuning (import from a notebook).

    import sys; sys.path.insert(0, f"{SKILL_DIR}/scripts")
    import show_eval
    eval_dir = show_eval.load_evaluation(run_id)       # download evaluation/ from MLflow
    show_eval.f1_table(eval_dir)                        # one row per held-out case
    show_eval.show_case(eval_dir, 0, image_dir=IMAGE_DIR)  # image + reference / before / after text

Run as a script: python show_eval.py EVAL_DIR
"""
import argparse
import json
from pathlib import Path


def load_evaluation(run_id: str, dst: str | None = None) -> Path:
    """Download the run's evaluation/ artifacts (generations.jsonl) and return the local dir."""
    import mlflow
    mlflow.set_tracking_uri("databricks")
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="evaluation", dst_path=dst))


def load_generations(eval_dir) -> list:
    path = Path(eval_dir) / "generations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _answer(text: str) -> str:
    if not text or "<answer>" not in text:
        return ""
    return text.rsplit("<answer>", 1)[1].split("</answer>", 1)[0].strip()


def f1_table(eval_dir):
    """Findings F1 per held-out case: base model (before) vs fine-tuned (after)."""
    import pandas as pd
    rows = [{"id": r["id"], "solution": r["solution"], "before_answer": _answer(r["before"]),
             "after_answer": _answer(r["after"]), "before_f1": r["before_f1"], "after_f1": r["after_f1"]}
            for r in load_generations(eval_dir)]
    df = pd.DataFrame(rows).set_index("id")
    mean = df[["before_f1", "after_f1"]].mean()
    print(f"Mean findings F1 — before: {mean['before_f1']:.3f}, after: {mean['after_f1']:.3f}")
    return df.round(3)


def _show_image(path: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 5))
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        vol = np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)
        ax.imshow(np.rot90(vol[:, :, vol.shape[2] // 2]), cmap="gray", vmin=-1000, vmax=400)  # lung/mediastinum window
        ax.set_title(f"{path.name} (axial mid-slice)", fontsize=10)
    else:
        from PIL import Image
        ax.imshow(Image.open(path), cmap="gray")
        ax.set_title(path.name, fontsize=10)
    ax.axis("off")
    return fig


def show_case(eval_dir, index: int = 0, image_dir: str | None = None, max_chars: int = 1500):
    """Print prompt, reference and both answers for one case; show the image if image_dir is given."""
    rec = load_generations(eval_dir)[index]
    fig = _show_image(Path(image_dir) / rec["image"]) if image_dir else None
    clip = (lambda t: (t or "")[:max_chars] + ("…" if t and len(t) > max_chars else ""))
    print(f"Case {rec['id']}  |  prompt: {rec['prompt']}\nVerified findings: {rec['solution']}\n")
    if rec.get("reference"):
        print("── Reference answer ──\n" + clip(rec["reference"]) + "\n")
    print(f"── Before fine-tuning (findings F1 {rec['before_f1']:.2f}) ──\n" + clip(rec["before"]) + "\n")
    print(f"── After fine-tuning (findings F1 {rec['after_f1']:.2f}) ──\n" + clip(rec["after"]))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", help="Local evaluation/ directory (from load_evaluation)")
    a = ap.parse_args()
    if not (Path(a.eval_dir) / "generations.jsonl").is_file():
        ap.error(f"{a.eval_dir} has no generations.jsonl")
    print(f1_table(a.eval_dir).to_string())


if __name__ == "__main__":
    main()
