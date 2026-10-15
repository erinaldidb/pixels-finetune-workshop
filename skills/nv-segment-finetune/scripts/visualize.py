#!/usr/bin/env python3
"""CPU-only visualisation helpers for NV-Segment fine-tuning (import from a notebook).

    import sys; sys.path.insert(0, f"{SKILL_DIR}/scripts")
    import visualize as viz
    viz.show_case(image_path, label_path, class_names)          # look at the data
    eval_dir = viz.load_evaluation(run_id)                       # download evaluation/ from MLflow
    viz.dice_table(eval_dir)                                     # per-class before/after Dice
    viz.plot_dice(eval_dir); viz.plot_before_after(eval_dir)     # charts

Needs matplotlib, numpy, pandas; nibabel for show_case; mlflow for load_evaluation.
Run as a script for a quick PNG: python visualize.py EVAL_DIR --out before_after.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Dataviz reference categorical palette, dark-mode steps (validated on #1a1a19: lightness band,
# chroma floor, CVD separation, contrast). CT slices render on a dark background, so the dark
# steps apply. Fixed order: class i always gets slot i; never cycled.
CLASS_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BEFORE_AFTER_COLORS = {"before": "#2a78d6", "after": "#eb6834"}  # light-mode slots 1-2 for the bar chart
INK = {"primary": "#1a1a19", "secondary": "#5f5e57", "on_dark": "#ffffff"}
OVERLAY_ALPHA = 0.55  # keeps anatomy visible under the mask
HU_WINDOW = (-200, 600)  # same soft-tissue window the launcher stores


def _overlay_rgba(mask, labels):
    from matplotlib.colors import to_rgba
    rgba = np.zeros(mask.shape + (4,))
    for i, lab in enumerate(labels):
        if i >= len(CLASS_COLORS):
            break  # more classes than palette slots: fold into "other" rather than invent hues
        rgba[mask == lab] = to_rgba(CLASS_COLORS[i], OVERLAY_ALPHA)
    return rgba


def _show(ax, image, mask, labels, title):
    ax.imshow(np.rot90(image), cmap="gray", vmin=HU_WINDOW[0], vmax=HU_WINDOW[1])
    if mask is not None:
        ax.imshow(np.rot90(_overlay_rgba(mask, labels)))
    ax.set_title(title, color=INK["primary"], fontsize=11)
    ax.axis("off")


def _legend(fig, labels, names):
    from matplotlib.patches import Patch
    handles = [Patch(color=CLASS_COLORS[i], label=names.get(str(lab), str(lab)))
               for i, lab in enumerate(labels[: len(CLASS_COLORS)])]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False,
               labelcolor=INK["primary"], fontsize=9)


def show_case(image_path: str, label_path: str, class_names: dict | None = None, view: str = "axial"):
    """Show one CT volume with its label overlay at the slice with the most foreground."""
    import matplotlib.pyplot as plt
    import nibabel as nib

    image = np.asanyarray(nib.as_closest_canonical(nib.load(image_path)).dataobj)
    label = np.asanyarray(nib.as_closest_canonical(nib.load(label_path)).dataobj).astype(np.int32)
    axis = {"axial": 2, "coronal": 1, "sagittal": 0}[view]
    fg = (label > 0).sum(axis=tuple(a for a in range(3) if a != axis))
    idx = int(fg.argmax()) if fg.any() else label.shape[axis] // 2
    img2d, lab2d = np.take(image, idx, axis=axis), np.take(label, idx, axis=axis)
    labels = [int(v) for v in np.unique(label) if v != 0]
    names = {str(k): v for k, v in (class_names or {}).items()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    _show(axes[0], img2d, None, labels, f"CT ({view} slice {idx})")
    _show(axes[1], img2d, lab2d, labels, "Ground-truth label")
    _legend(fig, labels, names)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    counts = {names.get(str(l), str(l)): int((label == l).sum()) for l in labels}
    print(f"shape {image.shape}, HU range [{image.min():.0f}, {image.max():.0f}], label voxels: {counts}")
    return fig


def load_evaluation(run_id: str, dst: str | None = None) -> Path:
    """Download the run's evaluation/ artifacts (dice.json + slices_*.npz) and return the local dir."""
    import mlflow
    mlflow.set_tracking_uri("databricks")
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="evaluation", dst_path=dst)
    return Path(path)


def _report(eval_dir) -> dict:
    return json.loads((Path(eval_dir) / "dice.json").read_text())


def dice_table(eval_dir):
    """Per-class mean Dice before vs after fine-tuning (table view of plot_dice)."""
    import pandas as pd
    rep = _report(eval_dir)
    rows = []
    for lab, name in rep["classes"].items():
        before, after = rep["mean"]["before"].get(lab), rep["mean"]["after"].get(lab)
        rows.append({"class": name, "before": before, "after": after,
                     "change": None if before is None or after is None else after - before})
    df = pd.DataFrame(rows).set_index("class")
    return df.round(3)


def plot_dice(eval_dir):
    """Horizontal grouped bars: per-class mean Dice, pretrained (before) vs fine-tuned (after)."""
    import matplotlib.pyplot as plt
    df = dice_table(eval_dir)
    y = np.arange(len(df))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 1.4))
    for offset, phase in ((-height / 2, "before"), (height / 2, "after")):
        vals = df[phase].fillna(0).to_numpy()
        ax.barh(y + offset, vals, height=height * 0.92, color=BEFORE_AFTER_COLORS[phase],
                label="pretrained (before)" if phase == "before" else "fine-tuned (after)")
    for i, v in enumerate(df["after"].to_numpy()):  # direct-label the outcome only
        if v == v:  # skip NaN
            ax.text(v + 0.01, i + height / 2, f"{v:.2f}", va="center", fontsize=8, color=INK["secondary"])
    ax.set_yticks(y, df.index, color=INK["primary"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Mean Dice on validation cases (higher is better)", color=INK["secondary"])
    ax.grid(axis="x", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)  # above the plot, never over bars
    ax.set_title("Dice per class: before vs after fine-tuning", color=INK["primary"], loc="left", pad=30)
    fig.tight_layout()
    return fig


def plot_before_after(eval_dir, case: str | None = None, view: str = "axial"):
    """CT | ground truth | pretrained (before) | fine-tuned (after) for one validation case."""
    import matplotlib.pyplot as plt
    eval_dir = Path(eval_dir)
    rep = _report(eval_dir)
    case = case or sorted(rep["cases"])[0]
    data = np.load(eval_dir / f"slices_{case}.npz")
    labels = [int(l) for l in rep["classes"]]
    scores = rep["cases"].get(case, {})

    def mean_dice(phase):
        vals = [v for v in scores.get(phase, {}).values() if v is not None]
        return f" — Dice {np.mean(vals):.2f}" if vals else " — n/a"

    image = data[f"{view}_image"].astype(np.float32)
    panels = [("CT", None), ("Ground truth", data[f"{view}_gt"])]
    for phase, title in (("before", "Pretrained (before)"), ("after", "Fine-tuned (after)")):
        key = f"{view}_{phase}"
        panels.append((title + mean_dice(phase), data[key] if key in data else None))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax, (title, mask) in zip(axes, panels):
        _show(ax, image, mask, labels, title)
    _legend(fig, labels, rep["classes"])
    fig.suptitle(f"{rep['model_id']} ({rep['workflow']}) — {case}, {view} slice {int(data[f'{view}_index'])}",
                 color=INK["primary"])
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def list_cases(eval_dir) -> list:
    return sorted(_report(eval_dir)["cases"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", help="Local evaluation/ directory (from load_evaluation)")
    ap.add_argument("--case", default=None)
    ap.add_argument("--view", default="axial", choices=["axial", "coronal"])
    ap.add_argument("--out", default="before_after.png")
    a = ap.parse_args()
    if not (Path(a.eval_dir) / "dice.json").is_file():
        ap.error(f"{a.eval_dir} has no dice.json")
    import matplotlib
    matplotlib.use("Agg")
    plot_before_after(a.eval_dir, a.case, a.view).savefig(a.out, dpi=120)
    plot_dice(a.eval_dir).savefig(Path(a.out).with_name("dice_" + Path(a.out).name), dpi=120)
    print(dice_table(a.eval_dir).to_string())


if __name__ == "__main__":
    main()
