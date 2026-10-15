#!/usr/bin/env python3
"""NV-Segment fine-tuning launcher for Databricks AI Runtime (MONAI bundles).

Fine-tunes the NVIDIA-Medtech VISTA3D segmentation models listed in MODELS
(NV-Segment-CT, NV-Segment-CTMR) through their upstream, documented entry
point: `python -m monai.bundle run` (single GPU) or `torchrun` (multi-GPU).
This script only stages data, pins upstream code and weights, launches
training, and logs metrics, configs and checkpoints to MLflow.

Usage (AI Runtime, from train.yaml):
    cd $CODE_SOURCE_PATH
    python train_segment_finetune.py --config finetune_config.yml

Smoke test / overrides (CLI > YAML > workflow defaults > DEFAULTS):
    python train_segment_finetune.py --config finetune_config.yml --max-cases 4 --epochs 1 --skip-mlflow

CPU-only check (resolve config, no downloads, no GPU):
    python train_segment_finetune.py --config finetune_config.yml --print-config
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

STAGING_MARKER = ".segment_finetune_staging"
MONAI_NUM_CLASSES = 255  # train_continual.json "num_classes": upper bound for standard-workflow class IDs

# ---------------------------------------------------------------------------
# Model registry: the single source of truth for upstream pins, workflows and
# GPU-node dependencies. scaffold_project.py and validate_project.py import it.
# To add a MONAI bundle model, add an entry here (see references/adding-a-model.md).
# ---------------------------------------------------------------------------
_SEGMENT_DEPS = [
    "monai==1.4.0",  # exact: MONAI 1.5.x can crash the VISTA3D finetune loss on boolean labels
    "numpy<2", "pytorch-ignite", "einops", "nibabel", "scipy", "typer", "PyYAML", "fire",
    "huggingface_hub", "tensorboard", "pandas",
    "mlflow>=2.16,<4",  # models-from-code pyfunc logging (segment_pyfunc.py)
]
_SEGMENT_WORKFLOWS = {
    "softmax": {  # fixed-channel, mutually exclusive classes; IDs must exist in label_dict.json
        "configs": ["configs/train_continual_softmax.json"],
        "checkpoint": "model_softmax.pt",
        "defaults": {"epochs": 50, "lr": 5e-5},
    },
    "standard": {  # promptable VISTA3D continual learning; new class IDs < 255 allowed
        "configs": ["configs/train.json", "configs/train_continual.json"],
        "checkpoint": "model_finetune.pt",
        "defaults": {"epochs": 50, "lr": 5e-5},
    },
}
MODELS = {
    "nv-segment-ctmr": {
        "family": "monai_bundle",
        "title": "NV-Segment-CTMR (VISTA3D, CT + MRI, 345 classes)",
        "repo_url": "https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git",
        "repo_commit": "cb921f5c58837c0f42a713855d68b32af88e1cdd",
        "bundle_subdir": "NV-Segment-CTMR",
        "hf_repo_id": "nvidia/NV-Segment-CTMR",
        "hf_revision": "4fb8b4a6b2532be9f1c449a3726fe5440ab4213a",
        "hf_weights": "vista3d_pretrained_model/model.pt",
        "workflows": _SEGMENT_WORKFLOWS,
        "default_workflow": "softmax",
        "default_gpu": "GPU_1xH100",
        "environment_version": "5",
        "dependencies": _SEGMENT_DEPS,
        "license": "NVIDIA model license (see Hugging Face model card)",
    },
    "nv-segment-ct": {
        "family": "monai_bundle",
        "title": "NV-Segment-CT (VISTA3D, CT, 124 classes)",
        "repo_url": "https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git",
        "repo_commit": "cb921f5c58837c0f42a713855d68b32af88e1cdd",
        "bundle_subdir": "NV-Segment-CT",
        "hf_repo_id": "nvidia/NV-Segment-CT",
        "hf_revision": "afb51518689f71e6abb367ee6301b2cd0225c66a",
        "hf_weights": "vista3d_pretrained_model/model.pt",
        "workflows": _SEGMENT_WORKFLOWS,
        "default_workflow": "standard",  # CT label dict has no cardiac chambers; standard can learn new IDs
        "default_gpu": "GPU_1xH100",
        "environment_version": "5",
        "dependencies": _SEGMENT_DEPS,
        "license": "NVIDIA model license (see Hugging Face model card)",
    },
}

# Default label mapping: ImageCHD → NV-Segment-CTMR class IDs (workshop dataset).
# Order matters: entry i becomes output channel i+1 in the softmax workflow.
DEFAULT_LABEL_TO_CTMR = {
    "1": [151, "left ventricle"],
    "2": [152, "right ventricle"],
    "3": [149, "left atrium"],
    "4": [153, "right atrium"],
    "5": [154, "left ventricle myocardium"],
    "6": [6,   "aorta"],
    "7": [171, "pulmonary artery"],
}

# Hardcoded defaults (lowest priority). None = "not set": use the model/workflow
# default.
DEFAULTS = {
    "model_id": "nv-segment-ctmr",
    "workflow": None,
    "repo_commit": None,
    "hf_revision": None,
    # monai_bundle data
    "src_dir": None,
    "image_glob": "ct_*_image.nii.gz",
    "label_from": "_image.nii.gz",
    "label_to": "_label.nii.gz",
    "datalist": None,
    "label_mapping": None,
    "check_labels": True,
    "max_cases": None,
    # training
    "epochs": None,
    "lr": None,
    "val_fraction": 0.15,
    "seed": 42,
    "patch_size": [96, 96, 96],
    "cache_rate": 1.0,
    "auto_seg": False,
    "num_gpus": "auto",
    # paths + tracking
    "output_dir": "/tmp/finetune_output",
    "msd_dir": "/tmp/msd_staging",
    "mlflow_experiment": None,
    "mlflow_run_name": None,
    "skip_mlflow": False,
    "log_model": True,
    # before/after evaluation on the first fold-0 (validation) cases; 0 disables it
    "eval_cases": 3,
}


# ── Config ─────────────────────────────────────────────────────────────────
def _load_yaml_config(path: str) -> dict:
    """Load a YAML config and flatten it into CLI-style keys."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    key_map = {
        ("model", "id"): "model_id",
        ("model", "workflow"): "workflow",
        ("model", "repo_commit"): "repo_commit",
        ("model", "hf_revision"): "hf_revision",
        ("bundle", "commit"): "repo_commit",  # legacy workshop key
        ("dataset", "src_dir"): "src_dir",
        ("dataset", "image_glob"): "image_glob",
        ("dataset", "label_from"): "label_from",
        ("dataset", "label_to"): "label_to",
        ("dataset", "datalist"): "datalist",
        ("dataset", "label_mapping"): "label_mapping",
        ("dataset", "check_labels"): "check_labels",
        ("dataset", "max_cases"): "max_cases",
        ("training", "epochs"): "epochs",
        ("training", "learning_rate"): "lr",
        ("training", "val_fraction"): "val_fraction",
        ("training", "random_seed"): "seed",
        ("training", "patch_size"): "patch_size",
        ("training", "cache_rate"): "cache_rate",
        ("training", "auto_seg"): "auto_seg",
        ("training", "num_gpus"): "num_gpus",
        ("paths", "output_dir"): "output_dir",
        ("paths", "msd_dir"): "msd_dir",
        ("mlflow", "experiment"): "mlflow_experiment",
        ("mlflow", "run_name"): "mlflow_run_name",
        ("mlflow", "skip"): "skip_mlflow",
        ("mlflow", "log_model"): "log_model",
        ("evaluation", "max_cases"): "eval_cases",
    }
    flat = {}
    for (section, key), arg_name in key_map.items():
        if isinstance(raw.get(section), dict):
            val = raw[section].get(key)
            if val is not None:
                flat[arg_name] = val
    return flat


def resolve_config(cli_dict: dict, yaml_cfg: dict) -> dict:
    """Merge CLI > YAML > model/workflow defaults > DEFAULTS and validate the model choice."""
    def pick(key):
        if cli_dict.get(key) is not None:
            return cli_dict[key]
        if yaml_cfg.get(key) is not None:
            return yaml_cfg[key]
        return None

    model_id = pick("model_id") or DEFAULTS["model_id"]
    if model_id not in MODELS:
        raise ValueError(f"unknown model.id {model_id!r}; choose one of {sorted(MODELS)}")
    spec = MODELS[model_id]
    workflow = pick("workflow") or spec["default_workflow"]
    if workflow not in spec["workflows"]:
        raise ValueError(f"{model_id} supports workflows {sorted(spec['workflows'])}, not {workflow!r}")
    wf_defaults = spec["workflows"][workflow].get("defaults", {})

    final = {}
    for key, default in DEFAULTS.items():
        val = pick(key)
        final[key] = val if val is not None else wf_defaults.get(key, default)
    final.update(model_id=model_id, workflow=workflow, family=spec["family"])
    final["repo_commit"] = final["repo_commit"] or spec["repo_commit"]
    final["hf_revision"] = final["hf_revision"] or spec["hf_revision"]
    final["mlflow_run_name"] = final["mlflow_run_name"] or f"{model_id}-{workflow}"

    if isinstance(final["patch_size"], str):
        final["patch_size"] = json.loads(final["patch_size"])
    if isinstance(final["patch_size"], int):
        final["patch_size"] = [final["patch_size"]] * 3
    return final


def check_required(cfg: dict) -> list:
    """Return human-readable errors for missing family-specific inputs."""
    return [] if cfg["src_dir"] else ["dataset.src_dir is required"]


def _str2bool(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


# ── Common helpers ─────────────────────────────────────────────────────────
def _reset_staging_dir(path: str):
    """Recreate a staging dir, refusing to delete anything this script did not create."""
    p = Path(path)
    if p.exists():
        if not (p / STAGING_MARKER).exists() and any(p.iterdir()):
            raise RuntimeError(
                f"{p} exists, is not empty and was not created by this script; "
                "choose a fresh staging path in finetune_config.yml (paths section)"
            )
        shutil.rmtree(p)
    p.mkdir(parents=True)
    (p / STAGING_MARKER).touch()


def clone_repo(url: str, commit: str, checkout_dir: str) -> Path:
    """Clone an upstream repo at a pinned commit (reused if already present)."""
    if not os.path.isdir(os.path.join(checkout_dir, ".git")):
        print(f"Cloning {url} ...")
        subprocess.run(["git", "clone", "--quiet", url, checkout_dir], check=True)
    subprocess.run(["git", "-C", checkout_dir, "checkout", "--quiet", "--detach", commit], check=True)
    head = subprocess.check_output(["git", "-C", checkout_dir, "rev-parse", "HEAD"], text=True).strip()
    print(f"Upstream checkout: {checkout_dir} @ {head[:12]}")
    return Path(checkout_dir)


def detect_num_gpus(requested) -> int:
    if requested not in (None, "auto"):
        return int(requested)
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def _package_versions() -> dict:
    """Record the behaviour-relevant library versions for reproducibility."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": sys.version.split()[0]}
    for pkg in ("torch", "monai", "pytorch-ignite", "nibabel", "numpy", "mlflow"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    try:
        import torch
        versions["cuda"] = torch.version.cuda
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return versions


def write_run_summary(path: Path, cfg: dict, **fields) -> Path:
    summary = {
        "model_id": cfg["model_id"],
        "workflow": cfg["workflow"],
        "repo_commit": cfg["repo_commit"],
        "hf_revision": cfg["hf_revision"],
        **fields,
        "versions": _package_versions(),
    }
    path.write_text(json.dumps(summary, indent=2, default=str))
    for w in fields.get("warnings", []):
        print(f"WARNING: {w}")
    return path


def run_logged(cmd: list, cwd: Path, env: dict, log_path: Path) -> int:
    """Run an upstream command, streaming output to stdout and a log file."""
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def mlflow_context(cfg: dict):
    """Start (or attach to) the MLflow run; returns (context manager, mlflow module or None)."""
    if cfg["skip_mlflow"]:
        print("  MLflow tracking: DISABLED")
        return nullcontext(), None
    import mlflow
    mlflow.set_tracking_uri("databricks")
    # If the runtime already provides a run (MLFLOW_RUN_ID), log into it instead of
    # switching experiments, which MLflow rejects for an existing run.
    if not os.environ.get("MLFLOW_RUN_ID") and cfg["mlflow_experiment"]:
        mlflow.set_experiment(cfg["mlflow_experiment"])
    return mlflow.start_run(run_name=cfg["mlflow_run_name"]), mlflow


def log_common_params(mlflow, cfg: dict):
    mlflow.log_params({
        "segment.model_id": cfg["model_id"],
        "segment.workflow": cfg["workflow"],
        "segment.repo_commit": cfg["repo_commit"],
        "segment.hf_revision": cfg["hf_revision"],
        "segment.num_gpus": cfg["num_gpus"],
    })
    mlflow.set_tags({"segment.model_id": cfg["model_id"], "segment.family": cfg["family"]})


# ── MONAI bundle fine-tuning ───────────────────────────────────────────────
def load_raw_mapping(value) -> dict | list:
    """Resolve label_mapping from None (built-in), a dict/list, a JSON file, or inline JSON."""
    if value is None:
        print("Using built-in ImageCHD label mapping (7 cardiac classes)")
        return DEFAULT_LABEL_TO_CTMR
    if isinstance(value, (dict, list)):
        return value
    if os.path.isfile(value):
        with open(value) as f:
            return json.load(f)
    return json.loads(value)


def parse_label_mapping(raw: dict | list) -> tuple:
    """Convert a label mapping into (ordered [[src, class_id], ...], info dict).

    Accepts {"src_label": [class_id, "name"], ...} or [[src_label, class_id], ...].
    Foreground only: padding entries such as [8, 0] would add bogus softmax
    channels instead of zeroing stray labels (check_labels does that).
    """
    info = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            info[int(k)] = (int(v[0]), str(v[1]) if len(v) > 1 else f"class {v[0]}")
    else:
        for src, class_id in raw:
            info[int(src)] = (int(class_id), f"class {class_id}")
    if not info:
        raise ValueError("label mapping is empty")
    src_labels = list(info)
    class_ids = [c for c, _ in info.values()]
    if any(v <= 0 for v in src_labels + class_ids):
        raise ValueError("label mapping must contain positive foreground labels and class IDs only")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class IDs in the label mapping must be unique")
    return [[src, c] for src, (c, _) in sorted(info.items())], info


def label_index(label_dict: dict) -> dict:
    """Normalise a bundle label_dict.json ({name: id} or {name: {"index": id}}) to {id: name}."""
    out = {}
    for name, v in label_dict.items():
        out[int(v["index"] if isinstance(v, dict) else v)] = name
    return out


def check_mapping_against_bundle(label_mapping: list, workflow: str, known: dict) -> list:
    """Validate class IDs against the bundle label dict; returns warnings, raises on errors."""
    warnings, unknown = [], [c for _, c in label_mapping if c not in known]
    if workflow == "softmax" and unknown:
        raise ValueError(f"softmax needs existing bundle class IDs; not in label_dict.json: {unknown}")
    too_big = [c for _, c in label_mapping if c >= MONAI_NUM_CLASSES]
    if workflow == "standard" and too_big:
        # train_continual.json indexes classes over num_classes=255 global IDs
        raise ValueError(f"standard workflow needs class IDs < {MONAI_NUM_CLASSES}: {too_big}; use softmax")
    if unknown:
        warnings.append(f"class IDs {unknown} are new to this bundle and are learned from scratch")
    return warnings


def collect_pairs(cfg: dict) -> list:
    """Return [(image, label, fold|None)] from a datalist JSON or a glob over src_dir."""
    src = cfg["src_dir"]
    if cfg["datalist"]:
        with open(cfg["datalist"]) as f:
            entries = json.load(f)["training"]
        pairs = [(os.path.join(src, e["image"]), os.path.join(src, e["label"]), e.get("fold")) for e in entries]
    else:
        pairs = []
        for img in sorted(glob.glob(os.path.join(src, cfg["image_glob"]))):
            lbl = img.replace(cfg["label_from"], cfg["label_to"])
            if lbl == img or not os.path.exists(lbl):
                print(f"  SKIP {os.path.basename(img)}: no matching label")
                continue
            pairs.append((img, lbl, None))
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found in {src} (glob {cfg['image_glob']!r})")
    return pairs[: cfg["max_cases"]] if cfg["max_cases"] else pairs


def prepare_msd_dir(pairs: list, msd_dir: str, mapped_labels: set, check_labels: bool = True):
    """Stage pairs in MSD layout (imagesTr/labelsTr); returns staged [(img, lbl, fold)].

    Images are symlinked. Labels are symlinked unless check_labels finds values
    outside the mapping: those voxels become background in a cleaned copy,
    because Relabeld leaves unmapped values untouched.
    """
    images_dir, labels_dir = Path(msd_dir) / "imagesTr", Path(msd_dir) / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if check_labels:
        import nibabel as nib
        import numpy as np
        allowed = np.array(sorted(mapped_labels | {0}))

    staged, n_cleaned = [], 0
    for i, (img_path, lbl_path, fold) in enumerate(pairs):
        case_id = f"case_{i:04d}"
        dst_img, dst_lbl = images_dir / f"{case_id}.nii.gz", labels_dir / f"{case_id}.nii.gz"
        os.symlink(img_path, dst_img)
        staged.append((dst_img, dst_lbl, fold))
        if check_labels:
            lbl = nib.load(lbl_path)
            data = np.asanyarray(lbl.dataobj)
            stray = np.setdiff1d(np.unique(data), allowed)
            if stray.size:
                print(f"  {os.path.basename(lbl_path)}: zeroing unmapped label values {stray.tolist()}")
                cleaned = np.where(np.isin(data, allowed), data, 0).astype(np.uint8)
                nib.save(nib.Nifti1Image(cleaned, lbl.affine, lbl.header), str(dst_lbl))
                n_cleaned += 1
                continue
        os.symlink(lbl_path, dst_lbl)
    print(f"MSD dir ready: {len(staged)} image/label pairs ({n_cleaned} labels cleaned)")
    return staged


def build_datalist(staged: list, datalist_path: str, val_fraction: float, seed: int):
    """Write datalist.json; keep folds from a user datalist, else split (fold 0 = validation)."""
    entries = [{"image": str(i), "label": str(l), "fold": f} for i, l, f in staged]
    if any(e["fold"] is None for e in entries):
        random.seed(seed)
        random.shuffle(entries)
        n_val = max(1, int(len(entries) * val_fraction))
        for i, e in enumerate(entries):
            e["fold"] = 0 if i < n_val else 1
    n_val = sum(1 for e in entries if e["fold"] == 0)
    n_train = len(entries) - n_val
    if n_train == 0 or n_val == 0:
        raise ValueError(f"need training and fold-0 validation cases, got {n_train} train / {n_val} val")
    print(f"Datalist: {n_train} train, {n_val} val")
    with open(datalist_path, "w") as f:
        json.dump({"training": entries}, f, indent=2)
    return n_train, n_val


def stage_bundle(cfg: dict, spec: dict) -> Path:
    """Clone the bundle repo, verify workflow configs, and stage pretrained weights."""
    checkout = clone_repo(spec["repo_url"], cfg["repo_commit"], f"/tmp/{cfg['model_id']}-{cfg['repo_commit'][:7]}")
    bundle_dir = checkout / spec["bundle_subdir"]
    for rel in spec["workflows"][cfg["workflow"]]["configs"] + ["configs/metadata.json", "configs/label_dict.json"]:
        if not (bundle_dir / rel).is_file():
            raise FileNotFoundError(f"{rel} missing in {bundle_dir} at commit {cfg['repo_commit']}")

    model_pt = bundle_dir / "models" / "model.pt"
    if not model_pt.exists():
        from huggingface_hub import hf_hub_download
        print(f"Downloading {spec['hf_repo_id']}@{cfg['hf_revision'][:12]} weights ...")
        src = hf_hub_download(spec["hf_repo_id"], spec["hf_weights"], revision=cfg["hf_revision"])
        model_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, model_pt)
    return bundle_dir


def build_bundle_override(cfg, spec, bundle_dir, datalist_path, label_mapping, ckpt_dir, tb_dir) -> dict:
    """Top-level overrides so derived values (LR schedule, sampling) follow @epochs / @patch_size."""
    override = {
        "bundle_root": str(bundle_dir),
        "dataset_dir": cfg["msd_dir"],
        "data_list_file_path": datalist_path,
        "label_mappings": {"default": label_mapping},
        "epochs": cfg["epochs"],
        "learning_rate": cfg["lr"],
        "patch_size": cfg["patch_size"],
        "patch_size_valid": cfg["patch_size"],
        "train_dataset_cache_rate": cfg["cache_rate"],
        "val_dataset_cache_rate": cfg["cache_rate"],
        "finetune": True,
        "finetune_model_path": str(bundle_dir / "models" / "model.pt"),
        "ckpt_dir": str(ckpt_dir),
        "output_dir": str(tb_dir),
        "val_interval": 1,
        "val_at_start": True,
        "use_tensorboard": True,
        "use_mlflow": False,  # MLflow is owned by this launcher, not the bundle
    }
    if cfg["workflow"] == "standard":
        override.update({"image_key": "image", "label_key": "label", "lr_schedule#activate": False})
        if cfg["auto_seg"]:  # class-prompt-only training, as in the upstream MSD06 tutorial
            override.update({"drop_label_prob": 0.0, "drop_point_prob": 1.0})
    return override


def harvest_tensorboard(tb_dir: Path, mlflow=None) -> dict:
    """Read MONAI TensorBoard scalars; optionally replay them into MLflow."""
    summary = {}
    if not tb_dir.exists():
        return summary
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        name = tag.replace("/", "_").replace(" ", "_")
        if mlflow:
            for e in events:
                mlflow.log_metric(name, e.value, step=e.step)
        print(f"  Logged {len(events)} values for {name}")
        if "val_mean_dice" in tag and events:
            best = max(events, key=lambda e: e.value)
            summary.update(best_val_mean_dice=best.value, best_val_epoch=best.step)
            print(f"  best_val_mean_dice = {best.value:.4f} (epoch {best.step})")
    return summary


# ── Before / after evaluation ──────────────────────────────────────────────
# "Before" is the pretrained model, "after" the fine-tuned checkpoint, both run
# through upstream configs on the same fold-0 (validation) cases:
#   softmax:  before = configs/inference.json with label_prompt (zero-shot VISTA3D),
#             after  = configs/inference_softmax.json with model_softmax.pt
#   standard: before/after = train.json + train_continual.json + configs/evaluate.json
#             with ckpt_path = model.pt / model_finetune.pt (NVIDIA's formal pre/post eval)
HU_WINDOW = (-200.0, 600.0)  # soft-tissue/contrast window used for the stored CT slices


def bundle_command(bundle_dir: Path, configs: list) -> list:
    return [sys.executable, "-m", "monai.bundle", "run",
            "--meta_file", str(bundle_dir / "configs" / "metadata.json"),
            "--logging_file", str(bundle_dir / "configs" / "logging.conf"),
            "--config_file", "[" + ",".join(f"'{c}'" for c in configs) + "]"]


def find_prediction(root: Path, case_id: str) -> Path | None:
    hits = sorted(root.rglob(f"{case_id}_*.nii.gz")) or sorted(root.rglob(f"{case_id}.nii.gz"))
    return hits[0] if hits else None


def to_dataset_labels(pred, label_mapping: list, values_are_class_ids: bool):
    """Map predicted class IDs back to dataset labels; anything else becomes background."""
    import numpy as np
    if not values_are_class_ids:
        return np.where(np.isin(pred, [s for s, _ in label_mapping]), pred, 0).astype(np.uint8)
    out = np.zeros(pred.shape, np.uint8)
    for src, cid in label_mapping:
        out[pred == cid] = src
    return out


def dice_per_class(gt, pred, labels: list) -> dict:
    """Dice per dataset label; None when the class is absent from both volumes."""
    import numpy as np
    scores = {}
    for lab in labels:
        g, p = gt == lab, pred == lab
        denom = int(g.sum()) + int(p.sum())
        scores[str(lab)] = None if denom == 0 else float(2.0 * np.logical_and(g, p).sum() / denom)
    return scores


def key_slices(image, gt, preds: dict) -> dict:
    """Axial and coronal slices through the largest ground-truth cross-section."""
    import numpy as np
    out = {}
    for view, axis in (("axial", 2), ("coronal", 1)):
        fg = (gt > 0).sum(axis=tuple(a for a in range(3) if a != axis))
        idx = int(fg.argmax()) if fg.any() else gt.shape[axis] // 2
        take = (lambda v: np.take(v, idx, axis=axis))
        out[f"{view}_image"] = np.clip(take(image), *HU_WINDOW).astype(np.float16)
        out[f"{view}_gt"] = take(gt).astype(np.uint8)
        for name, vol in preds.items():
            out[f"{view}_{name}"] = take(vol).astype(np.uint8)
        out[f"{view}_index"] = np.array(idx)
    return out


def _load_canonical(path):
    import nibabel as nib
    import numpy as np
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]


def run_predictions(cfg, spec, bundle_dir, cases, label_mapping, ckpt, out, env) -> dict:
    """Run before/after inference with upstream configs; returns {phase: {case_id: path}}."""
    configs_dir, eval_root = bundle_dir / "configs", out / "evaluation_runs"
    class_ids = [c for _, c in label_mapping]
    preds = {"before": {}, "after": {}}
    if cfg["workflow"] == "standard":
        datalist = eval_root / "eval_datalist.json"
        datalist.parent.mkdir(parents=True, exist_ok=True)
        datalist.write_text(json.dumps({"training": [{**c, "fold": 0} for c in cases]}))
        for phase, weights in (("before", bundle_dir / "models" / "model.pt"), ("after", ckpt)):
            override = eval_root / f"{phase}_override.json"
            override.write_text(json.dumps({
                "bundle_root": str(bundle_dir), "dataset_dir": cfg["msd_dir"],
                "data_list_file_path": str(datalist), "label_mappings": {"default": label_mapping},
                "ckpt_path": str(weights), "output_dir": str(eval_root / phase),
                "use_mlflow": False, "use_tensorboard": False,
            }))
            stack = [str(configs_dir / c) for c in ("train.json", "train_continual.json", "evaluate.json")]
            if run_logged(bundle_command(bundle_dir, stack + [str(override)]), bundle_dir, env,
                          eval_root / f"{phase}.log") == 0:
                for c in cases:
                    preds[phase][c["id"]] = find_prediction(eval_root / phase, c["id"])
        return preds

    for c in cases:
        jobs = {
            "before": ("inference.json", {"input_dict": {"image": c["image"], "label_prompt": class_ids},
                                          "checkpoint_path": str(bundle_dir / "models" / "model.pt")}),
            "after": ("inference_softmax.json", {"input_file": c["image"], "checkpoint_path": str(ckpt),
                                                 "label_mappings": {"default": label_mapping},
                                                 "patch_size": cfg["patch_size"]}),
        }
        for phase, (config, extra) in jobs.items():
            override = eval_root / f"{phase}_{c['id']}.json"
            override.parent.mkdir(parents=True, exist_ok=True)
            override.write_text(json.dumps({"bundle_root": str(bundle_dir), "output_postfix": phase,
                                            "output_dir": str(eval_root / phase), **extra}))
            rc = run_logged(bundle_command(bundle_dir, [str(configs_dir / config), str(override)]),
                            bundle_dir, env, eval_root / f"{phase}_{c['id']}.log")
            preds[phase][c["id"]] = find_prediction(eval_root / phase, c["id"]) if rc == 0 else None
    return preds


def evaluate_before_after(cfg, spec, bundle_dir, datalist_path, label_mapping, label_info, ckpt, out, env):
    """Score pretrained vs fine-tuned on validation cases; writes out/evaluation/. Returns (metrics, warnings)."""
    import numpy as np

    entries = json.loads(Path(datalist_path).read_text())["training"]
    cases = [{"id": Path(e["image"]).name.replace(".nii.gz", ""), "image": e["image"], "label": e["label"]}
             for e in entries if e["fold"] == 0][: cfg["eval_cases"]]
    eval_dir = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds = run_predictions(cfg, spec, bundle_dir, cases, label_mapping, ckpt, out, env)
    # softmax inference_softmax.json already writes dataset labels; the rest write class IDs
    ids_phase = {"before": True, "after": cfg["workflow"] == "standard"}

    labels = [s for s, _ in label_mapping]
    report = {"workflow": cfg["workflow"], "model_id": cfg["model_id"],
              "classes": {str(s): n for s, (_, n) in label_info.items()}, "cases": {}, "warnings": []}
    for c in cases:
        gt, zooms = _load_canonical(c["label"])
        image, _ = _load_canonical(c["image"])
        vols, case_rep = {}, {}
        for phase in ("before", "after"):
            path = preds[phase].get(c["id"])
            if path is None:
                report["warnings"].append(f"{c['id']}: no {phase} prediction (see evaluation_runs/*.log)")
                continue
            pred, _ = _load_canonical(path)
            if pred.shape != gt.shape:
                report["warnings"].append(f"{c['id']}: {phase} shape {pred.shape} != label {gt.shape}")
                continue
            vols[phase] = to_dataset_labels(pred, label_mapping, ids_phase[phase])
            case_rep[phase] = dice_per_class(gt, vols[phase], labels)
        report["cases"][c["id"]] = case_rep
        np.savez_compressed(eval_dir / f"slices_{c['id']}.npz", spacing=np.array(zooms, np.float32),
                            **key_slices(image, gt, vols))

    def mean_over_cases(phase):
        per_class = {}
        for lab in map(str, labels):
            vals = [r[phase][lab] for r in report["cases"].values() if phase in r and r[phase][lab] is not None]
            per_class[lab] = float(np.mean(vals)) if vals else None
        return per_class

    report["mean"] = {phase: mean_over_cases(phase) for phase in ("before", "after")}
    metrics = {}
    for phase in ("before", "after"):
        vals = [v for v in report["mean"][phase].values() if v is not None]
        if vals:
            metrics[f"eval_{phase}_mean_dice"] = float(np.mean(vals))
    (eval_dir / "dice.json").write_text(json.dumps(report, indent=2))
    print(f"  Before/after on {len(cases)} validation cases: "
          + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics, report["warnings"]


# ── MLflow model for Unity Catalog ─────────────────────────────────────────
PYFUNC_FILE = "segment_pyfunc.py"  # shipped next to this launcher by scaffold_project.py
MODEL_PIP_REQUIREMENTS = [
    "monai==1.4.0", "torch", "numpy<2", "nibabel", "pytorch-ignite", "einops", "scipy",
    "huggingface_hub", "fire", "PyYAML", "pandas",
]


def log_pyfunc_model(mlflow, cfg: dict, bundle_dir: Path, ckpt: Path, mapping_path: Path, out: Path):
    """Log a pyfunc model (bundle configs + checkpoint + label mapping) that UC can register."""
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

    code = Path(__file__).resolve().with_name(PYFUNC_FILE)
    if not code.exists():
        print(f"WARNING: {PYFUNC_FILE} not found next to the launcher; MLflow model not logged")
        return None
    package = out / "model_package" / "bundle"
    for sub in ("configs", "scripts"):  # code + configs only; weights are a separate artifact
        shutil.copytree(bundle_dir / sub, package / sub, dirs_exist_ok=True)
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "image_path")]),
        outputs=Schema([ColSpec("string", "label_path"), ColSpec("string", "class_voxels")]),
        params=ParamSchema([ParamSpec("output_dir", "string", "/tmp/nv_segment_predictions")]),
    )
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=str(code),
        artifacts={"bundle": str(package), "checkpoint": str(ckpt), "label_mapping": str(mapping_path)},
        model_config={"workflow": cfg["workflow"], "patch_size": cfg["patch_size"],
                      "model_id": cfg["model_id"], "repo_commit": cfg["repo_commit"]},
        signature=signature,
        pip_requirements=MODEL_PIP_REQUIREMENTS,
    )
    mlflow.set_tag("segment.model_uri", info.model_uri)
    print(f"  Logged MLflow model: {info.model_uri}")
    return info.model_uri


def run_monai_bundle(cfg: dict, spec: dict) -> bool:
    label_mapping, label_info = parse_label_mapping(load_raw_mapping(cfg["label_mapping"]))
    for src, (cid, name) in sorted(label_info.items()):
        print(f"    {src} ({name}) -> class {cid}")

    print("\n── Step 1: Stage data ──")
    _reset_staging_dir(cfg["msd_dir"])
    staged = prepare_msd_dir(collect_pairs(cfg), cfg["msd_dir"], set(label_info), cfg["check_labels"])
    datalist_path = os.path.join(cfg["msd_dir"], "datalist.json")
    n_train, n_val = build_datalist(staged, datalist_path, cfg["val_fraction"], cfg["seed"])

    print("\n── Step 2: Stage bundle + weights ──")
    bundle_dir = stage_bundle(cfg, spec)
    known = label_index(json.loads((bundle_dir / "configs" / "label_dict.json").read_text()))
    warnings = check_mapping_against_bundle(label_mapping, cfg["workflow"], known)

    out = Path(cfg["output_dir"])
    ckpt_dir, tb_dir = out / "checkpoints", out / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    override = build_bundle_override(cfg, spec, bundle_dir, datalist_path, label_mapping, ckpt_dir, tb_dir)
    override_path = out / "override.json"
    override_path.write_text(json.dumps(override, indent=2))
    mapping_path = out / "label_mapping.json"
    mapping_path.write_text(json.dumps({
        "label_mappings": {"default": label_mapping},
        "classes": {str(s): {"class_id": c, "name": n} for s, (c, n) in label_info.items()},
    }, indent=2))

    stack = [str(bundle_dir / c) for c in spec["workflows"][cfg["workflow"]]["configs"]]
    if cfg["num_gpus"] > 1:
        stack.append(str(bundle_dir / "configs" / "multi_gpu_train.json"))
    stack.append(str(override_path))
    bundle_args = ["-m", "monai.bundle", "run",
                   "--meta_file", str(bundle_dir / "configs" / "metadata.json"),
                   "--logging_file", str(bundle_dir / "configs" / "logging.conf"),
                   "--config_file", "[" + ",".join(f"'{s}'" for s in stack) + "]"]
    if cfg["num_gpus"] > 1:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={cfg['num_gpus']}"] + bundle_args
    else:
        cmd = [sys.executable] + bundle_args
    env = {**os.environ, "PYTHONPATH": f"{bundle_dir}:{os.environ.get('PYTHONPATH', '')}",
           "NV_SEGMENT_CTMR_ROOT": str(bundle_dir)}

    print("\n── Step 3: Train ──")
    ctx, mlflow = mlflow_context(cfg)
    with ctx:
        if mlflow:
            log_common_params(mlflow, cfg)
            mlflow.log_params({
                "epochs": cfg["epochs"], "learning_rate": cfg["lr"],
                "patch_size": "x".join(map(str, cfg["patch_size"])), "cache_rate": cfg["cache_rate"],
                "n_train": n_train, "n_val": n_val, "n_classes": len(label_info),
                "label_mapping": json.dumps(label_mapping),
            })
        t0 = time.time()
        rc = run_logged(cmd, bundle_dir, env, out / "train.log")
        elapsed = time.time() - t0
        print(f"\nTraining exited with code {rc} after {elapsed:.0f}s")

        metrics = harvest_tensorboard(tb_dir, mlflow)
        ckpt = ckpt_dir / spec["workflows"][cfg["workflow"]]["checkpoint"]
        ok = rc == 0 and ckpt.exists()
        if rc == 0 and not ckpt.exists():
            print(f"WARNING: {ckpt} was not written")
        if metrics.get("best_val_epoch") == 0:
            # val_at_start=true scores the initial weights at epoch 0; if that stays best,
            # training never improved on them.
            warnings.append("best checkpoint is from epoch 0 (before any training step)")
        if "best_val_mean_dice" not in metrics:
            warnings.append("no val_mean_dice found in TensorBoard events")
        if ok and cfg["eval_cases"]:
            print("\n── Step 4: Before / after evaluation ──")
            try:
                eval_metrics, eval_warnings = evaluate_before_after(
                    cfg, spec, bundle_dir, datalist_path, label_mapping, label_info, ckpt, out, env)
                metrics.update(eval_metrics)
                warnings += eval_warnings
            except Exception as exc:  # evaluation must never fail a finished training run
                warnings.append(f"before/after evaluation failed: {exc}")
        summary = write_run_summary(out / "run_summary.json", cfg, ok=ok, elapsed_s=elapsed,
                                    n_train=n_train, n_val=n_val, label_mapping=label_mapping,
                                    checkpoint=str(ckpt) if ckpt.exists() else None,
                                    warnings=warnings, **metrics)
        if mlflow:
            mlflow.log_metric("training_time_s", elapsed)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for p in (summary, override_path, mapping_path, Path(datalist_path), out / "train.log"):
                mlflow.log_artifact(str(p), "configs" if p.suffix == ".json" and p != summary else None)
            if ok:
                mlflow.log_artifact(str(ckpt), "checkpoints")
            if (out / "evaluation").is_dir():
                mlflow.log_artifacts(str(out / "evaluation"), "evaluation")
            if ok and cfg["log_model"]:
                print("\n── Step 5: Log MLflow model ──")
                log_pyfunc_model(mlflow, cfg, bundle_dir, ckpt, mapping_path, out)
            mlflow.set_tag("status", "SUCCESS" if ok else "FAILED")
            if not ok:
                mlflow.end_run(status="FAILED")
    return ok


# ── CLI ────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune NV-Segment VISTA3D models on AI Runtime",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag defaults to None so YAML values can fill it (never required=True)
    p.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    p.add_argument("--model-id", default=None, choices=sorted(MODELS), help="Model to fine-tune")
    p.add_argument("--workflow", default=None, choices=sorted(_SEGMENT_WORKFLOWS), help="Bundle workflow")
    p.add_argument("--src-dir", default=None, help="NIfTI dataset directory")
    p.add_argument("--label-mapping", default=None, help="Label mapping JSON file or inline JSON")
    p.add_argument("--check-labels", type=_str2bool, default=None, help="Zero label values outside the mapping")
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--patch-size", default=None, help="Patch size, e.g. '[96,96,96]' or 128")
    p.add_argument("--num-gpus", default=None, help="'auto' or an integer")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--mlflow-experiment", default=None, help="MLflow experiment path")
    p.add_argument("--mlflow-run-name", default=None, help="MLflow run name")
    p.add_argument("--skip-mlflow", action="store_true", default=None, help="Disable MLflow tracking")
    p.add_argument("--eval-cases", type=int, default=None, help="Validation cases for before/after evaluation (0 = off)")
    p.add_argument("--print-config", action="store_true", help="Print the resolved config and exit (no GPU)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    yaml_cfg = _load_yaml_config(args.config) if args.config else {}
    cli = {k: v for k, v in vars(args).items() if k not in ("config", "print_config")}
    try:
        cfg = resolve_config(cli, yaml_cfg)
    except ValueError as exc:
        parser.error(str(exc))
    errors = check_required(cfg)
    if errors:
        parser.error("; ".join(errors))
    spec = MODELS[cfg["model_id"]]
    print(f"Model: {spec['title']} | workflow: {cfg['workflow']} | license: {spec['license']}")

    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return

    cfg["num_gpus"] = detect_num_gpus(cfg["num_gpus"])
    _reset_staging_dir(cfg["output_dir"])
    ok = run_monai_bundle(cfg, spec)
    print("\n" + "=" * 60)
    print(("TRAINING COMPLETE" if ok else "TRAINING FAILED — check logs above") +
          f"  ({cfg['model_id']} / {cfg['workflow']})")
    print(f"  Summary: {Path(cfg['output_dir']) / 'run_summary.json'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
