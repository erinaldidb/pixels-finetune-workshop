#!/usr/bin/env python3
"""CPU-only validation of an AI Runtime NV-Segment (MONAI bundle) fine-tuning project.

Checks train.yaml, finetune_config.yml and train_segment_finetune.py against
the model registry (MODELS in the launcher) and the bundle label IDs in
assets/label_ids/, without importing MONAI or torch, and without a GPU. Progress goes to stderr; a JSON report goes
to stdout.

    python scripts/validate_project.py PROJECT_DIR [--check-dataset]

Exit code 0 = all checks passed, 1 = at least one failure, 2 = bad arguments.
"""
import argparse
import contextlib
import glob
import importlib.util
import json
import os
import py_compile
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # PyYAML is the only non-stdlib dependency
    raise SystemExit("PyYAML is required: pip install PyYAML")

SKILL_DIR = Path(__file__).resolve().parents[1]
LABEL_IDS_DIR = SKILL_DIR / "assets" / "label_ids"
WORKLOAD_FILE = "train.yaml"
CONFIG_FILE = "finetune_config.yml"
SCRIPT_FILE = "train_segment_finetune.py"
PYFUNC_FILE = "segment_pyfunc.py"  # MLflow model wrapper, logged by the launcher
REQUIRED_MARKERS = ("--config", "_load_yaml_config", "resolve_config", "MODELS")
FORBIDDEN_DEPS = ("torch", "torchvision")  # come with the AI Runtime environment; pinning them breaks CUDA builds
GPUS_PER_TYPE = {"GPU_1xH100": 1, "GPU_1xA100": 1, "GPU_8xH100": 8}
REQUIRED_ARG_RE = re.compile(r"add_argument\([^)]*required\s*=\s*True")
PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\}\}")
SMOKE_EPOCHS = 1  # CLI override used to prove CLI > YAML > defaults


class Report:
    def __init__(self):
        self.checks, self.warnings = [], []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(("  ✓ " if ok else "  ✗ ") + name + (f" — {detail}" if detail else ""), file=sys.stderr)
        return ok

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  ⚠ {msg}", file=sys.stderr)

    @property
    def failures(self):
        return [c for c in self.checks if not c["ok"]]


def _dep_name(dep: str) -> str:
    return re.split(r"[=<>!~\[ ]", dep.strip(), maxsplit=1)[0].lower().replace("_", "-")


def load_launcher(script: Path):
    spec = importlib.util.spec_from_file_location("train_segment_finetune", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # heavy imports (monai, mlflow, torch) are lazy inside functions
    return mod


def validate_train_yaml(rep: Report, path: Path, model: dict):
    print(f"\n[{WORKLOAD_FILE}] {path}", file=sys.stderr)
    try:
        wl = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        rep.check(False, "train.yaml parses", exc)
        return
    rep.check(not PLACEHOLDER_RE.search(path.read_text()), "no unfilled {{PLACEHOLDER}} in train.yaml")
    rep.check(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(wl.get("experiment_name", "")))),
              "experiment_name is a valid task key", wl.get("experiment_name"))
    env = wl.get("environment") or {}
    rep.check(isinstance(env.get("version"), str), "environment.version is a quoted string", repr(env.get("version")))
    deps = [str(d).replace(" ", "") for d in env.get("dependencies") or []]
    names = {_dep_name(d) for d in deps}
    rep.check(not names & set(FORBIDDEN_DEPS), "torch is not listed in dependencies")
    for req in model["dependencies"]:
        if any(op in req for op in ("==", ">=", "<")):
            rep.check(req.replace(" ", "") in deps, f"dependency {req}")
        else:
            rep.check(_dep_name(req) in names, f"dependency {req}")

    compute = wl.get("compute") or {}
    gpu_type, n_acc = compute.get("accelerator_type"), compute.get("num_accelerators")
    rep.check(gpu_type in GPUS_PER_TYPE, "compute.accelerator_type supported", gpu_type)
    if gpu_type in GPUS_PER_TYPE:
        rep.check(n_acc == GPUS_PER_TYPE[gpu_type], "num_accelerators matches accelerator_type", f"{n_acc} for {gpu_type}")
    rep.check((wl.get("code_source") or {}).get("type") == "snapshot", "code_source.type is snapshot")
    command = wl.get("command") or ""
    rep.check("$CODE_SOURCE_PATH" in command, "command cds into $CODE_SOURCE_PATH")
    rep.check(SCRIPT_FILE in command and CONFIG_FILE in command, f"command runs {SCRIPT_FILE} --config {CONFIG_FILE}")
    rep.check("--watch" not in command, "command does not use --watch")


def validate_segment_data(rep: Report, mod, cfg: dict, check_dataset: bool):
    try:
        mapping, _ = mod.parse_label_mapping(mod.load_raw_mapping(cfg["label_mapping"]))
        rep.check(True, "label mapping valid", json.dumps(mapping))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        rep.check(False, "label mapping valid", exc)
        return
    ids_file = LABEL_IDS_DIR / f"{cfg['model_id']}.json"
    if ids_file.is_file():
        known = {int(k): v for k, v in json.loads(ids_file.read_text())["labels"].items()}
        try:
            for w in mod.check_mapping_against_bundle(mapping, cfg["workflow"], known):
                rep.warn(w)
            rep.check(True, f"class IDs valid for {cfg['model_id']} {cfg['workflow']}")
        except ValueError as exc:
            rep.check(False, f"class IDs valid for {cfg['model_id']} {cfg['workflow']}", exc)
    if not check_dataset:
        return
    print(f"\n[dataset] {cfg['src_dir']}", file=sys.stderr)
    if cfg["datalist"]:
        rep.check(os.path.isfile(cfg["datalist"]), "datalist exists", cfg["datalist"])
    else:
        imgs = glob.glob(os.path.join(cfg["src_dir"], cfg["image_glob"]))
        lbls = [i.replace(cfg["label_from"], cfg["label_to"]) for i in imgs]
        n_pairs = sum(1 for i, l in zip(imgs, lbls) if l != i and os.path.exists(l))
        rep.check(n_pairs > 0 and n_pairs == len(imgs), "dataset image/label pairs",
                  f"{len(imgs)} images, {n_pairs} with labels")
        rep.check(n_pairs >= 2, "at least 2 cases (1 train + 1 validation)", n_pairs)


def validate_script_and_config(rep: Report, script: Path, config: Path, check_dataset: bool):
    print(f"\n[{SCRIPT_FILE}] {script}", file=sys.stderr)
    try:
        py_compile.compile(str(script), doraise=True)
    except py_compile.PyCompileError as exc:
        rep.check(False, "script compiles", exc.msg)
        return None, None
    rep.check(True, "script compiles")
    src = script.read_text()
    for marker in REQUIRED_MARKERS:
        rep.check(marker in src, f"script contains {marker!r}")
    rep.check(not REQUIRED_ARG_RE.search(src), "no argparse required=True")
    mod = load_launcher(script)

    print(f"\n[{CONFIG_FILE}] {config}", file=sys.stderr)
    rep.check(not PLACEHOLDER_RE.search(config.read_text()), "no unfilled {{PLACEHOLDER}} in config")
    try:
        yaml_cfg = mod._load_yaml_config(str(config))
        cfg = mod.resolve_config({}, yaml_cfg)
    except Exception as exc:  # noqa: BLE001 - surface any YAML/registry problem
        rep.check(False, "finetune_config.yml resolves", exc)
        return mod, None
    rep.check(True, "finetune_config.yml resolves", f"{cfg['model_id']} / {cfg['workflow']}")
    missing = mod.check_required(cfg)
    rep.check(not missing, "required inputs present", "; ".join(missing) or "ok")
    rep.check(bool(cfg["mlflow_experiment"]) or cfg["skip_mlflow"], "mlflow.experiment set (or mlflow.skip)")

    merged = mod.resolve_config({"epochs": SMOKE_EPOCHS}, yaml_cfg)
    rep.check(merged["epochs"] == SMOKE_EPOCHS, "CLI override wins", f"epochs={merged['epochs']}")
    validate_segment_data(rep, mod, cfg, check_dataset)
    return mod, cfg


def validate(project_dir: str, check_dataset: bool = False) -> dict:
    """Validate a project directory and return a JSON-serialisable report."""
    root = Path(project_dir)
    rep = Report()
    print(f"Project: {root}", file=sys.stderr)
    files = {name: root / name for name in (WORKLOAD_FILE, CONFIG_FILE, SCRIPT_FILE, PYFUNC_FILE)}
    for name, p in files.items():
        rep.check(p.is_file(), f"{name} exists", f"{p.stat().st_size} bytes" if p.is_file() else "missing")

    mod = cfg = None
    if files[SCRIPT_FILE].is_file() and files[CONFIG_FILE].is_file():
        # The launcher prints progress; keep stdout reserved for the JSON report
        with contextlib.redirect_stdout(sys.stderr):
            mod, cfg = validate_script_and_config(rep, files[SCRIPT_FILE], files[CONFIG_FILE], check_dataset)
    if files[WORKLOAD_FILE].is_file() and cfg:
        validate_train_yaml(rep, files[WORKLOAD_FILE], mod.MODELS[cfg["model_id"]])

    ok = not rep.failures
    print("\n" + ("ALL CHECKS PASSED" if ok else f"{len(rep.failures)} CHECK(S) FAILED"), file=sys.stderr)
    return {
        "project_dir": str(root),
        "model_id": cfg["model_id"] if cfg else None,
        "workflow": cfg["workflow"] if cfg else None,
        "ok": ok,
        "n_failed": len(rep.failures),
        "checks": rep.checks,
        "warnings": rep.warnings,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_dir", help="Directory holding train.yaml, finetune_config.yml, train_segment_finetune.py")
    ap.add_argument("--check-dataset", action="store_true", help="Also check the dataset on disk")
    a = ap.parse_args()
    if not Path(a.project_dir).is_dir():
        ap.error(f"project_dir does not exist: {a.project_dir}")
    result = validate(a.project_dir, a.check_dataset)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
