#!/usr/bin/env python3
"""Scaffold an AI Runtime project that fine-tunes an NV-Segment VISTA3D model.

Writes train.yaml, finetune_config.yml, train_segment_finetune.py and segment_pyfunc.py into
PROJECT_DIR from this skill's assets/ and model registry, then runs the
CPU-only validator. Existing files are kept unless --force is given. A JSON
report goes to stdout; progress goes to stderr.

    python scripts/scaffold_project.py --list-models
    python scripts/scaffold_project.py PROJECT_DIR --model nv-segment-ctmr \
        --dataset-path /Volumes/cat/sch/vol/ImageCHD_dataset \
        --experiment-name /Users/me@company.com/Segment-Finetune \
        [--workflow softmax|standard] [--label-mapping JSON] [--gpu-type GPU_1xH100] [--check-dataset]

Exit code 0 = project written and valid, 1 = validation failed, 2 = bad arguments.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = SKILL_DIR / "assets"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_project import CONFIG_FILE, GPUS_PER_TYPE, PYFUNC_FILE, SCRIPT_FILE, WORKLOAD_FILE, validate  # noqa: E402

TIMEOUT_MINUTES = 180  # full 50-epoch workshop run; smoke runs finish far sooner


def load_registry():
    spec = importlib.util.spec_from_file_location("train_segment_finetune", ASSETS_DIR / SCRIPT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(template: str, values: dict) -> str:
    for key, val in values.items():
        template = template.replace("{{" + key + "}}", str(val))
    return template


def write_file(dst: Path, content: str, force: bool, reference_sha: str | None = None) -> dict:
    """Write content to dst unless it already exists (and --force is not set)."""
    if dst.exists() and not force:
        info = {"action": "kept", "bytes": dst.stat().st_size}
        if reference_sha is not None:
            info["matches_reference"] = _sha256(dst) == reference_sha
        print(f"  kept     {dst}", file=sys.stderr)
        return info
    action = "overwritten" if dst.exists() else "created"
    dst.write_text(content)
    print(f"  {action:<8} {dst}", file=sys.stderr)
    return {"action": action, "bytes": dst.stat().st_size, "sha256": _sha256(dst)}


def render_workload(model_id: str, spec: dict, workflow: str, gpu_type: str, job_label: str) -> str:
    n = GPUS_PER_TYPE[gpu_type]
    deps = "\n".join(f"    - {d}" for d in spec["dependencies"])
    return _render((ASSETS_DIR / WORKLOAD_FILE).read_text(), {
        "MODEL_TITLE": spec["title"], "WORKFLOW": workflow, "AIR_JOB_LABEL": job_label,
        "ENV_VERSION": spec["environment_version"], "DEPENDENCIES": deps, "NUM_GPUS": n,
        "GPU_TYPE": gpu_type, "PRE_COMMAND": "", "TIMEOUT": TIMEOUT_MINUTES,
    })


def render_config(mod, model_id: str, spec: dict, workflow: str, args) -> str:
    values = {
        "MODEL_TITLE": spec["title"], "MODEL_ID": model_id, "WORKFLOW": workflow,
        "REPO_COMMIT": spec["repo_commit"], "HF_REVISION": spec["hf_revision"],
        "EXPERIMENT_NAME": args.experiment_name, "RUN_NAME": f"{model_id}-{workflow}",
    }
    mapping = args.label_mapping or json.dumps(mod.DEFAULT_LABEL_TO_CTMR)
    values.update(DATASET_PATH=args.dataset_path, LABEL_MAPPING=mapping)
    return _render((ASSETS_DIR / CONFIG_FILE).read_text(), values)


def missing_config_inputs(spec: dict, args) -> list:
    need = ["experiment_name", "dataset_path"]
    return [f"--{n.replace('_', '-')}" for n in need if not getattr(args, n)]


def scaffold(args, mod) -> dict:
    model_id = args.model
    spec = mod.MODELS[model_id]
    workflow = args.workflow or spec["default_workflow"]
    if workflow not in spec["workflows"]:
        raise SystemExit(f"{model_id} supports --workflow {sorted(spec['workflows'])}, not {workflow!r}")
    gpu_type = args.gpu_type or spec["default_gpu"]
    project = Path(args.project_dir)
    project.mkdir(parents=True, exist_ok=True)
    files = {}

    ref_script = ASSETS_DIR / SCRIPT_FILE
    files[SCRIPT_FILE] = write_file(project / SCRIPT_FILE, ref_script.read_text(), args.force, _sha256(ref_script))
    ref_pyfunc = ASSETS_DIR / PYFUNC_FILE
    files[PYFUNC_FILE] = write_file(project / PYFUNC_FILE, ref_pyfunc.read_text(), args.force, _sha256(ref_pyfunc))

    cfg_dst = project / CONFIG_FILE
    if cfg_dst.exists() and not args.force:
        files[CONFIG_FILE] = write_file(cfg_dst, "", force=False)
    else:
        missing = missing_config_inputs(spec, args)
        if missing:
            raise SystemExit(f"{CONFIG_FILE} is missing: pass {' '.join(missing)} to create it for {model_id}")
        files[CONFIG_FILE] = write_file(cfg_dst, render_config(mod, model_id, spec, workflow, args), args.force)

    files[WORKLOAD_FILE] = write_file(
        project / WORKLOAD_FILE, render_workload(model_id, spec, workflow, gpu_type, args.job_label), args.force)
    if (project / "train_ctmr_finetune.py").exists():
        print("  note     legacy train_ctmr_finetune.py found; the new train.yaml does not use it", file=sys.stderr)

    print("\nValidating project...", file=sys.stderr)
    validation = validate(str(project), check_dataset=args.check_dataset)
    if validation["model_id"] and validation["model_id"] != model_id:
        validation["warnings"].append(
            f"existing {CONFIG_FILE} targets {validation['model_id']}, not --model {model_id}; use --force to regenerate")
    return {
        "project_dir": str(project),
        "model_id": model_id,
        "workflow": workflow,
        "gpu_type": gpu_type,
        "license": spec["license"],
        "files": files,
        "validation": validation,
        "ok": validation["ok"],
        "next_step": f"cd {project} && databricks air run --file {WORKLOAD_FILE}" if validation["ok"] else
                     "fix the failed checks in validation.checks, then re-run this script",
    }


def list_models(mod) -> list:
    return [{"id": k, "title": v["title"], "workflows": sorted(v["workflows"]),
             "default_workflow": v["default_workflow"], "default_gpu": v["default_gpu"], "license": v["license"]}
            for k, v in mod.MODELS.items()]


def main():
    mod = load_registry()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_dir", nargs="?", help="Target directory, e.g. /Workspace/Users/<you>/air-segment-finetune")
    ap.add_argument("--list-models", action="store_true", help="Print the model registry as JSON and exit")
    ap.add_argument("--model", choices=sorted(mod.MODELS), help="Model to fine-tune")
    ap.add_argument("--workflow", help="softmax (fixed classes) | standard (promptable)")
    ap.add_argument("--dataset-path", help="UC Volume dir with NIfTI image/label pairs")
    ap.add_argument("--label-mapping", help='JSON {"<label>": [<class id>, "<name>"], ...}; default: ImageCHD 7 cardiac classes')
    ap.add_argument("--experiment-name", help="MLflow experiment workspace path")
    ap.add_argument("--gpu-type", choices=sorted(GPUS_PER_TYPE), help="AI Runtime accelerator (default: per model)")
    ap.add_argument("--job-label", default="Segment-Finetune", help="AI Runtime task key (train.yaml experiment_name)")
    ap.add_argument("--check-dataset", action="store_true", help="Also check the dataset on disk")
    ap.add_argument("--force", action="store_true", help="Overwrite existing project files")
    args = ap.parse_args()

    if args.list_models:
        print(json.dumps(list_models(mod), indent=2))
        return
    if not args.project_dir or not args.model:
        ap.error("PROJECT_DIR and --model are required (or use --list-models)")
    for flag in ("dataset_path",):
        val = getattr(args, flag)
        if val and not val.startswith("/"):
            ap.error(f"--{flag.replace('_', '-')} must be an absolute path (e.g. /Volumes/...)")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.job_label):
        ap.error("--job-label: alphanumerics, '-' and '_' only, max 100 characters")

    result = scaffold(args, mod)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
