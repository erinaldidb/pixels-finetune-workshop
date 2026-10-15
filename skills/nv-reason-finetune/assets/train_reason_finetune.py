#!/usr/bin/env python3
"""NV-Reason fine-tuning launcher for Databricks AI Runtime (TRL vision-language models).

Fine-tunes the NVIDIA-Medtech reasoning VLMs listed in MODELS (NV-Reason-CXR,
NV-Reason-CT) through their upstream, documented entry point:
`accelerate launch train/vlm_{sft,grpo}_train.py --config configs/...`.
This script only validates the JSONL manifest, pins upstream code and
weights, launches training, and logs metrics and the final model to MLflow.

Usage (AI Runtime, from train.yaml):
    cd $CODE_SOURCE_PATH
    python train_reason_finetune.py --config finetune_config.yml

Smoke test / overrides (CLI > YAML > upstream config):
    python train_reason_finetune.py --config finetune_config.yml --max-cases 16 --max-steps 8 --skip-mlflow

CPU-only check (resolve config, no downloads, no GPU):
    python train_reason_finetune.py --config finetune_config.yml --print-config
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

STAGING_MARKER = ".reason_finetune_staging"

# ---------------------------------------------------------------------------
# Model registry: the single source of truth for upstream pins, workflows and
# GPU-node dependencies. scaffold_project.py and validate_project.py import it.
# To add a TRL/accelerate model, add an entry here (see references/adding-a-model.md).
# ---------------------------------------------------------------------------
MODELS = {
    "nv-reason-cxr": {
        "family": "hf_vlm",
        "title": "NV-Reason-CXR-3B (chest X-ray reasoning VLM, Qwen2.5-VL)",
        "repo_url": "https://github.com/NVIDIA-Medtech/NV-Reason-CXR.git",
        "repo_commit": "83a4d51c9fbbff68156a5f01796f04e26519b6ad",
        "hf_repo_id": "nvidia/NV-Reason-CXR-3B",
        "hf_revision": "056bd0383b35226554da9dc5866e095df174ae19",
        "workflows": {
            "sft": {"script": "train/vlm_sft_train.py", "config": "configs/sft_config.yaml",
                    "fixed_args": {"dataset_streaming": "false"}},
            "grpo": {"script": "train/vlm_grpo_train.py", "config": "configs/grpo_config.yaml",
                     # vLLM 0.10.1.1 pins torch 2.7.1; fall back to transformers generation on AI Runtime
                     "fixed_args": {"dataset_streaming": "false", "use_vllm": "false"}},
        },
        "data_args": "root_and_name",  # --dataset_root DIR --dataset_path FILE
        "eval_runner": "transformers_chat",  # README usage snippet (no inference CLI upstream)
        "manifest_keys": {"id": "", "image": "", "subject_id": "", "study_id": "", "solution": ""},
        "default_workflow": "sft",
        "default_gpu": "GPU_8xH100",
        "environment_version": "5",
        "dependencies": [
            "transformers==4.56.1", "trl==0.22.2", "accelerate", "datasets", "peft", "qwen-vl-utils",
            "liger-kernel", "safetensors", "huggingface_hub", "Pillow", "PyYAML", "mlflow>=2.10,<4",
        ],
        "deepspeed": "deepspeed",
        "license": "NVIDIA OneWay Noncommercial License (non-commercial use only)",
    },
    "nv-reason-ct": {
        "family": "hf_vlm",
        "title": "NV-Reason-CT (3D CT reasoning VLM, Qwen3.5 + 3D encoder)",
        "repo_url": "https://github.com/NVIDIA-Medtech/NV-Reason-CT.git",
        "repo_commit": "b4abd354fef90819ec40dd637c3bd14521b8023c",
        "hf_repo_id": "nvidia/NV-Reason-CT",
        "hf_revision": "386b93e034983f6c1fc841a43833a1b6a0cd9c13",
        "workflows": {
            "sft": {"script": "train/vlm_sft_train.py", "config": "configs/sft_config.yaml", "fixed_args": {}},
            "grpo": {"script": "train/vlm_grpo_train.py", "config": "configs/grpo_config.yaml", "fixed_args": {}},
        },
        "data_args": "path",  # --dataset_path FILE
        "eval_runner": "inference_cli",  # upstream inference.py IMAGE --model DIR --region R --prompt P
        "manifest_keys": {},
        "default_workflow": "sft",
        "default_gpu": "GPU_8xH100",
        "environment_version": "5",
        # requirements-train.txt pins: the custom GRPO trainer follows TRL 1.2.0 internals
        "dependencies": [
            "transformers==5.6.2", "trl==1.2.0", "accelerate==1.13.0", "datasets==4.8.4",
            "liger-kernel==0.7.0", "monai>=1.5.1", "nibabel>=5.3.3", "dynamic-network-architectures==0.4.3",
            "scipy", "timm", "einops", "safetensors", "Pillow", "PyYAML", "mlflow>=2.10,<4",
        ],
        "deepspeed": "deepspeed==0.18.9",
        "license": "OpenMDW-1.1",
    },
}

# Hardcoded defaults (lowest priority). None = "not set": use the model/workflow
# value from the upstream configs/<workflow>_config.yaml.
DEFAULTS = {
    "model_id": "nv-reason-cxr",
    "workflow": None,
    "repo_commit": None,
    "hf_revision": None,
    # data
    "manifest": None,
    "image_dir": None,
    "max_cases": None,
    # training
    "epochs": None,
    "lr": None,
    "max_steps": None,
    "batch_size": None,
    "grad_accum": None,
    "extra_args": {},
    "num_gpus": "auto",
    # paths + tracking
    "output_dir": "/tmp/finetune_output",
    "mlflow_experiment": None,
    "mlflow_run_name": None,
    "skip_mlflow": False,
    "log_model": True,
    # rows held out of training and answered by the base and fine-tuned model; 0 disables it
    "eval_cases": 4,
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
        ("dataset", "max_cases"): "max_cases",
        ("dataset", "manifest"): "manifest",
        ("dataset", "image_dir"): "image_dir",
        ("training", "epochs"): "epochs",
        ("training", "learning_rate"): "lr",
        ("training", "max_steps"): "max_steps",
        ("training", "per_device_train_batch_size"): "batch_size",
        ("training", "gradient_accumulation_steps"): "grad_accum",
        ("training", "extra_args"): "extra_args",
        ("training", "num_gpus"): "num_gpus",
        ("paths", "output_dir"): "output_dir",
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

    if isinstance(final["extra_args"], str):
        final["extra_args"] = json.loads(final["extra_args"])
    return final


def check_required(cfg: dict) -> list:
    """Return human-readable errors for missing family-specific inputs."""
    errors = []
    if not cfg["manifest"]:
        errors.append("dataset.manifest (JSONL) is required")
    if not cfg["image_dir"]:
        errors.append("dataset.image_dir is required")
    return errors


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
    for pkg in ("torch", "transformers", "trl", "accelerate", "deepspeed", "monai", "flash-attn", "mlflow"):
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
    # "reason." prefix avoids clashing with params the HF MLflowCallback logs itself
    mlflow.log_params({
        "reason.model_id": cfg["model_id"],
        "reason.workflow": cfg["workflow"],
        "reason.repo_commit": cfg["repo_commit"],
        "reason.hf_revision": cfg["hf_revision"],
        "reason.num_gpus": cfg["num_gpus"],
    })
    mlflow.set_tags({"reason.model_id": cfg["model_id"], "reason.family": cfg["family"]})


# ── TRL fine-tuning ────────────────────────────────────────────────────────
def _image_refs(row: dict) -> list:
    refs = []
    for msg in row.get("messages", []):
        for part in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if part.get("type") == "image" and part.get("image"):
                refs.append(part["image"])
    return refs


def check_manifest_row(row: dict, workflow: str) -> list:
    """Return schema problems for one JSONL row (empty list = valid)."""
    problems = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["missing 'messages' list"]
    if not _image_refs(row):
        problems.append("no {'type': 'image'} content in messages")
    if workflow == "sft" and msgs[-1].get("role") != "assistant":
        problems.append("SFT rows must end with an assistant message")
    if workflow == "grpo":
        if msgs[-1].get("role") != "user":
            problems.append("GRPO rows must contain the user turn only")
        if not row.get("solution"):
            problems.append("GRPO rows need a non-empty 'solution'")
    return problems


def prepare_manifest(cfg: dict, spec: dict, dst: Path, check_images: bool = True,
                     holdout: int = 0, holdout_dst: Path | None = None) -> int:
    """Validate the JSONL manifest, fill schema keys, apply max_cases; returns training row count.

    With holdout > 0 the last `holdout` rows are written to holdout_dst instead of
    the training manifest, so the before/after evaluation uses unseen cases.
    """
    rows, errors, missing = [], [], 0
    with open(cfg["manifest"]) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for p in check_manifest_row(row, cfg["workflow"]):
                errors.append(f"line {n}: {p}")
            for key, fill in spec.get("manifest_keys", {}).items():  # fixed datasets.Features schema
                row[key] = fill if row.get(key) is None else str(row[key])
            if check_images:
                missing += sum(not os.path.exists(os.path.join(cfg["image_dir"], r)) for r in _image_refs(row))
            rows.append(row)
            if cfg["max_cases"] and len(rows) >= cfg["max_cases"]:
                break
    if errors:
        raise ValueError(f"{len(errors)} manifest problems, first: {errors[:5]}")
    if missing:
        raise FileNotFoundError(f"{missing} referenced images not found under image_dir {cfg['image_dir']}")
    if holdout and len(rows) > 2 * holdout:  # keep at least as many training rows as held-out ones
        rows, held = rows[:-holdout], rows[-holdout:]
        holdout_dst.write_text("".join(json.dumps(r) + "\n" for r in held))
        print(f"Held out {len(held)} rows for before/after evaluation → {holdout_dst}")
    elif holdout:
        print(f"WARNING: {len(rows)} rows are too few to hold out {holdout}; skipping evaluation")
    dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Manifest ready: {len(rows)} rows → {dst}")
    return len(rows)


# ── Before / after evaluation ──────────────────────────────────────────────
EVAL_MAX_NEW_TOKENS = 1024  # bounds evaluation time; upstream reports fit well within this
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def answer_set(text: str) -> set:
    """Findings in the last <answer>...</answer> block, as a lowercase set."""
    found = ANSWER_RE.findall(text or "")
    return {x.strip().lower() for x in (found[-1] if found else "").split(",") if x.strip()}


def findings_f1(prediction: str, solution: str) -> float:
    pred = answer_set(prediction)
    ref = {x.strip().lower() for x in (solution or "").split(",") if x.strip()}
    if not pred and not ref:
        return 1.0
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def user_turn(row: dict) -> tuple:
    """(prompt text, image path) of the first user message."""
    user = next(m for m in row["messages"] if m.get("role") == "user")
    text = " ".join(p.get("text", "") for p in user["content"] if p.get("type") == "text").strip()
    return text, _image_refs(row)[0]


def generate_inference_cli(checkout: Path, model_dir: str, rows: list, image_dir: str, env: dict, log: Path) -> list:
    """NV-Reason-CT: the upstream inference.py, one subprocess per case."""
    texts = []
    with open(log, "a") as fh:
        for row in rows:
            prompt, image = user_turn(row)
            cmd = [sys.executable, str(checkout / "inference.py"), os.path.join(image_dir, image),
                   "--model", model_dir, "--region", row.get("anatomy_region") or "chest",
                   "--prompt", prompt, "--max-new-tokens", str(EVAL_MAX_NEW_TOKENS)]
            proc = subprocess.run(cmd, cwd=str(checkout), env=env, capture_output=True, text=True)
            fh.write(proc.stderr[-4000:])
            texts.append(proc.stdout.strip() if proc.returncode == 0 else f"[generation failed: rc={proc.returncode}]")
    return texts


def generate_transformers_chat(model_dir: str, rows: list, image_dir: str) -> list:
    """NV-Reason-CXR: the generation snippet from the upstream README."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(model_dir)
    texts = []
    for row in rows:
        prompt, image_path = user_turn(row)
        image = Image.open(os.path.join(image_dir, image_path))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False)
        texts.append(processor.batch_decode([generated[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0])
    del model
    torch.cuda.empty_cache()
    return texts


def evaluate_before_after(cfg, spec, checkout: Path, pretrained: str, finetuned: str, holdout: Path,
                          out: Path, env: dict) -> tuple:
    """Answer held-out rows with the base and fine-tuned model; writes out/evaluation/generations.jsonl."""
    rows = [json.loads(line) for line in holdout.read_text().splitlines() if line.strip()]
    eval_dir = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    answers = {}
    for phase, model_dir in (("before", pretrained), ("after", finetuned)):
        print(f"  Generating {len(rows)} {phase} answers with {model_dir}")
        if spec["eval_runner"] == "inference_cli":
            answers[phase] = generate_inference_cli(checkout, model_dir, rows, cfg["image_dir"], env,
                                                    eval_dir / "generation.log")
        else:
            answers[phase] = generate_transformers_chat(model_dir, rows, cfg["image_dir"])
    records = []
    for i, row in enumerate(rows):
        prompt, image = user_turn(row)
        reference = row["messages"][-1] if row["messages"][-1].get("role") == "assistant" else None
        records.append({
            "id": row.get("id"), "image": image, "prompt": prompt, "solution": row.get("solution"),
            "reference": " ".join(p.get("text", "") for p in reference["content"]) if reference else None,
            "before": answers["before"][i], "after": answers["after"][i],
            "before_f1": findings_f1(answers["before"][i], row.get("solution")),
            "after_f1": findings_f1(answers["after"][i], row.get("solution")),
        })
    (eval_dir / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    metrics = {f"eval_{phase}_findings_f1": sum(r[f"{phase}_f1"] for r in records) / len(records)
               for phase in ("before", "after")}
    warnings = [f"{r['id']}: {phase} generation failed" for r in records for phase in ("before", "after")
                if r[phase].startswith("[generation failed")]
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics, warnings


def build_vlm_command(cfg: dict, spec: dict, checkout: Path, model_path: str, manifest: Path, out_dir: Path) -> list:
    wf = spec["workflows"][cfg["workflow"]]
    n = cfg["num_gpus"]
    accel_cfg = "accelerate/ddp.yaml" if n == 1 else "accelerate/zero2.yaml"  # zero2 needs deepspeed
    cmd = ["accelerate", "launch", "--config_file", str(checkout / accel_cfg),
           "--num_processes", str(n), "--num_machines", "1",
           str(checkout / wf["script"]), "--config", str(checkout / wf["config"]),
           "--model_name_or_path", model_path, "--output_dir", str(out_dir),
           "--image_dir", cfg["image_dir"],
           "--report_to", "none" if cfg["skip_mlflow"] else "mlflow",
           "--run_name", cfg["mlflow_run_name"]]
    if spec["data_args"] == "root_and_name":
        cmd += ["--dataset_root", str(manifest.parent), "--dataset_path", manifest.name]
    else:
        cmd += ["--dataset_path", str(manifest)]
    args = dict(wf.get("fixed_args", {}))
    for key, flag in (("epochs", "num_train_epochs"), ("lr", "learning_rate"), ("max_steps", "max_steps"),
                      ("batch_size", "per_device_train_batch_size"), ("grad_accum", "gradient_accumulation_steps")):
        if cfg[key] is not None:
            args[flag] = cfg[key]
    args.update({k: v for k, v in (cfg["extra_args"] or {}).items()})
    for k, v in args.items():
        cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    return cmd


def run_hf_vlm(cfg: dict, spec: dict) -> bool:
    out = Path(cfg["output_dir"])
    print("\n── Step 1: Validate manifest ──")
    manifest, holdout = out / "manifest.jsonl", out / "holdout.jsonl"
    n_rows = prepare_manifest(cfg, spec, manifest, holdout=cfg["eval_cases"], holdout_dst=holdout)

    print("\n── Step 2: Stage upstream code + weights ──")
    checkout = clone_repo(spec["repo_url"], cfg["repo_commit"], f"/tmp/{cfg['model_id']}-{cfg['repo_commit'][:7]}")
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(spec["hf_repo_id"], revision=cfg["hf_revision"],
                                   local_dir=str(out / "pretrained"))
    model_out = out / "model"
    cmd = build_vlm_command(cfg, spec, checkout, model_path, manifest, model_out)

    print("\n── Step 3: Train ──")
    ctx, mlflow = mlflow_context(cfg)
    with ctx as run:
        env = dict(os.environ)
        if mlflow:
            log_common_params(mlflow, cfg)
            mlflow.log_params({"reason.n_rows": n_rows})
            # Let the transformers MLflowCallback (rank 0) log into this same run
            env.update({"MLFLOW_TRACKING_URI": "databricks", "MLFLOW_RUN_ID": run.info.run_id,
                        "MLFLOW_EXPERIMENT_ID": run.info.experiment_id})
        t0 = time.time()
        rc = run_logged(cmd, checkout, env, out / "train.log")
        elapsed = time.time() - t0
        print(f"\nTraining exited with code {rc} after {elapsed:.0f}s")

        metrics, warnings = {}, []
        state_path = model_out / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            losses = [h["loss"] for h in state.get("log_history", []) if "loss" in h]
            metrics = {"global_step": state.get("global_step"), "final_train_loss": losses[-1] if losses else None}
            if losses and any(l != l for l in losses):  # NaN check
                warnings.append("non-finite training loss logged")
        ok = rc == 0 and (model_out / "config.json").exists()
        if ok and holdout.exists():
            print("\n── Step 4: Before / after evaluation ──")
            try:
                eval_metrics, eval_warnings = evaluate_before_after(
                    cfg, spec, checkout, model_path, str(model_out), holdout, out, env)
                metrics.update(eval_metrics)
                warnings += eval_warnings
            except Exception as exc:  # evaluation must never fail a finished training run
                warnings.append(f"before/after evaluation failed: {exc}")
        summary = write_run_summary(out / "run_summary.json", cfg, ok=ok, elapsed_s=elapsed, n_rows=n_rows,
                                    command=cmd, model_dir=str(model_out), warnings=warnings, **metrics)
        if mlflow:
            mlflow.log_metric("training_time_s", elapsed)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            for p in (summary, manifest, out / "train.log"):
                mlflow.log_artifact(str(p))
            if (out / "evaluation").is_dir():
                mlflow.log_artifacts(str(out / "evaluation"), "evaluation")
            if ok and cfg["log_model"]:
                # final model + processor only; checkpoint-* dirs hold optimizer state
                for f in sorted(model_out.iterdir()):
                    if f.is_file():
                        mlflow.log_artifact(str(f), "model")
            mlflow.set_tag("status", "SUCCESS" if ok else "FAILED")
            if not ok:
                mlflow.end_run(status="FAILED")
    return ok


# ── CLI ────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune NV-Reason VLMs on AI Runtime",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag defaults to None so YAML values can fill it (never required=True)
    p.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    p.add_argument("--model-id", default=None, choices=sorted(MODELS), help="Model to fine-tune")
    p.add_argument("--workflow", default=None, choices=["grpo", "sft"], help="TRL workflow")
    p.add_argument("--manifest", default=None, help="JSONL manifest")
    p.add_argument("--image-dir", default=None, help="Root for image paths in the manifest")
    p.add_argument("--max-cases", type=int, default=None, help="Limit manifest rows")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--max-steps", type=int, default=None, help="Optimizer steps (overrides epochs)")
    p.add_argument("--num-gpus", default=None, help="'auto' or an integer")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--mlflow-experiment", default=None, help="MLflow experiment path")
    p.add_argument("--mlflow-run-name", default=None, help="MLflow run name")
    p.add_argument("--skip-mlflow", action="store_true", default=None, help="Disable MLflow tracking")
    p.add_argument("--eval-cases", type=int, default=None, help="Rows held out for before/after evaluation (0 = off)")
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
    ok = run_hf_vlm(cfg, spec)
    print("\n" + "=" * 60)
    print(("TRAINING COMPLETE" if ok else "TRAINING FAILED — check logs above") +
          f"  ({cfg['model_id']} / {cfg['workflow']})")
    print(f"  Summary: {Path(cfg['output_dir']) / 'run_summary.json'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
