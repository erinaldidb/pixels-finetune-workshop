"""CPU tests for the nv-reason-finetune scaffold, validator and launcher.

Run from the skill directory:
    python -m pytest tests -q
Needs PyYAML, pytest. No torch, transformers or GPU.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CXR = FIXTURES / "reason_cxr_micro"
RCT = FIXTURES / "reason_ct_micro"
SCAFFOLD = SKILL_DIR / "scripts" / "scaffold_project.py"
VALIDATE = SKILL_DIR / "scripts" / "validate_project.py"
LAUNCHER = SKILL_DIR / "assets" / "train_reason_finetune.py"
SCRIPT_NAME = LAUNCHER.name
EXPERIMENT = "/Users/fixture@example.com/Reason"

spec = importlib.util.spec_from_file_location("train_reason_finetune", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

MODEL_ARGS = {
    "nv-reason-cxr": ["--manifest", CXR / "sft.jsonl", "--image-dir", CXR / "images"],
    "nv-reason-ct": ["--manifest", RCT / "sft.jsonl", "--image-dir", RCT],
}


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout) if proc.stdout.strip().startswith(("{", "[")) else None, proc.stderr


def _scaffold(project, model, *extra):
    return _run(SCAFFOLD, project, "--model", model, *MODEL_ARGS[model], "--experiment-name", EXPERIMENT, *extra)


def test_registry_entries_are_complete():
    common = {"family", "title", "repo_url", "repo_commit", "hf_repo_id", "hf_revision", "workflows",
              "default_workflow", "default_gpu", "environment_version", "dependencies", "license"}
    for model_id, m in launcher.MODELS.items():
        assert common <= set(m), model_id
        assert m["default_workflow"] in m["workflows"], model_id
        assert len(m["repo_commit"]) == 40 and len(m["hf_revision"]) == 40, model_id
        assert not any(d.split("=")[0].split("<")[0].split(">")[0] == "torch" for d in m["dependencies"]), model_id
        assert {"data_args", "deepspeed", "manifest_keys"} <= set(m)


def test_list_models():
    rc, out, _ = _run(SCAFFOLD, "--list-models")
    assert rc == 0 and {m["id"] for m in out} == set(launcher.MODELS)


@pytest.mark.parametrize("model", sorted(MODEL_ARGS))
def test_scaffold_creates_valid_project(tmp_path, model):
    rc, out, err = _scaffold(tmp_path / "proj", model, "--check-dataset")
    assert rc == 0, err
    assert out["ok"] and out["model_id"] == model and out["validation"]["n_failed"] == 0
    assert {f["action"] for f in out["files"].values()} == {"created"}
    assert (tmp_path / "proj" / SCRIPT_NAME).read_bytes() == LAUNCHER.read_bytes()
    assert "databricks air run --file train.yaml" in out["next_step"]


def test_multi_gpu_reasoning_installs_deepspeed(tmp_path):
    _scaffold(tmp_path / "p", "nv-reason-ct", "--gpu-type", "GPU_8xH100")
    wl = (tmp_path / "p" / "train.yaml").read_text()
    assert "num_accelerators: 8" in wl and "--no-build-isolation 'deepspeed==0.18.9'" in wl
    assert "transformers==5.6.2" in wl and "trl==1.2.0" in wl


def test_grpo_rejects_sft_manifest(tmp_path):
    rc, out, _ = _run(SCAFFOLD, tmp_path / "p", "--model", "nv-reason-cxr", "--workflow", "grpo",
                      "--manifest", CXR / "sft.jsonl", "--image-dir", CXR / "images",
                      "--experiment-name", EXPERIMENT, "--check-dataset")
    assert rc == 1
    assert any("grpo schema" in c["name"] and not c["ok"] for c in out["validation"]["checks"])


def test_missing_images_are_reported(tmp_path):
    rc, out, _ = _run(SCAFFOLD, tmp_path / "p", "--model", "nv-reason-cxr", "--manifest", CXR / "sft.jsonl",
                      "--image-dir", tmp_path, "--experiment-name", EXPERIMENT, "--check-dataset")
    assert rc == 1
    assert any(c["name"] == "referenced images exist" and not c["ok"] for c in out["validation"]["checks"])


def test_scaffold_requires_inputs_when_config_missing(tmp_path):
    proc = subprocess.run([sys.executable, str(SCAFFOLD), str(tmp_path / "p"), "--model", "nv-reason-ct"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "--manifest" in proc.stderr


def test_three_tier_merge_and_model_defaults():
    yaml_cfg = {"model_id": "nv-reason-ct", "manifest": "/m.jsonl", "image_dir": "/i", "lr": 1e-5}
    cfg = launcher.resolve_config({"max_steps": 8}, yaml_cfg)
    assert cfg["max_steps"] == 8 and cfg["lr"] == 1e-5 and cfg["workflow"] == "sft"
    assert cfg["epochs"] is None  # hf_vlm: keep the upstream YAML value
    with pytest.raises(ValueError):
        launcher.resolve_config({"workflow": "softmax"}, yaml_cfg)
    with pytest.raises(ValueError):
        launcher.resolve_config({"model_id": "nv-unknown"}, {})


def test_reset_staging_refuses_foreign_dir(tmp_path):
    (tmp_path / "keep.txt").write_text("user data")
    with pytest.raises(RuntimeError):
        launcher._reset_staging_dir(str(tmp_path))


def test_vlm_command_matches_upstream_cli(tmp_path):
    for model, data_flag in (("nv-reason-cxr", "--dataset_root"), ("nv-reason-ct", "--dataset_path")):
        cfg = launcher.resolve_config({"max_steps": 8}, {"model_id": model, "manifest": "/m.jsonl", "image_dir": "/i"})
        cfg["num_gpus"] = 8
        cmd = launcher.build_vlm_command(cfg, launcher.MODELS[model], tmp_path, "/w", tmp_path / "manifest.jsonl", tmp_path / "out")
        assert cmd[:2] == ["accelerate", "launch"] and str(tmp_path / "accelerate/zero2.yaml") in cmd
        assert data_flag in cmd and cmd[cmd.index("--max_steps") + 1] == "8"
        assert cmd[cmd.index("--report_to") + 1] == "mlflow"
    cfg["num_gpus"], cfg["skip_mlflow"] = 1, True
    cmd = launcher.build_vlm_command(cfg, launcher.MODELS["nv-reason-ct"], tmp_path, "/w", tmp_path / "m.jsonl", tmp_path / "o")
    assert str(tmp_path / "accelerate/ddp.yaml") in cmd and cmd[cmd.index("--report_to") + 1] == "none"


def test_cxr_manifest_normalised_to_upstream_features(tmp_path):
    cfg = launcher.resolve_config({"max_cases": 2}, {"model_id": "nv-reason-cxr", "manifest": str(CXR / "sft.jsonl"),
                                                     "image_dir": str(CXR / "images")})
    dst = tmp_path / "manifest.jsonl"
    assert launcher.prepare_manifest(cfg, launcher.MODELS["nv-reason-cxr"], dst) == 2
    row = json.loads(dst.read_text().splitlines()[0])
    assert row["subject_id"] == "1000" and isinstance(row["study_id"], str)  # datasets Value('string')
