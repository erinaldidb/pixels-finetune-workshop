"""CPU tests for the nv-segment-finetune scaffold, validator and launcher.

Run from the skill directory:
    python -m pytest tests -q
Needs PyYAML, nibabel, numpy, pytest. No MONAI, torch or GPU.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[1]
SEG = Path(__file__).resolve().parent / "fixtures" / "imagechd_micro"
SCAFFOLD = SKILL_DIR / "scripts" / "scaffold_project.py"
VALIDATE = SKILL_DIR / "scripts" / "validate_project.py"
LAUNCHER = SKILL_DIR / "assets" / "train_segment_finetune.py"
SCRIPT_NAME = LAUNCHER.name
EXPERIMENT = "/Users/fixture@example.com/Segment"

spec = importlib.util.spec_from_file_location("train_segment_finetune", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

MODEL_ARGS = {
    "nv-segment-ctmr": ["--dataset-path", SEG],
    "nv-segment-ct": ["--dataset-path", SEG],
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
        assert {"bundle_subdir", "hf_weights"} <= set(m) and "monai==1.4.0" in m["dependencies"]


def test_list_models():
    rc, out, _ = _run(SCAFFOLD, "--list-models")
    assert rc == 0 and {m["id"] for m in out} == set(launcher.MODELS)


@pytest.mark.parametrize("model", sorted(MODEL_ARGS))
def test_scaffold_creates_valid_project(tmp_path, model):
    rc, out, err = _scaffold(tmp_path / "proj", model, "--check-dataset")
    assert rc == 0, err
    assert out["ok"] and out["model_id"] == model and out["validation"]["n_failed"] == 0
    assert {f["action"] for f in out["files"].values()} == {"created"} and len(out["files"]) == 4
    assert (tmp_path / "proj" / SCRIPT_NAME).read_bytes() == LAUNCHER.read_bytes()
    assert "databricks air run --file train.yaml" in out["next_step"]


def test_segment_ct_softmax_rejects_ctmr_only_ids(tmp_path):
    rc, out, _ = _scaffold(tmp_path / "p", "nv-segment-ct", "--workflow", "softmax")
    assert rc == 1
    assert any("class IDs valid" in c["name"] and not c["ok"] for c in out["validation"]["checks"])


def test_scaffold_keeps_existing_files(tmp_path):
    proj = tmp_path / "proj"
    _scaffold(proj, "nv-segment-ctmr")
    cfg = proj / "finetune_config.yml"
    cfg.write_text(cfg.read_text().replace("epochs: 50", "epochs: 7"))
    rc, out, _ = _run(SCAFFOLD, proj, "--model", "nv-segment-ctmr")
    assert rc == 0
    assert out["files"]["finetune_config.yml"]["action"] == "kept"
    assert out["files"][SCRIPT_NAME]["matches_reference"] is True
    assert "epochs: 7" in cfg.read_text()


def test_scaffold_requires_inputs_when_config_missing(tmp_path):
    proc = subprocess.run([sys.executable, str(SCAFFOLD), str(tmp_path / "p"), "--model", "nv-segment-ct"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "--dataset-path" in proc.stderr


def test_validator_flags_unquoted_version_and_torch(tmp_path):
    proj = tmp_path / "proj"
    _scaffold(proj, "nv-segment-ctmr")
    wl = proj / "train.yaml"
    wl.write_text(wl.read_text().replace("version: '5'", "version: 5").replace("- monai==1.4.0", "- torch\n    - monai==1.4.0"))
    rc, out, _ = _run(VALIDATE, proj)
    failed = {c["name"] for c in out["checks"] if not c["ok"]}
    assert rc == 1
    assert {"environment.version is a quoted string", "torch is not listed in dependencies"} <= failed


def test_legacy_workshop_config_resolves_to_ctmr_softmax():
    # The original workshop finetune_config.yml has no model: section, only bundle.commit
    legacy = {"src_dir": "/x", "bundle_commit": None, "repo_commit": "cb921f5c58837c0f42a713855d68b32af88e1cdd"}
    cfg = launcher.resolve_config({}, legacy)
    assert (cfg["model_id"], cfg["workflow"], cfg["epochs"], cfg["lr"]) == ("nv-segment-ctmr", "softmax", 50, 5e-5)


@pytest.mark.parametrize("raw", [{"1": [151, "lv"], "8": [0, "pad"]}, {"1": [151, "a"], "2": [151, "b"]}, {}])
def test_label_mapping_rejects_invalid(raw):
    with pytest.raises(ValueError):
        launcher.parse_label_mapping(raw)


def test_class_id_rules_per_workflow():
    ct = {int(k): v for k, v in json.loads((SKILL_DIR / "assets/label_ids/nv-segment-ct.json").read_text())["labels"].items()}
    assert launcher.check_mapping_against_bundle([[1, 6], [2, 200]], "standard", ct)  # new ID → warning only
    with pytest.raises(ValueError):
        launcher.check_mapping_against_bundle([[1, 200]], "softmax", ct)
    with pytest.raises(ValueError):
        launcher.check_mapping_against_bundle([[1, 300]], "standard", {300: "x"})  # >= num_classes 255


def test_staging_zeroes_stray_labels(tmp_path):
    import nibabel as nib
    import numpy as np

    cfg = launcher.resolve_config({}, {"src_dir": str(SEG)})
    msd = tmp_path / "msd"
    launcher._reset_staging_dir(str(msd))
    staged = launcher.prepare_msd_dir(launcher.collect_pairs(cfg), str(msd), set(range(1, 8)))
    cleaned = [l for _, l, _ in staged if not l.is_symlink()]
    assert len(cleaned) == 1  # only ct_1001 holds the stray value 9
    assert set(np.unique(np.asanyarray(nib.load(cleaned[0]).dataobj))) <= set(range(8))
    assert launcher.build_datalist(staged, str(msd / "d.json"), 0.15, 42) == (3, 1)


def test_reset_staging_refuses_foreign_dir(tmp_path):
    (tmp_path / "keep.txt").write_text("user data")
    with pytest.raises(RuntimeError):
        launcher._reset_staging_dir(str(tmp_path))


def test_bundle_override_follows_workflow(tmp_path):
    for wf, extra_key in (("softmax", None), ("standard", "lr_schedule#activate")):
        cfg = launcher.resolve_config({"workflow": wf}, {"src_dir": "/x"})
        ov = launcher.build_bundle_override(cfg, launcher.MODELS["nv-segment-ctmr"], tmp_path, "d.json",
                                            [[1, 151]], tmp_path / "ck", tmp_path / "tb")
        assert ov["epochs"] == 50 and ov["dataset_dir"] == cfg["msd_dir"] and ov["use_mlflow"] is False
        assert "train#trainer#max_epochs" not in ov and "data_file_base_dir" not in ov
        assert (extra_key in ov) == (extra_key is not None)
