"""CPU tests for before/after evaluation, CPU visualisation, the pyfunc model and UC registration helpers.

A stub `monai.bundle` package stands in for MONAI so the real launcher / pyfunc
plumbing (override files, subprocess launch, prediction discovery, Dice, MLflow
logging) runs end to end without a GPU.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

SKILL_DIR = Path(__file__).resolve().parents[1]
SEG = Path(__file__).resolve().parent / "fixtures" / "imagechd_micro"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launcher = _load("train_segment_finetune", SKILL_DIR / "assets" / "train_segment_finetune.py")
viz = _load("visualize", SKILL_DIR / "scripts" / "visualize.py")
reg = _load("register_model", SKILL_DIR / "scripts" / "register_model.py")

# Stub `python -m monai.bundle run`: reads the last override JSON in --config_file and writes
# a prediction NIfTI copied from the matching staged label (so "after" is perfect) or an
# all-background mask (so "before" scores 0), mirroring upstream output naming.
STUB_MAIN = r'''
import ast, json, sys, shutil
from pathlib import Path
import nibabel as nib, numpy as np
args = sys.argv[1:]
configs = ast.literal_eval(args[args.index("--config_file") + 1])
ov = json.loads(Path(configs[-1]).read_text())
main_cfg = Path(configs[0]).name
image = ov.get("input_file") or (ov.get("input_dict") or {}).get("image")
case = Path(image).name.replace(".nii.gz", "")
out = Path(ov["output_dir"]) / case
out.mkdir(parents=True, exist_ok=True)
label = Path(image).parent.parent / "labelsTr" / f"{case}.nii.gz"
img = nib.load(str(label))
data = np.asanyarray(img.dataobj).astype(np.uint8)
if main_cfg == "inference.json":      # pretrained: class IDs, but here predicts nothing
    data = np.zeros_like(data)
nib.save(nib.Nifti1Image(data, img.affine), str(out / f"{case}_{ov.get('output_postfix', 'seg')}.nii.gz"))
'''


@pytest.fixture()
def fake_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    for d in ("configs", "scripts", "models", "monai/bundle"):
        (bundle / d).mkdir(parents=True)
    for f in ("inference.json", "inference_softmax.json", "metadata.json", "logging.conf"):
        (bundle / "configs" / f).write_text("{}")
    (bundle / "monai" / "__init__.py").write_text("")
    (bundle / "monai" / "bundle" / "__init__.py").write_text("")
    (bundle / "monai" / "bundle" / "__main__.py").write_text(STUB_MAIN)
    (bundle / "models" / "model.pt").write_bytes(b"pretrained")
    return bundle


def _staged(tmp_path):
    cfg = launcher.resolve_config({"eval_cases": 2}, {"src_dir": str(SEG), "msd_dir": str(tmp_path / "msd")})
    launcher._reset_staging_dir(cfg["msd_dir"])
    staged = launcher.prepare_msd_dir(launcher.collect_pairs(cfg), cfg["msd_dir"], set(range(1, 8)))
    datalist = tmp_path / "msd" / "datalist.json"
    launcher.build_datalist(staged, str(datalist), 0.5, 42)  # 2 validation cases
    return cfg, datalist


def test_dice_and_label_mapping_helpers():
    gt = np.array([0, 1, 1, 2, 2, 0])
    pred_ids = np.array([0, 151, 151, 0, 152, 6])  # VISTA3D class IDs
    mapping = [[1, 151], [2, 152], [3, 149]]
    pred = launcher.to_dataset_labels(pred_ids, mapping, values_are_class_ids=True)
    assert pred.tolist() == [0, 1, 1, 0, 2, 0]  # unmapped class 6 → background
    d = launcher.dice_per_class(gt, pred, [1, 2, 3])
    assert d["1"] == 1.0 and abs(d["2"] - 2 / 3) < 1e-9 and d["3"] is None


def test_evaluate_before_after_end_to_end(tmp_path, fake_bundle):
    cfg, datalist = _staged(tmp_path)
    mapping, info = launcher.parse_label_mapping(launcher.DEFAULT_LABEL_TO_CTMR)
    out = tmp_path / "out"
    out.mkdir()
    ckpt = out / "model_softmax.pt"
    ckpt.write_bytes(b"ft")
    env = {**os.environ, "PYTHONPATH": str(fake_bundle)}
    metrics, warnings = launcher.evaluate_before_after(cfg, launcher.MODELS["nv-segment-ctmr"], fake_bundle,
                                                       datalist, mapping, info, ckpt, out, env)
    assert not warnings, warnings
    assert metrics["eval_after_mean_dice"] == pytest.approx(1.0)
    assert metrics["eval_before_mean_dice"] == pytest.approx(0.0)
    rep = json.loads((out / "evaluation" / "dice.json").read_text())
    assert len(rep["cases"]) == 2 and rep["classes"]["1"] == "left ventricle"
    case = sorted(rep["cases"])[0]
    slices = np.load(out / "evaluation" / f"slices_{case}.npz")
    assert {"axial_image", "axial_gt", "axial_before", "axial_after", "coronal_after"} <= set(slices.files)

    # CPU visualisation on the produced evaluation/ folder
    import matplotlib
    matplotlib.use("Agg")
    table = viz.dice_table(out / "evaluation")
    assert list(table.columns) == ["before", "after", "change"] and table["after"].min() == 1.0
    assert viz.plot_before_after(out / "evaluation").axes and viz.plot_dice(out / "evaluation").axes


def test_show_case_renders():
    import matplotlib
    matplotlib.use("Agg")
    fig = viz.show_case(str(SEG / "ct_1002_image.nii.gz"), str(SEG / "ct_1002_label.nii.gz"),
                        {k: v[1] for k, v in launcher.DEFAULT_LABEL_TO_CTMR.items()})
    assert len(fig.axes) == 2


def test_pyfunc_logs_reloads_and_predicts(tmp_path, fake_bundle, monkeypatch):
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # keep artifacts in tmp_path (the default is ./mlruns under the current directory)
    exp = mlflow.create_experiment("pyfunc-test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=exp)
    mapping, info = launcher.parse_label_mapping(launcher.DEFAULT_LABEL_TO_CTMR)
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(json.dumps({"label_mappings": {"default": mapping},
                                        "classes": {str(s): {"class_id": c, "name": n} for s, (c, n) in info.items()}}))
    ckpt = tmp_path / "model_softmax.pt"
    ckpt.write_bytes(b"ft")
    # the launcher looks for segment_pyfunc.py next to itself
    monkeypatch.setattr(launcher, "__file__", str(SKILL_DIR / "assets" / "train_segment_finetune.py"))
    cfg = launcher.resolve_config({}, {"src_dir": "/x"})
    with mlflow.start_run():
        uri = launcher.log_pyfunc_model(mlflow, cfg, fake_bundle, ckpt, mapping_path, tmp_path)
    assert uri
    model = mlflow.pyfunc.load_model(uri)
    assert [c.name for c in model.metadata.get_input_schema().inputs] == ["image_path"]

    # stage one case like the launcher does, so the stub can find its label
    msd = tmp_path / "msd"
    (msd / "imagesTr").mkdir(parents=True)
    (msd / "labelsTr").mkdir()
    os.symlink(SEG / "ct_1002_image.nii.gz", msd / "imagesTr" / "case_0000.nii.gz")
    os.symlink(SEG / "ct_1002_label.nii.gz", msd / "labelsTr" / "case_0000.nii.gz")
    # the logged bundle holds configs/ + scripts/ only; expose the stub monai via PYTHONPATH
    monkeypatch.setenv("PYTHONPATH", str(fake_bundle))
    res = model.predict(pd.DataFrame({"image_path": [str(msd / "imagesTr" / "case_0000.nii.gz")]}),
                        params={"output_dir": str(tmp_path / "pred")})
    voxels = json.loads(res.loc[0, "class_voxels"])
    assert Path(res.loc[0, "label_path"]).exists() and voxels["left ventricle"] > 0


@pytest.mark.parametrize("user,expected", [("Emanuele.Rinaldi@databricks.com", "ws_emanuele_rinaldi"),
                                           ("a-b+c@x.org", "ws_a_b_c"), ("plain", "ws_plain")])
def test_default_schema(user, expected):
    assert reg.default_schema(user) == expected


def test_register_dry_run(monkeypatch):
    class Client:
        def get_run(self, run_id):
            data = type("D", (), {"tags": {"status": "SUCCESS", "segment.model_id": "nv-segment-ctmr",
                                           "segment.model_uri": "models:/m-123"},
                                  "params": {"segment.workflow": "softmax"}, "metrics": {}})
            return type("R", (), {"data": data})

    import mlflow
    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: Client())
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    args = type("A", (), {"run_id": "abc", "catalog": "workshop_cat", "schema": "ws_jane_doe",
                          "model_name": None, "alias": "workshop", "dry_run": True})()
    out = reg.register(args)
    assert out["registered_model"] == "workshop_cat.ws_jane_doe.nv_segment_ctmr_finetuned"
    assert out["model_uri"] == "models:/m-123" and out["dry_run"] is True
    args.catalog = "Bad-Catalog"
    with pytest.raises(SystemExit):
        reg.register(args)
    assert reg.register_run("abc", "workshop_cat", schema="ws_jane_doe", dry_run=True)["dry_run"] is True
