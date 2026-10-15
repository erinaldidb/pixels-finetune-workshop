"""CPU tests for the NV-Reason before/after evaluation and the show_eval helpers.

A stub upstream inference.py stands in for NV-Reason-CT generation so the real
hold-out, subprocess, scoring and report code runs without a GPU.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RCT = FIXTURES / "reason_ct_micro"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launcher = _load("train_reason_finetune", SKILL_DIR / "assets" / "train_reason_finetune.py")
show_eval = _load("show_eval", SKILL_DIR / "scripts" / "show_eval.py")

# Stub NV-Reason-CT inference.py: the "fine-tuned" model dir answers correctly, the base one does not
STUB_INFERENCE = r'''
import sys
args = sys.argv[1:]
model = args[args.index("--model") + 1]
print("<think>stub</think>\n<answer>" + ("No Chest Finding" if "finetuned" in model else "Atelectasis") + "</answer>")
'''


@pytest.mark.parametrize("pred,solution,expected", [
    ("<answer>Atelectasis, Pleural effusion</answer>", "Atelectasis, Pleural effusion", 1.0),
    ("<think>x</think><answer>atelectasis</answer>", "Atelectasis, Pleural effusion", 2 / 3),
    ("no answer block", "Atelectasis", 0.0),
    ("<answer></answer>", "", 1.0),
])
def test_findings_f1(pred, solution, expected):
    assert launcher.findings_f1(pred, solution) == pytest.approx(expected)


def _ct_rows(tmp_path, n):
    rows = [json.loads(l) for l in (RCT / "sft.jsonl").read_text().splitlines()]
    rows = [json.loads(json.dumps(r)) for r in (rows * ((n // len(rows)) + 1))[:n]]  # independent copies
    for i, r in enumerate(rows):
        r["id"] = f"case-{i}"
    manifest = tmp_path / "in.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def test_holdout_split(tmp_path):
    cfg = launcher.resolve_config({}, {"model_id": "nv-reason-ct", "manifest": str(_ct_rows(tmp_path, 10)),
                                       "image_dir": str(RCT)})
    n = launcher.prepare_manifest(cfg, launcher.MODELS["nv-reason-ct"], tmp_path / "train.jsonl",
                                  holdout=4, holdout_dst=tmp_path / "holdout.jsonl")
    held = (tmp_path / "holdout.jsonl").read_text().splitlines()
    assert n == 6 and len(held) == 4 and json.loads(held[0])["id"] == "case-6"
    # too few rows: no hold-out file, all rows train
    n = launcher.prepare_manifest(cfg, launcher.MODELS["nv-reason-ct"], tmp_path / "t2.jsonl",
                                  holdout=6, holdout_dst=tmp_path / "h2.jsonl")
    assert n == 10 and not (tmp_path / "h2.jsonl").exists()


def test_evaluate_before_after_with_cli_runner(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "inference.py").write_text(STUB_INFERENCE)
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text((RCT / "sft.jsonl").read_text())
    cfg = launcher.resolve_config({}, {"model_id": "nv-reason-ct", "manifest": "/m", "image_dir": str(RCT)})
    out = tmp_path / "out"
    metrics, warnings = launcher.evaluate_before_after(cfg, launcher.MODELS["nv-reason-ct"], checkout,
                                                       str(tmp_path / "pretrained"), str(tmp_path / "finetuned"),
                                                       holdout, out, dict(os.environ))
    assert not warnings
    assert metrics == {"eval_before_findings_f1": 0.0, "eval_after_findings_f1": 1.0}

    table = show_eval.f1_table(out / "evaluation")
    assert list(table["after_answer"]) == ["No Chest Finding"] * 3 and table["before_f1"].max() == 0.0
    import matplotlib
    matplotlib.use("Agg")
    assert show_eval.show_case(out / "evaluation", 0, image_dir=str(RCT)) is not None
