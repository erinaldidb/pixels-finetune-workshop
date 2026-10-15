"""MLflow pyfunc wrapper for a fine-tuned NV-Segment VISTA3D model (models-from-code).

Logged by train_segment_finetune.py with three artifacts:
  bundle         upstream bundle configs/ + scripts/ at the pinned commit
  checkpoint     model_softmax.pt (softmax workflow) or model_finetune.pt (standard)
  label_mapping  label_mapping.json with the ordered [dataset_label, class_id] pairs

predict() takes a DataFrame with an `image_path` column (NIfTI, e.g. a UC Volume
path) and runs the upstream inference config for each image in a subprocess:
  softmax   configs/inference_softmax.json  → mask in dataset labels (1..N)
  standard  configs/inference.json + label_prompt → mask in VISTA3D class IDs
It returns `label_path` (written under params["output_dir"]) and `class_voxels`
(JSON {class name: voxel count}). Needs a CUDA GPU for practical runtimes.
"""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import mlflow
import pandas as pd

DEFAULT_OUTPUT_DIR = "/tmp/nv_segment_predictions"


class NVSegmentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = Path(context.artifacts["bundle"])
        self.checkpoint = context.artifacts["checkpoint"]
        mapping = json.loads(Path(context.artifacts["label_mapping"]).read_text())
        self.label_mapping = mapping["label_mappings"]["default"]
        self.classes = mapping["classes"]  # {dataset_label: {"class_id", "name"}}
        cfg = context.model_config or {}
        self.workflow = cfg.get("workflow", "softmax")
        self.patch_size = cfg.get("patch_size", [128, 128, 128])

    def _override(self, image: str, out_dir: Path) -> tuple:
        if self.workflow == "softmax":
            return "inference_softmax.json", {
                "input_file": image, "checkpoint_path": self.checkpoint,
                "label_mappings": {"default": self.label_mapping}, "patch_size": self.patch_size,
            }
        return "inference.json", {
            "input_dict": {"image": image, "label_prompt": [c for _, c in self.label_mapping]},
            "checkpoint_path": self.checkpoint,
        }

    def _names_by_value(self) -> dict:
        key = "class_id" if self.workflow == "standard" else None
        return {(v["class_id"] if key else int(s)): v["name"] for s, v in self.classes.items()}

    def predict(self, context, model_input, params=None):
        import nibabel as nib
        import numpy as np

        out_root = Path((params or {}).get("output_dir") or DEFAULT_OUTPUT_DIR)
        names = self._names_by_value()
        rows = []
        for image in model_input["image_path"]:
            out_dir = out_root / uuid.uuid4().hex[:8]
            config, extra = self._override(str(image), out_dir)
            override = out_dir / "override.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            override.write_text(json.dumps({"bundle_root": str(self.bundle), "output_dir": str(out_dir),
                                            "output_postfix": "seg", **extra}))
            cmd = [sys.executable, "-m", "monai.bundle", "run",
                   "--meta_file", str(self.bundle / "configs" / "metadata.json"),
                   "--config_file", f"['{self.bundle / 'configs' / config}','{override}']"]
            env = {**os.environ, "PYTHONPATH": f"{self.bundle}:{os.environ.get('PYTHONPATH', '')}"}
            proc = subprocess.run(cmd, cwd=str(self.bundle), env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"inference failed for {image}: {proc.stderr[-2000:]}")
            pred_path = next(iter(sorted(out_dir.rglob("*_seg.nii.gz"))), None)
            if pred_path is None:
                raise RuntimeError(f"no prediction written for {image} under {out_dir}")
            values, counts = np.unique(np.asanyarray(nib.load(str(pred_path)).dataobj), return_counts=True)
            voxels = {names[int(v)]: int(n) for v, n in zip(values, counts) if int(v) in names}
            rows.append({"label_path": str(pred_path), "class_voxels": json.dumps(voxels)})
        return pd.DataFrame(rows, columns=["label_path", "class_voxels"])


mlflow.models.set_model(NVSegmentModel())
