# Evaluation — nv-segment-finetune Skill

## Overview

Evaluates the `nv-segment-finetune` skill (scaffolding and validating AI
Runtime projects for NV-Segment-CT / NV-Segment-CTMR VISTA3D fine-tuning) on
4 benchmark tasks: the workshop project, choosing a workflow for NV-Segment-CT,
a dataset with stray labels, and repairing a script with wrong MONAI override
keys.

**Skill under test**: `/Workspace/Users/<user>/.assistant/skills/nv-segment-finetune/SKILL.md`

## Benchmark Tasks

| task_id | difficulty | Description | Key skill behaviour tested |
|---------|-----------|-------------|---------------------------|
| seg-001 | easy | Workshop project for NV-Segment-CTMR | Uses the scaffold, validates, does not submit |
| seg-002 | hard | ImageCHD on NV-Segment-CT with softmax | Detects that the chamber IDs are missing from the CT label dict; offers standard or CTMR |
| seg-003 | hard | 4-chamber dataset with leftover labels 5-9 | Ordered foreground-only mapping, `check_labels`, no `[v, 0]` padding |
| seg-004 | hard | Repair a script with `train#trainer#max_epochs` / `data_file_base_dir` | Top-level `epochs`/`dataset_dir`, softmax config run alone |

## Scorer Summary

Generate `scorers.py` and the scoring notebook with the
[skill-eval](https://github.com/databricks-industry-solutions/hls-skills/tree/main/skills/skill-eval)
skill from `evalset.json` and `expectations.json`.

- **Deterministic**: `required_artifacts` present, `forbidden_patterns`
  absent, and the project passes `scripts/validate_project.py` (exit code 0).
- **LLM judges**: `Correctness` against `expected_facts`, task completion, and
  tool-use quality. The judge model must differ from the generator.

## Ship Gate

- **Pass rule**: `all_metrics` — a task passes only if all its metrics pass.
- **Ship threshold**: `win_rate > 0` AND zero regressions.

## Run Protocol

1. Generate the seg-003 data on a UC Volume (`nibabel`, `numpy`):
   `python generate_data.py /Volumes/<catalog>/<schema>/eval/nv_segment/chambers_stray_labels`.
   Update the `dataset` paths in `evalset.json` if you use another Volume.
2. For seg-004, prepare a starting project in the folder named in the query: a hand-written
   `train_ctmr_finetune.py` that sets the MONAI overrides `train#trainer#max_epochs` and
   `data_file_base_dir` and stacks `train.json` + `train_continual.json` +
   `train_continual_softmax.json`, plus a matching `train.yaml` and `finetune_config.yml`.
3. **Baseline (skill OFF)**: move the skill folder out of `.assistant/skills/`.
   For each task, open a fresh Genie Code chat and paste the query verbatim.
4. **With skill (skill ON)**: restore the folder and repeat in fresh chats.
5. Score both arms with the skill-eval notebook. Write `baseline_scores.json`,
   `with_skill_scores.json`, and `eval_report.md` here.

> This folder holds the answer key and is excluded from the Unity Catalog sync.
> Do not copy `expectations.json` into `references/`.
