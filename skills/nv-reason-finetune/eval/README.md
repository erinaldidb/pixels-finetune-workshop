# Evaluation — nv-reason-finetune Skill

## Overview

Evaluates the `nv-reason-finetune` skill (scaffolding and validating AI
Runtime projects for NV-Reason-CXR / NV-Reason-CT SFT and GRPO fine-tuning)
on 4 benchmark tasks: a CXR SFT project, an 8-GPU CT smoke run, a GRPO
request with an SFT-format manifest, and a single-GPU CT request.

**Skill under test**: `/Workspace/Users/<user>/.assistant/skills/nv-reason-finetune/SKILL.md`

## Benchmark Tasks

| task_id | difficulty | Description | Key skill behaviour tested |
|---------|-----------|-------------|---------------------------|
| rsn-001 | easy | NV-Reason-CXR SFT project | Uses the scaffold, model-specific pins, manifest validation, license surfaced |
| rsn-002 | medium | NV-Reason-CT SFT on 8×H100, 8-step smoke | DeepSpeed in `command` with `--no-build-isolation`, `max_steps` override |
| rsn-003 | hard | GRPO with an SFT-format manifest | Validator failure surfaced; GRPO row schema explained; no bypass |
| rsn-004 | hard | NV-Reason-CT on a single H100 | OOM warning, `freeze_llm` / 8-GPU options, `ddp.yaml` without DeepSpeed |

## Scorer Summary

Generate `scorers.py` and the scoring notebook with the
[skill-eval](https://github.com/databricks-industry-solutions/hls-skills/tree/main/skills/skill-eval)
skill from `evalset.json` and `expectations.json`.

- **Deterministic**: `required_artifacts` present, `forbidden_patterns`
  absent, and the project passes `scripts/validate_project.py` (exit code 0;
  rsn-003 is expected to fail validation until the manifest is fixed).
- **LLM judges**: `Correctness` against `expected_facts`, task completion, and
  tool-use quality. The judge model must differ from the generator.

## Ship Gate

- **Pass rule**: `all_metrics` — a task passes only if all its metrics pass.
- **Ship threshold**: `win_rate > 0` AND zero regressions.

## Run Protocol

1. Generate the synthetic data on a UC Volume (`numpy`, `nibabel`, `Pillow`):
   `python generate_data.py /Volumes/<catalog>/<schema>/eval/nv_reason`.
   Update the `dataset` paths in `evalset.json` to match.
2. **Baseline (skill OFF)**: move the skill folder out of `.assistant/skills/`.
   For each task, open a fresh Genie Code chat and paste the query verbatim.
3. **With skill (skill ON)**: restore the folder and repeat in fresh chats.
4. Score both arms with the skill-eval notebook. Write `baseline_scores.json`,
   `with_skill_scores.json`, and `eval_report.md` here.

> This folder holds the answer key and is excluded from the Unity Catalog sync.
> Do not copy `expectations.json` into `references/`.
