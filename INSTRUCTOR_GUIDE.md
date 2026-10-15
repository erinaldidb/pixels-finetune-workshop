# Instructor Guide: Fine-Tune NVIDIA VISTA3D on Databricks AI Runtime with Genie Code

This guide covers everything the instructor does **around** the participant notebook `AI-Runtime-MONAI-Workshop.ipynb`: preparation, the rehearsal run, how to run the session, what to say at each step, and how to recover when something goes wrong.

| | |
| --- | --- |
| **Audience** | Data scientists and ML engineers new to AI Runtime, MONAI or Genie Code. No medical-imaging background required |
| **Duration** | About 2 hours, including 20–30 min of GPU training (estimate; confirm it in the rehearsal) |
| **Participants build** | A fine-tuned NV-Segment-CTMR (VISTA3D) model for 7 heart structures, a before/after comparison, and a model version in `<catalog>.ws_<user>` |
| **Instructor demo** | NV-Reason-CXR / NV-Reason-CT (notebook Appendix A) |
| **Materials** | `AI-Runtime-MONAI-Workshop.ipynb`, `skills/nv-segment-finetune/`, `skills/nv-reason-finetune/` |

---

## 1. How the workshop is designed

Participants **don't write code**. Each notebook step has the same four parts:

1. **Goal**: what the step achieves
2. **Ask Genie Code**: the prompt to paste into Genie Code (Agent mode)
3. **What Genie Code does — and why**: each component used, explained
4. **What you should see**: the expected result, for self-checking

Genie Code does the work through two **skills**, folders with a `SKILL.md`, templates and tested scripts:

| Skill | Used for |
| --- | --- |
| `nv-segment-finetune` | Participant track: scaffold, validate, train launcher, before/after evaluation, CPU plots, Unity Catalog registration |
| `nv-reason-finetune` | Instructor demo: SFT/GRPO for the NV-Reason vision-language models |

The key message for participants: **the skill keeps Genie Code on a pinned, validated, upstream path.** It runs NVIDIA's own training entry point (`python -m monai.bundle run`) at a pinned commit and weights revision. It doesn't improvise training code. Encourage participants to read Genie Code's plan before approving each cell.

---

## 2. Preparation timeline

### About 2 weeks before

- [ ] **Confirm workspace features:**
  - AI Runtime enabled
  - Genie Code with **Agent mode** available to all participants
  - Serverless notebook compute available
- [ ] **Request GPU capacity.** Every participant submits one `GPU_1xH100` job at roughly the same time (Step 5). Plan for *N participants = N concurrent H100s*, or set up a GPU pool. If capacity is limited, see [Running with limited GPUs](#running-with-limited-gpus).
- [ ] **Check the license:** NV-Segment models are under NVIDIA's model license (see the Hugging Face model card). The workshop is for engineering and research use, **not clinical use**. Say this at the start.
- [ ] **Collect the participant list** as a workspace group, for example `monai-workshop-participants`.

### About 1 week before

#### 2.1 Upload the dataset

Upload ImageCHD (110 CT volumes) to a Unity Catalog Volume as `ct_XXXX_image.nii.gz` / `ct_XXXX_label.nii.gz` pairs, e.g.:

```
/Volumes/pixels_workshop_catalog/image_chd/default/imagechd/extracted/ImageCHD_dataset
```

Check: 110 images, 110 labels, matching IDs.

#### 2.2 Install the skills workspace-wide

Copy both skill folders to `/Workspace/.assistant/skills/`. This needs workspace-admin rights. From the folder containing `skills/`:

```bash
databricks workspace import-dir skills/nv-segment-finetune /Workspace/.assistant/skills/nv-segment-finetune
databricks workspace import-dir skills/nv-reason-finetune  /Workspace/.assistant/skills/nv-reason-finetune
```

Then check that:

- `SKILL.md` is at the top of each skill folder
- the `.py` files under `scripts/` and `assets/` were imported as **files**, not notebooks. The notebook imports `visualize.py` and `register_model.py` from `SKILL_DIR/scripts` via `sys.path`, which only works with files.
- participants can **read** `/Workspace/.assistant/skills/` (open it as a non-admin test user)

The `tests/` and `eval/` folders aren't needed at runtime. It's harmless to copy them.

#### 2.3 Grant Unity Catalog permissions

Replace the catalog, schema, volume and group names with yours:

```sql
-- Participants create their own ws_<user> schema in the workshop catalog (Step 0c) and register into it (Step 8)
GRANT USE CATALOG, CREATE SCHEMA ON CATALOG <workshop_catalog> TO `monai-workshop-participants`;

-- Read access to the dataset volume
GRANT USE CATALOG ON CATALOG pixels_workshop_catalog TO `monai-workshop-participants`;
GRANT USE SCHEMA  ON SCHEMA  pixels_workshop_catalog.image_chd TO `monai-workshop-participants`;
GRANT READ VOLUME ON VOLUME  pixels_workshop_catalog.image_chd.default TO `monai-workshop-participants`;
```

The schema creator owns their `ws_<user>` schema, so no further grants are needed for registration.

#### 2.4 Do a full rehearsal run, which becomes the reference run

Run the **whole participant notebook yourself**, as a participant would, from a clone. This is the most important preparation step: **the GPU-side code (training, before/after evaluation, MLflow model logging) has so far been tested only on CPU, with stubs. It has not been run on a GPU or AI Runtime.**

Check:

| Check | Where |
| --- | --- |
| Job provisions and installs dependencies (`monai==1.4.0`, `numpy<2`, …) | Job Run logs |
| Bundle clone and weights download succeed | Job Run logs: "Stage bundle" |
| Training runs and `val_mean_dice` rises per epoch | MLflow run → metrics |
| **Wall-clock time for 30 cases × 10 epochs**; update the agenda if it isn't about 20–30 min | Job Run duration |
| `evaluation/` artifact has `dice.json` and `slices_*.npz`; the run has `eval_before_mean_dice` / `eval_after_mean_dice` | MLflow run → artifacts / metrics |
| A `run_summary.json` without evaluation warnings | MLflow run → artifacts |
| `model/` artifact exists and run tag `segment.model_uri` is set | MLflow run |
| Steps 7 plots render; Step 8 registers version 1 with alias `workshop` | Notebook |
| Whether the job's `experiment_name` (in `train.yaml`) creates a second MLflow experiment next to `EXPERIMENT_NAME` | MLflow UI. If it does, tell participants where to look in Step 6 |
| The before/after difference is clearly visible with the short run | Step 7 plots |

If evaluation fails, training still succeeds. The run records a warning in `run_summary.json` instead of failing. Fix the cause before the workshop; otherwise Step 7 has nothing to show.

#### 2.5 Share the reference run

1. Note the rehearsal **MLflow run ID**. This is `REFERENCE_RUN_ID`.
2. Give the participant group **CAN READ** on the rehearsal's MLflow experiment (Experiment → Permissions).
3. Test with a non-admin user: `visualize.load_evaluation(REFERENCE_RUN_ID)` works, and `register_run(REFERENCE_RUN_ID, CATALOG, schema=<their schema>)` registers into *their* schema. Cross-user registration from a run you can only read hasn't been verified; confirm it works.

### Day before

- [ ] Prepare the **three values** to share, on a slide or in chat:

  ```
  CATALOG          = "<workshop_catalog>"
  DATASET_PATH     = "/Volumes/pixels_workshop_catalog/image_chd/default/imagechd/extracted/ImageCHD_dataset"
  REFERENCE_RUN_ID = "<rehearsal run id>"
  ```

- [ ] Put the participant notebook somewhere everyone can read and clone it, e.g. `/Workspace/Shared/monai-workshop/`
- [ ] Check GPU capacity again
- [ ] Prepare the Appendix A demo (see [section 5](#5-instructor-demo-nv-reason-appendix-a)), or skip it
- [ ] Have the rehearsal run's Job Run page, MLflow run and Unity Catalog model open in browser tabs as a fallback

---

## 3. Run of show

| Time | Step | Instructor focus |
| --- | --- | --- |
| 0:00 | Intro (10 min) | Goal, model, data, how skills + Genie Code work, architecture diagram, "not for clinical use" |
| 0:10 | **0** Setup | Share the three values. Watch for schema permission errors |
| 0:20 | **1** Explore data | Explain HU, label masks, class IDs |
| 0:30 | **2** Generate project | Emphasise: the skill writes the code, pinned and validated |
| 0:40 | **3** Project files | The four files and their separate jobs; settings priority |
| 0:55 | **4** Validate + size | Everyone must reach `"ok": true` before submitting |
| 1:05 | **5** Submit | Get everyone submitted within 5 min. **This is the critical path** |
| 1:10 | **6** Monitor | While the jobs train (20–30 min): theory break, Q&A, or the Appendix A demo |
| 1:35 | **7** Before vs after | Use the reference run if a job isn't finished |
| 1:50 | **8** Register in UC | Dry run first, then register |
| 2:00 | **9** Wrap-up | Clean-up, next steps |

**The main timing rule:** get every job submitted by about 1:10, so it finishes before Step 7. If a group is slow in Steps 2–4, help them so they can submit, and explain the files afterwards.

---

## 4. Step-by-step facilitation notes

### Intro

- **Show the architecture diagram:** the notebook runs on **CPU**, and only the training job uses a GPU. Plots and registration need no GPU because the job leaves small evaluation artifacts behind.
- **Explain what a skill is:** `SKILL.md` + templates + scripts. Genie Code picks it when the prompt matches its description. Open `/Workspace/.assistant/skills/nv-segment-finetune/SKILL.md` briefly to show it's plain, readable instructions.
- **Explain the softmax workflow** with the class-ID table: each of the 7 structures starts from a class VISTA3D already knows (e.g. 151 = left ventricle). That's why fine-tuning works with a short run.

### Step 0: Setup

- **Install once:** `%pip install` then `%restart_python` runs **once**. Later restarts lose variables, so participants only re-run the 0b configuration cell.
- **Schema name:** `SCHEMA = ws_<user>` is derived from the email address, so `jane.doe@x.com` becomes `ws_jane_doe`. Two participants with the same local part in different domains would collide. Rare; have them add a suffix.
- **Common issue:** schema creation fails because of a missing `USE CATALOG` / `CREATE SCHEMA` grant (section 2.3).

### Step 1: Explore the data

- **Talking points:**
  - Volumes are ordinary paths, governed by UC.
  - The soft-tissue window (−200..600 HU) makes contrast-filled blood bright.
- **Point at the voxel counts:** myocardium and pulmonary artery are small, so they are hard. That's the prediction to check in Step 7.

### Step 2: Generate the project

- **What to look for:** Genie Code should **run `scripts/scaffold_project.py`**, not write files from scratch. If it starts writing its own training code, stop it and re-prompt: *"Use the nv-segment-finetune skill's scaffold script."*
- **Expected result:** `"ok": true`, four files `created`, `"workflow": "softmax"`.
- **Re-running:** existing files are kept unless `--force` is used. A re-run won't overwrite a participant's edits.

### Step 3: Project files

- **The table in the notebook is the lesson:**
  - `train.yaml` = infrastructure
  - `finetune_config.yml` = what to train
  - launcher = runs upstream on the GPU
  - pyfunc = the model wrapper
- **Stress the two most common mistakes:**
  - never put `torch` in `dependencies`
  - `environment.version` must be a **quoted** string (`'5'`)

  The validator catches both.
- **Label mapping:** the order defines output channels. Never add padding entries.

### Step 4: Validate and size the run

- **Size the run:** participants set `max_cases: 30`, `epochs: 10`. Use the timing you measured in the rehearsal and correct the agenda if needed.
- **Gate:** don't let anyone submit with a `✗`. Read the check's detail with them.
- **Confirm the resolved settings:** `--print-config` should show `epochs: 10`, `max_cases: 30`, `eval_cases: 3`.

### Step 5: Submit

- **Why credentials are exported:** `air` is a subprocess and doesn't inherit the notebook login. Make sure Genie Code **masks the token** in its output.
- **No `--watch`:** it would block the cell for the whole run.
- **Provisioning time:** 2–10 min, longer when many jobs start at once.
- **Count successful submissions** (e.g. with a show of hands or chat reactions), so you know who needs the reference run later.

### Step 6: Monitor (training time)

- **Show the run:** walk through the Job Run logs on the projector (stage data → stage bundle → train → evaluation → log model) and the MLflow `val_mean_dice` curve. Epoch 0 validates the starting weights.
- **Use the 20–30 min wait:**
  - **Option A:** the Appendix A demo (NV-Reason)
  - **Option B:** Q&A on adapting the skill to their own data (`references/adding-a-model.md`)
  - **Option C:** a break
- **Jobs that aren't finished yet:** participants set `RUN_ID = REFERENCE_RUN_ID` and continue.

### Step 7: Before vs after

- **How before and after were produced:** both on the **same held-out validation cases**, through the upstream MONAI configs.
  - **Before:** pretrained zero-shot with class prompts (`inference.json`)
  - **After:** the fine-tuned softmax model (`inference_softmax.json`)
- **Expected:** "after" higher than "before" for most structures. Small structures usually gain the most.
- **Caveat to say aloud:** 3 cases with a short run is an engineering check, not a clinical evaluation. Absolute Dice is below the full 50-epoch recipe.

### Step 8: Register in Unity Catalog

- **Dry run first:** it shows the target name `CATALOG.ws_<user>.nv_segment_ctmr_finetuned` and the model URI without changing anything.
- **Talking points:**
  - **Versions vs aliases:** `@workshop` points at the version just made, so consumers don't hard-code version numbers.
  - **Tags:** they record the upstream commit, the weights revision and the before/after Dice. That's what makes the model traceable.
  - **Signature:** UC requires one. Input `image_path`; outputs `label_path` and `class_voxels`.
- **Running the model:** it needs a GPU, a job or a GPU serving endpoint. Registering it doesn't.
- **Show lineage:** Catalog Explorer → their schema → Models → version → lineage to the run.

### Step 9: Wrap-up

- **Clean-up:** make sure everyone **cancels runs still going**. Remind them that `DROP SCHEMA … CASCADE` is optional and deletes their registered model.
- **Next steps:**
  - the full-quality run (`max_cases: null`, `epochs: 50`, 2–3 h)
  - batch inference with `@workshop`
  - their own data
  - NV-Segment-CT (Appendix B)

---

## 5. Instructor demo: NV-Reason (Appendix A)

**Purpose:** show that the same pattern (skill → scaffold → validate → submit → before/after) applies to vision-language models trained with Hugging Face TRL (SFT and GRPO).

### Prerequisites

You can't do this live without them.

| | NV-Reason-CXR-3B | NV-Reason-CT |
| --- | --- | --- |
| Data | MIMIC-CXR-JPG / CheXpert (**credentialed**) | CT-RATE (**gated** on Hugging Face) |
| Default compute | `GPU_8xH100` | `GPU_8xH100` (1 GPU only with `freeze_llm`) |
| License | NVIDIA OneWay **Noncommercial** | OpenMDW-1.1 |
| Pins | `transformers==4.56.1`, `trl==0.22.2` | `transformers==5.6.2`, `trl==1.2.0`, `accelerate==1.13.0` |

You need a JSONL manifest (SFT rows end with the assistant answer; GRPO rows have a `solution`) and an image folder in a Volume.

### Recommended approach: pre-recorded

Run the smoke job (`training.max_steps: 8`, `dataset.max_cases: 32`) **before** the workshop. In the session, show:

1. **Generate:** the scaffold prompt live. It's CPU-only and fast. Point out the model-specific pins and the DeepSpeed install line in `train.yaml`.
2. **Explain:** the launcher runs upstream `accelerate launch train/vlm_sft_train.py` with these AI Runtime overrides:
   - pinned weights downloaded first
   - MLflow instead of W&B
   - DeepSpeed ZeRO-2
   - for CXR GRPO, vLLM off
3. **Results:** the pre-recorded run's MLflow page.
4. **Before vs after:** `show_eval.f1_table(...)` and `show_eval.show_case(...)` on the pre-recorded run. Findings F1 compares predicted `<answer>` findings with the verified `solution`. **Read the texts too**, since F1 is a coarse metric.

**State the limits:** UC registration for NV-Reason isn't part of the workshop. The weights are logged under the run's `model/` artifacts. Like the segmentation track, the NV-Reason GPU path hasn't been run on AI Runtime yet, so rehearse the demo.

---

## 6. Troubleshooting (instructor view)

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Genie Code ignores the skill and writes its own code | Prompt didn't trigger the skill, or skills not visible | Re-prompt naming the skill explicitly; check `/Workspace/.assistant/skills/` is readable |
| `ModuleNotFoundError: visualize` / `register_model` | Scripts imported as notebooks, or wrong `SKILL_DIR` | Re-import as files (section 2.2); check `sys.path` in Step 0b |
| `Cannot create <catalog>.ws_<user>` | Missing grant | `GRANT USE CATALOG, CREATE SCHEMA …` (section 2.3) |
| Dataset count ≠ 110 or permission denied | Missing `READ VOLUME` / wrong `DATASET_PATH` | Fix the grant or the shared path |
| `air run` authentication error | Credentials not exported in the same process | Re-run the Step 5 export |
| `INTERNAL_ERROR` at provisioning | GPU capacity | Retry, switch to `GPU_1xA100` or the pool, or use the reference run |
| Job stuck in provisioning > 15 min | Capacity queue | Move the participant to the reference run; let the job continue |
| CUDA out of memory | Patch/cache too large | `training.patch_size: [96,96,96]` or smaller; lower `cache_rate` |
| Run finished but Step 7 finds no `evaluation/` | Evaluation failed (non-fatal) | Check `run_summary.json` warnings; use the reference run |
| `register_run` says the run isn't successful | Run failed or still running | Wait, or use the reference run |
| Old code runs on the GPU | Files edited after submitting | Re-submit; each run uploads a fresh snapshot |
| Variables lost | Python restarted | Re-run the Step 0b cell only |

### Running with limited GPUs

If you can't get one H100 per participant:

- **Pairs:** one job per pair. Both participants use the same `RUN_ID` for Steps 7–8 and register into their own schemas.
- **Instructor-only training:** participants do Steps 0–4, one volunteer submits, and everyone else uses `REFERENCE_RUN_ID` for Steps 7–8.
- **Smaller runs:** a smaller `max_cases` or fewer `epochs` reduce GPU-minutes. The before/after gap shrinks, so check it in the rehearsal.

---

## 7. After the workshop

- [ ] Cancel any jobs still running: Jobs UI, or `databricks air cancel <run>`
- [ ] Decide on a retention period for the `ws_<user>` schemas and announce it. To clean up: `DROP SCHEMA <catalog>.ws_<user> CASCADE`
- [ ] Remove or keep the grants on the workshop catalog and dataset volume
- [ ] Keep the reference run and its experiment permissions until participants have finished any follow-up work
- [ ] Collect feedback, especially on step timing and on any prompts that didn't trigger the skill

---

## 8. Known limitations (be transparent with participants)

- **Not run on a GPU or AI Runtime yet:**
  - **What was tested:** all skill code was tested on CPU only (24 segment + 18 reason tests), with stand-ins for MONAI and the NV-Reason inference.
  - **Before the first session:** the rehearsal (section 2.4) is mandatory.
- **Training time:** "20–30 min for 30 cases × 10 epochs" is an estimate.
- **NV-Segment-CT standard workflow (Appendix B):** evaluation through `configs/evaluate.json` hasn't been tried with brand-new classes. "Before" has no meaningful mask for classes the pretrained model has never seen.
- **Evaluation size:** 3 validation cases, an engineering check rather than a clinical evaluation.
- **NV-Reason:** no Unity Catalog registration. The weights are logged as run artifacts.
