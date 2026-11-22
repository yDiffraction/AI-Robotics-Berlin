# Dataset sources and splits

Everything listed here is either our own recording or a public dataset. **The Durance hand-gesture dataset is not
used for the throw models and is not in this repo.**

## Sources

| Source | What | Where | Used by |
|---|---|---|---|
| Own recordings, sessions `01`–`04` (21 Nov 2026) | GenX320 on the Kria, one thrower (person01), throws, feints, idle, walking, picking up the ball | `dataset/01`–`04`, clips in `dataset/_dataset01_*` | throwdet, early_ctx, pretrain fine-tune |
| Own live sessions (22 Nov 2026) | camera → board → label page, a person presses SPACE when the net should fire | `perception/hitl/sessions/` (not in git, ~18 MB of `.npz`; available on request) | HITL RL step → `model_hitl_step1` |
| Augmented / spliced / synthetic clips | derived from the own recordings; procedural synthetic throws and feints; phone videos converted to events | regenerate with `training/throwdet/datagen/` (see `DATASET.md` there, manifest in `manifest.csv`) | throwdet |
| DailyDVS-200 (public, MIT licence) | real DVS recordings of 47 people; throw/toss classes as positives, arm swings without release and idle actions as negatives | `training/throwdet/datagen/dailydvs_import.py` | throwdet |
| IBM DVS Gesture (public) | 11 hand/arm gestures | `training/pretrain/prep_dvsgesture.py` | backbone pretraining |

## Splits

**Rule used everywhere: split by recording / session, never by clip.** Clips cut from the same recording look
almost identical, so a clip-level split leaks.

| Model / study | Train | Held out | Notes |
|---|---|---|---|
| throwdet CV (`training/throwdet/train.py`) | 3 of 4 own recording groups + all external data | 1 own group per fold (grouped 4-fold CV by recording); the 5.8 min idle recording is cut into 4 contiguous time chunks, one per fold; DailyDVS persons 3, 4, 5, 7, 10, 11, 16, 24, 27, 31, 33, 37, 41, 42, 43, 45 (the official val/test persons) | aug/splice clips derived from a held-out recording are removed from that fold's training set |
| `D_e10_fold3` (parent of d_neg11) | own 02, 03, 04, idle chunks 0–2, all external | own 01 (25 swings, played as one 55 s stream) | the only throwdet model with an out-of-fold threshold |
| `D_person01_neg` → `d_neg11` | head fine-tune of `D_e10_fold3` on 121 labelled 3 s windows of person01 | **none** | its 45/53 throws, 2/68 false alarms are **in-sample** |
| `hitl_step1` (RL) | live session `20261122_151118` | chosen in a separate live A/B session (`20261122_153235`) against its start model and RL snapshots 80–470 | that scoreboard was shown live and not saved; it can be recomputed from the session recording |
| early_ctx study (`experiments/early_ctx_eval`) | fold f1 sessions / fold f0 sessions | f0 = 170441, 201703, 215305; f1 = 195808, 200001, 200828, 201452, 201622 | 211821 (idle) is training-only and never scored |
