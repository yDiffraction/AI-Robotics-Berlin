# panel / synthesize (phase Synthesize)

{
 "primary_metric": "Within-fold pairwise AUC for throw vs feint, on held-out clips from datasets 02 and 03 in folds A, B and C (42 throws, 65 feints, 993 within-fold pairs), averaged over training seeds. For each clip, the float model runs on the binary clip from frame 0 with a fresh state. S(t) = sum over u in [t-9, t] of (throw spikes(u) - no-throw spikes(u)). Throw score = max S(t) for t in [r, r+25], where r is the bank's 'release': release_ms/10 for the 33 tracked throws, else 70. r is the same for every variant. Feint score = max S(t) for t >= 10. AUC = (sum over folds k of sum over held-out throws i and feints j of fold k of [s_i > s_j] + 0.5*[s_i = s_j]) / (sum over k of n_throw_k * n_feint_k). Pairs never cross folds, so differences in calibration between the three fold models cannot enter. d01 clips and fold D are excluded and reported separately. Do not rank by compare.py's auc_all. It pools the three fold models, and 465 of its 533 negatives are idle clips, so about 87% of it measures throw vs idle.",
 "eval_protocol": "FOLDS: ball_cv's fixed spec A={200828,201703}, B={200001,195808,201452}, C={170441,201622,d01}. D={215305} stays in training and is tested only in E9. The 04 idle take is cut into 4 contiguous time blocks with the guard clips dropped; block k is the test idle of fold k. Scene-idle clips go with their session. EVERY RUN: 1200 steps, bs 32, init runs/robust_tim/init/best.pt, EMA 0.99 with clean BN refresh, the same eval code, float model on binary clips. Replicates use --seed with a distinct tag per seed. A reused tag overwrites runs_ball/<tag>, and queue.sh skips tags already in done.txt. Folds are deterministic, so never use --cv-seeds for replicates. Run ONE GPU job at a time: the running job uses 7.3 of 8.2 GB and hit an OOM retry during eval. Expect about 25 min per 3-fold run (fold A of b0_tim took 8.2 min). GPU ORDER: b0_tim (running) -> w1_win -> d1_drop -> w1_win_s1 -> w2_paste (feint-only) -> a1_aux -> m1_mil -> w3_sceneidle -> at most two seed-1 confirmations -> E9 -> E10. Stretch, only if time is left: b1_fix, p1_pop5, w4_erase. The budget is about 280 GPU min. If time runs short, drop in this order: E9 fold D, the second confirmation, E8, E7. DECISION RULE: noise floor d0 = |primary(w1_win) - primary(w1_win_s1)|. A challenger passes the seed-0 screen if its paired delta vs w1_win is >= max(0.03, d0) AND at least 2 of folds A/B/C improve. The two best passers get seed 1, paired with w1_win_s1. A change is adopted into E9 if all of the following hold: the mean delta over both seeds is >= max(0.03, d0); the delta is > 0 in each seed; the guardrails hold: rec250 on 02/03 throws drops by <= 0.10, busy% and scene% rise by <= 0.05 absolute, early@f10 rises by <= 0.05. E8 is judged instead on scene%: a relative drop of >= 30% with a primary delta >= -0.02. If d0 > 0.06, single-seed screens are uninformative: run E4 and E5 at two seeds each and skip E7 and E8. UNCERTAINTY: report deltas per fold and per seed, plus a paired bootstrap CI over clips (2000 resamples stratified by fold x class, the same resample for both runs, as compare.py does). State that this CI excludes seed noise; d0 is that component. ALWAYS REPORT: pooled auc_tf and auc_all, rec50/100/250, median latency, busy%, scene%, early@f10, d01 recall, rec_tracked and cf_erase (ball reliance = rec_tracked - cf_erase), cf_paste (except for paste-trained runs, where it is circular; cf_erase is circular for erase-trained runs), and the E1 tracker rows. WINNER'S CURSE: E9 is picked on the same folds and seeds, so its gain is optimistic. The only unbiased check is a fresh recording in the demo framing (10 throws, 10 feints, 20-30 min idle) scored with the E10 rule.",
 "experiments": [
  {
   "id": "E1",
   "name": "Causal classical tracker and count-only baselines (CPU; tracker in progress in runs/ball/tracker)",
   "change": "No training. Score runs/ball/tracker/causal_tracker.py with exactly the ball_cv clip protocol. Per held-out fold A/B/C: a throw is a hit if the tracker fires in [r, r+25] (r = bank release); a feint or idle clip is a false trigger if it fires at any frame >= 10. Report TPR, feint FA, and busy- and scene-idle FA per fold, split 02/03 vs d01, plus triggers/min on the 04 idle take played as one continuous stream. Add two count-only baselines on the same folds (no spatial information): (1) L2 logistic regression on the 2x120 per-frame ON/OFF active-cell counts; (2) the same restricted to frames < 55 (prefix only). Report within-fold AUC_tf for both. For every SNN run, read off its recall at the tracker's feint FA.",
   "rationale": "This sets the bar. The look-back tracker finds 33 of 42 02/03 throws with 0 of 65 feints (tuned in-sample). The legacy SNN (b0_tim, fold A) has AUC_tf 0.59 and rec@f10 0.19. If the causal tracker still beats the SNN by more than 0.15 at equal feint FA, the GPU hours belong to the ball-centric changes (E4-E6), and a host-side confirm becomes the demo fallback. A prefix-only AUC above 0.6 would show label information before the release that any model can exploit.",
   "baseline": "None; these are reference rows for every SNN run",
   "local_minutes": 0,
   "depends_on": "none (CPU, runs alongside the GPU queue)"
  },
  {
   "id": "E2",
   "name": "Fixed baseline w1_win at two seeds (reference and noise floor)",
   "change": "Run the queued w1_win config unchanged. It is ball_cv.py v2: windows anchored at frame 70 for every clip; target at the release corrected for jitter and speed, taken from the bank's 'release'; clean BN refresh; --shift-x 8 --shift-y 2 --loss win --win 10 --idle-train busy; 1200 steps; init best.pt; folds A,B,C. Run it as tag w1_win (--seed 0) and again as tag w1_win_s1 (--seed 1). Move both ahead of b1_fix in jobs.txt. b0_tim (legacy Tim timing, running now) stays as the reference row.",
   "rationale": "Every challenger is measured against this config, and nothing in jobs.txt is replicated, so there is no noise estimate. With 42 ball-visible throws, the AUC's sampling SE alone is about 0.05. d0, the difference between two seeds of the identical config, is the smallest effect that counts in E4-E9. Compared with b0_tim, it shows what the timing and BN fixes bought, as an explicit bundle.",
   "baseline": "b0_tim (legacy reference; bundle comparison only)",
   "local_minutes": 50,
   "depends_on": "none"
  },
  {
   "id": "E3",
   "name": "Deployment-path check on the E2 fold models: fixed point, count input, idle stream (CPU)",
   "change": "No training. On CPU (CUDA_VISIBLE_DEVICES=''), load runs_ball/w1_win/fold_{A,B,C}.pt. Score the held-out clips with the ball_cv streaming metrics, changing one thing at a time; the reference is the float model on binary clips. (a) Board arithmetic: prepare_ptq(net.export_module(), Q2.6 weights / Q4.12 potentials, fuse=True). Report within-fold AUC_tf, rec250, busy%/scene%, and the ratio of fixed-point to float output spikes. (b) Live input format: throw/feint clips re-binned from the raw clip .bin without clamping, as min(count, 7), instead of binary; report the AUC_tf drop. Keep idle binary, because the idle clips are pre-thinned. (c) Continuous stream: per fold, play the held-out block of the 04 idle take in time order with state carried and a 150-frame refractory period; report triggers/min at the fold's f10 threshold.",
   "rationale": "All CV numbers are float on binary clips. The board runs Q2.6/Q4.12: lenses report +18% output spikes after PTQ on a fine-tuned model, and the fixed-point sim matched the board within about 0.2%. Despot sends counts, and the old 5-class model collapsed on counts on the board. A fixed-point AUC drop above 0.03 means the final training needs a QAT tail. A large drop with counts makes binarising before SPLEAT mandatory. Finding this early keeps thresholds from being calibrated in the wrong domain.",
   "baseline": "The same fold models, float, binary input",
   "local_minutes": 10,
   "depends_on": "E2 seed 0 (w1_win fold models)"
  },
  {
   "id": "E4",
   "name": "Drop arm-only throws (new tag d1_drop)",
   "change": "Add only --drop-throws 04,01 to the E2 config; screen at seed 0. This removes the 28 throws from 04 and the 7 from 01 from the positives. Their feints stay as negatives. No code change, so it can run right after w1_win.",
   "rationale": "In 04 the ball leaves the image at once, and in 01 it never separates from the arm. These 35 throw labels therefore mean 'arm motion', which is exactly what feints copy, and they are in the training set of every A/B/C fold (fold D is never tested). Removing them makes 'throw' mean 'the ball leaves the hand'. The price: positives fall from about 61 to 25-33 per fold, and d01 recall in fold C will likely drop. Report d01 recall separately.",
   "baseline": "E2 (w1_win, same seed)",
   "local_minutes": 25,
   "depends_on": "E2"
  },
  {
   "id": "E5",
   "name": "Feint + real ball = throw, feint-only pastes (w2_paste after a one-line fix)",
   "change": "Add only --synth-throw 0.3, after a small code change. Currently ball_cv.py line 183 sends half of the pastes into idle windows with a random anchor. Add --paste-idle (default 0.5 = current behaviour) and run with 0, so every paste goes into a training feint window. Everything else stays as already implemented: the paste lands at the feint's own swing phase (anchor + (rk - 70)), x at the activity centroid of the 10 frames before, tracks from training sessions only.",
   "rationale": "A pasted feint has feint arm motion and timing but a throw label, so the ball becomes the only cue consistent with every label. Each training fold also gets far more ball appearances than its roughly 20-25 tracked throws. Pastes into idle windows teach 'a rising blob without an arm = throw', which leads straight to false triggers on idle impostors such as head or hair edges reaching the top band. cf_paste is circular for this run; judge it on real clips and on cf_erase.",
   "baseline": "E2 (w1_win, same seed)",
   "local_minutes": 27,
   "depends_on": "E2; the --paste-idle flag"
  },
  {
   "id": "E6",
   "name": "Dense ball supervision of hidden features, alone (a1_aux instead of the bundled a1_aux_paste)",
   "change": "Add only --aux-ball 0.5. This is a training-only 1x1 conv on the layer-2 spikes that predicts the tracked ball cells per frame (BCE, pos_weight 20), with zeros on feints and idle and untracked throws masked. Run it without paste. a1_aux_paste changes two things relative to w1_win. If E5 is adopted, a1_aux_paste vs w2_paste becomes the attributable follow-up.",
   "rationale": "This distils the tracker's per-frame ball masks into the backbone: about 8-17 supervised ball frames per tracked throw instead of one clip label. Deployment cost is zero: the aux head is dropped at export and the arch stays v1. It should make the ball representable at layer 2 even when the clip-level loss could be satisfied by arm cues.",
   "baseline": "E2 (w1_win, same seed)",
   "local_minutes": 27,
   "depends_on": "E2"
  },
  {
   "id": "E7",
   "name": "MIL first-passage loss (m1_mil)",
   "change": "Change only the loss: --loss mil with the defaults --mil-tau 1 --neg-weight 1.4 --mil-dense 0.5. The soft max over the bag [r+3, r+23) is pushed up, the soft max over all negative frames is pushed down, and a dense win loss guards against collapse; the W=10 readout is unchanged. The current soft max is not length-normalised. If the run collapses or underperforms, rerun once with tau*(logsumexp - log n) before discarding MIL.",
   "rationale": "In 03 the ball reaches the top band 4-9 frames after release. The win loss asks for 'throw' on 20 frames (r+3..r+22), mostly without a ball, which rewards follow-through cues that feints imitate. MIL asks for one crossing inside the detection window and pushes down the any-time maximum on negatives. That is the same per-clip statistic the eval and the live trigger use.",
   "baseline": "E2 (w1_win, same seed)",
   "local_minutes": 25,
   "depends_on": "E2"
  },
  {
   "id": "E8",
   "name": "Same-scene idle in training (w3_sceneidle)",
   "change": "Add only --idle-train both. The 255 gap-idle clips from the 01-03 recordings join the 04 idle take as training negatives. They are grouped by session, so test sessions stay unseen.",
   "rationale": "With --idle-train busy, every training negative except the feints comes from the 04 idle take. The sessions 201452, 201622 and 201703 contain throws only, which invites scene shortcuts. The gap clips include walking and ball pickups: 15 overlap idle-labelled bursts, and none overlaps a throw label. Judge it on scene% (held-out same-scene idle that triggers at the f10 threshold), with the primary metric as a non-inferiority guardrail.",
   "baseline": "E2 (w1_win, same seed)",
   "local_minutes": 25,
   "depends_on": "E2"
  },
  {
   "id": "E9",
   "name": "Explicit bundle of the adopted changes, plus fold D",
   "change": "An explicit bundle: every change from E4-E8 that passed the decision rule, on top of the E2 config. Run seed 0 with --test-folds A,B,C,D; run seed 1 only if time is left. Skip E9 if nothing passed.",
   "rationale": "The deployed model combines the winners, and their effects may overlap (E4, E5 and E6 all push the model towards the ball). One paired check against E2 guards against interactions before the final build. Fold D shows what happens if the demo framing ends up like 04, where the ball is invisible.",
   "baseline": "E2 (same seed)",
   "local_minutes": 33,
   "depends_on": "Decisions on E4-E8"
  },
  {
   "id": "E10",
   "name": "Host trigger rule and threshold from dumped spikes, then the final model build",
   "change": "No new training config. (1) Dump per-frame output spikes, float and fixed point, for all held-out clips and the per-fold idle streams from the chosen config's fold models. (2) On CPU, sweep host rules on e(t) = throw minus no-throw spikes: box W in {5,10,20,50}; EMA tau in {3,5,10}; band-pass EMA4 - EMA30; the current rule (gesture_node mean over 50 + ratio > 0.6, confirm 2). Each rule fires on the first crossing, followed by a 150-frame refractory period. Set theta for fold k on the other folds at <= 10% any-time feint FA. Pick the rule with the best throw recall within [r, r+25]; break ties by lower median latency, then by fewer idle triggers/min. Also report an arm stage on a lower threshold of the same score: lead time before release and arm rate on idle. (3) Train --final with the chosen config. Drop the 04/01 throws in the final only if E4 was adopted. Export with prepare_ptq and pad the head to 11 outputs; export_tim.py and compare_fpga.py must be adapted to 'arch'. Set theta from fixed-point outputs, then replay a few recordings on the board in binary mode (replay.py + compare_fpga.py). Add a 300-step QAT tail only if E3a showed a fixed-point AUC drop above 0.03.",
   "rationale": "Window, threshold and refractory period decide recall, latency and false triggers live, and choosing them costs no training. The current host rules dilute a 4-9 frame ball burst over 50 frames and break with a 2-output model. Thresholds set on float outputs over-fire on the board.",
   "baseline": "The current host rule on the same spikes",
   "local_minutes": 15,
   "depends_on": "E9 (or E2 if nothing was adopted) and E3"
  }
 ],
 "warnings": [
  "[verified, still open in ball_cv v2] Line 183 sends half of the --synth-throw pastes into idle windows with a random anchor. A ball rising with no arm, labelled 'throw', teaches the model to fire on any rising blob, such as the head and hair impostors in the 04 idle take. Use feint-only pastes (E5).",
  "[verified, still open] Bank.idle_idx ping-pongs over each 120-frame idle clip from a random start in [0, 240). Nearly every idle training window, every overlay chunk and every clean-BN batch therefore contains a time reversal, and about half its frames play backwards with polarity not swapped. Eval idle is played forward. This is a non-physical cue that exists only in training.",
  "[verified] compare.py ranks by pooled auc_all and bootstraps it. In folds A/B/C, 465 of the 533 negatives are idle (fold A: 191 idle vs 32 feints), so auc_all is about 87% throw-vs-idle and mixes three fold models with different calibration. Rank by within-fold AUC_tf on 02/03 instead.",
  "[verified] jobs.txt runs every config at seed 0 only, so there is no noise estimate. a1_aux_paste changes two things relative to w1_win and is attributable only against w2_paste. b1_fix, p1_pop5 and w4_erase have the lowest expected value per GPU minute; move them to the end as stretch runs.",
  "[verified] Seeds need distinct tags. ball_cv writes runs_ball/<tag>/ (fold_*.pt, result.json), and queue.sh skips any tag already in done.txt.",
  "[verified, ball_cv mil_loss] The MIL soft max is not length-normalised. With tau=1, logsumexp adds up to about 4.7 over ~110 negative frames, against about 3 over 20 positive frames. This drives throw minus no-throw negative everywhere and can saturate the no-throw neuron. Normalise before writing MIL off.",
  "[verified] GPU memory: the single running job uses 7.3 of 8.2 GB. It logged an OOM retry for a 1.47 GB allocation, which is evaluate() gathering all ~239 test clips of fold A as float at once. Do not run two jobs in parallel, and chunk the gather before adding fold D.",
  "[verified; fixed in ball_cv v2 and make_bank v2, still present in Tim's robust_cv.py] The jitter/speed label bug (rels.append(a.pre) while the window moves by +-8 frames), the release-centred banks (class-correlated empty border frames), and the eval window that depended on the bank are all fixed. The new bank_mix.pt has peak 70 for all 181 swings and 'release' 61-87 for the 33 tracked throws. Results from robust_cv.py or from the deleted bank_new/bank_mix_700 runs are not comparable with v2 runs.",
  "[verified] Tim's prep_raw.prep_clips ignores release_ms (peak = release_us//10 = 70 for every swing). Its counts() bins from the first event: 69 of 288 idle clips start at least 10 ms late, up to 222 ms. make_bank v2 bins on the clip's own time axis. The mark_release.py docstring claim that Tim's prep_raw 'falls back' to release_ms is wrong.",
  "[verified mechanism] In Tim's ts+sym, sym equalises the throw and no-throw logits for t < 70, which is exactly while the 03 ball is visible (release frames 61-69). The ts CE starts at frame 75, when the 03 ball has mostly left the image (it reaches the top band at frames 70-78). Result for b0_tim, fold A: AUC_tf 0.59, rec@f10 0.19, rec_tracked 0.2.",
  "[verified, tracks.npz] 21 of 33 tracks hit the MAXBACK cap (release = top - 9 frames), so release_ms is the latest possible separation and earlier ball cells exist. Erasing only the tracked cells (w4_erase, cf_erase) leaves them in, so 'throw minus ball' negatives leak the label and cf_erase understates ball reliance. Tracks span 8-17 frames; 03 balls reach the top band at frames 70-78, 02 balls at 84-96.",
  "[verified] The win loss labels 'throw' on frames r+3..r+22, which is 10-15 frames past the point where the 03 ball is still visible. Many positive frames carry no ball evidence.",
  "[verified] Fold D (04/215305: 28 throws whose labels mean arm motion, 36 feints, labels from hint keys) is in the training set of every A/B/C run and is never tested by default. All 04 and 02 swing labels, and 29 of the 92 swing labels in 03, were verified by 'claude' (a model), not by a person. labels.csv marks 6 low-confidence 04 clips: t18628, t19812, t28972, t10301, t10513, t13730.",
  "[verified] The 255 scene-idle clips in bank_mix overlap no current throw label. They overlap 2 feint windows, 15 idle-labelled bursts (walking, ball pickup) and 1 invalid swing, so they are clean same-scene negatives. 'All idle comes from one take' is only true when training with --idle-train busy.",
  "[verified] In robust_cv, folds_of(group='session') ignores the seed, so --cv-seeds 0,1 duplicates identical test rows and aggregate.py's CIs are too narrow. robust_cv.summarize decides by argmax at the window end with ties scored 0.5, so a model that never fires scores 0.5.",
  "[verified] update_bn on augmented batches (robust_cv, and ball_cv with --bn-clean 0 as in b0_tim) calibrates the folded BN biases to thinned input. --freeze leaves the BN statistics trainable, and update_bn resets them anyway.",
  "[verified] Augmentation: --shift 8 in y and --scale 0.15 push top-band content out of view, so a ball 10-25 rows from the top leaves up to about 4 frames earlier. The v2 jobs use --shift-y 2; Tim's configs did not. Tim's thin_min 0.3 plus drop 0.1 can delete most ball cells while the label stays 'throw'.",
  "[reported by lenses, not re-verified] On the binary grid, the active-cell count peaks within +-100 ms of 700 ms in 97% of 03 throws but only 22% of feints: a timing/extent cue that works without the ball. The tracker fires on 24 of 36 04 feints and on 93 of 288 clips of the 04 idle take. PTQ adds 18% output spikes on a fine-tuned model; 0.5-1.4% of fused layer-0 weights clip at Q2.6; 15-18% of head weights round to 0; per-layer rescaling made fixed-point agreement worse.",
  "Statistics: 42 ball-visible throws and 65 feints give an AUC sampling SE of about 0.05. The reported run-to-run SD of balanced accuracy is 0.03-0.05, and about 14 earlier variants were compared on the same folds. Single-seed differences below about 0.05 are not evidence."
 ],
 "deployment_risks": [
  "Input format: despot sends float32 per-cell counts, but every model is trained on binary frames. On the board, the old 5-class model collapsed with counts (ref5_count output sums [0,0,0,0,143] vs ref5_binary [10,12,13,14,134]). With input scaling 4096, counts >= 8 exceed the Q4.12 range, and it is unknown whether they saturate or wrap. Binarise before SPLEAT (a despot setting or a small proxy) and confirm on the board with replay.py in binary vs count mode.",
  "Cadence: live1.npz has 46.9 results/s instead of 100, arriving in bursts (dt median 1.6 ms, p95 111 ms, 224 gaps > 50 ms in 35 s). If frames are dropped, motion looks about 2x faster than the 0.85-1.15 speed augmentation covers, and the 50-result host window spans about 1 s. If results are batched, about 100 ms of latency is added. Log frame ids, fix TCP_NODELAY and concurrency, and only then calibrate theta.",
  "Host logic: the sim's gesture_trigger.py assumes 3 classes (idle_class=2). With 2 outputs it raises IndexError. Padded with a dead channel 2, p_active is always 1: it arms on silence (total defaults to 1.0), never sees 'idle' again, so it triggers once per launch, and s_throw/total jumps to 1.0 on a single spike. gesture_node's 50-frame mean dilutes a 4-9 frame ball burst. Replace both with the E10 rule: an absolute threshold on a short window of throw minus no-throw, plus a refractory period.",
  "Robot link: tools/go2_reflex.py is keyboard-driven, and the path from the SNN to the dash exists only in the sim. The trigger must be bridged to arm/dash/disarm and tested end to end with replayed recordings before the demo.",
  "Firmware: a weights-only deploy needs the v1 arch with exactly 11 outputs (78501 values; a 2-output export gives 64092). export_tim.py and compare_fpga.py build the model from cfg['widths'] and write num_classes outputs, so they break on ball_cv checkpoints. Pad the head to 11 with zero weights and zero bias on the unused channels, keep 'outputs: channels: 11', and match the host channel indices. --pop M uses 2M channels averaged per class (M <= 5). The aux-ball head is training-only and must not be exported. Any non-v1 arch or --pool 2 needs a new firmware build (about 8 min, needs the user's OK).",
  "Fixed point: thresholds tuned on float outputs will over-fire on the board (reported +18% output spikes after PTQ on a fine-tuned model). Calibrate on the fixed-point sim, which matched the board within about 0.2% of output spikes, using the clean-BN weights. Add a short QAT tail if E3 shows an AUC drop above 0.03.",
  "Framing decides feasibility: in 04-like framing (head near the top edge) the ball is never visible after release, so release detection is impossible and only arm cues remain, which feints copy. Set up the demo like 03: thrower about 3 m away, with at least 15-20 free rows above the head on the 80x80 grid. More headroom keeps the ball in view longer than the current 4-9 frames.",
  "Latency: the trigger can only fire after a few ball frames plus the host window, so expect 50-150 ms after release plus network delay. The team's manual timing test assumed a reaction right at release; re-test the dodge with a 150-250 ms delay.",
  "Unknown inputs: all negatives come from two scenes (the 04 idle take and the 01-03 gaps), and the 02/04 labels were checked by a model. Record 20-30 min of venue idle (people walking, picking up the ball, robot standing) plus 10 throws and 10 feints in the demo framing, through the live path. Use it as the lockbox and the threshold calibration set. Zero triggers in 30 min only bounds the rate at about 0.1/min.",
  "Thresholds do not transfer from the CV fold models to the --final model. Set theta on the final model's fixed-point outputs (ideally on the venue recording), keep a refractory period so one swing gives one dodge, and plan for re-tuning on site.",
  "Two-stage reflex: the arm stage expects an idle or swing signal that the 2-output model does not have. Options: dash only; arm on a lower threshold of the same score (free, but test it on clips first); or add a swing output in a spare channel (weights-only, not in tonight's plan).",
  "Fallback: if the SNN stays far below the causal tracker (E1) on ball-visible sessions and the host can tap the despot frames, a host-side confirm (SNN trigger AND a tracker hit within about 100 ms) cuts feint false triggers, at the cost of the pure-SNN story."
 ]
}