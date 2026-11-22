# research / research:snn_readout (phase Research)

{
 "findings": [
  {
   "title": "Max-pooling (MIL) loss for streaming keyword spotting, plus a one-knob latency control (Sun et al. 2016; Jose et al., Amazon, Interspeech 2022)",
   "url": "https://arxiv.org/abs/2206.07261",
   "year": "2022 (max-pooling loss: 2016)",
   "what_they_did": "The wake-word detectors are trained with a max-pooling loss. For a positive clip, cross-entropy is applied only at the frame with the highest keyword posterior. For a negative clip, it is applied at the frame with the highest keyword posterior, which is the worst frame. No frame alignment is needed, and the training pushes down exactly the frame that would cause a false accept. Jose et al. show that this loss delays detections because the net waits for extra context. They fix it with one parameter: with probability b, the positive loss is taken at the frame just before the argmax, which moves posterior peaks earlier. No keyword endpoint labels are needed. For finer frame strides they suggest drawing the shift from a Poisson distribution.",
   "key_numbers": "About 20k h of training audio. CNN with 5 conv + 3 FC layers, 10 ms frame shift, 76-frame input. b in {0, .1, .2, .33, .5, 1}: b=0 gives the best DET curve but the highest latency, and b=0.33 matches HMM-aligned CE accuracy at lower latency. About 25% fewer relative false accepts at fixed latency than the endpoint-based 'max-latency' baseline. Sun et al.: best results when max-pool training starts from a CE-pretrained net.",
   "idea_for_us": "Both our live trigger and our eval take a max over time: max S over a feint or idle clip, and max S within 250 ms after release. The 'win' loss in ball_cv.py, however, averages CE over all masked frames, so the few frames that cause false triggers are diluted by many easy ones. Replace or mix it with a max-pool (soft-max or top-k) loss. Negatives are feint, idle and throw wind-up frames, and the loss penalizes the single most throw-like window. Positives reward the best window in [release+post_from, release+det_frames], with the 'one frame earlier' Bernoulli shift (b around 0.2-0.33, or a Poisson shift) to keep the reaction early. Initialise from the current CE-trained weights.",
   "how_to_test_tonight": "Make a copy of runs/ball/ball_cv.py and add --loss maxpool. Take z_t = throw logit minus no-throw logit from win_logits. Negative loss = softplus(tau*logsumexp(z_t/tau)) over the negative mask (tau 1-2, or the top-3 mean). Positive loss = softplus(-z_{t*}), where t* = argmax over the positive mask, replaced by t*-1 with probability b. Optionally keep 0.5x the old win loss. Run folds A,B,C for 1200 steps with the same seed, b in {0, 0.25}. Compare rec@f10, idle_per_min@f10, auc_all and the median first-crossing frame after release against the current win loss.",
   "confidence": "high (read the full text of Jose et al., which also gives the max-pool definition; the Sun et al. CE-init claim comes from its abstract)"
  },
  {
   "title": "End-of-keyword labelling with a masked loss and a threshold at a fixed false-alarm rate ('Hey Snips', Coucke et al., ICASSP 2019)",
   "url": "https://arxiv.org/abs/1811.07684",
   "year": "2019",
   "what_they_did": "A streaming wake-word detector built from 24 gated dilated conv layers (1.83 s receptive field). Only frames within ±160 ms of the keyword END get label 1. All other frames of positive utterances are masked out of the loss, and negative audio is 0. The model therefore waits until it has seen the whole keyword, never fires on partial keywords, and is robust to imprecise window edges. At inference, the per-frame posterior is averaged over 30 frames and the detector fires above a threshold set so that the negative audio gives 0.5 false alarms per hour.",
   "key_numbers": "Training set: 5,876 positive and 45,344 negative utterances (~2.2k speakers). FRR at 0.5 false alarms per hour: 0.12% clean vs 2.09% for a max-pooling LSTM and 2.51% for a CNN; 1.60% vs 11.2% at 5 dB SNR.",
   "idea_for_us": "Our 'win' loss already follows this recipe: positive frames just after release, wind-up negative up to release minus a margin, the rest masked. That is good evidence the target design is sound. What we can still take: (1) set the operating threshold from a false-alarm budget on long negative streams (idle triggers per minute plus feint triggers), not only from the 10%-feint quantile; (2) before the demo, record 2-3 minutes of idle at the venue and check or set the threshold on it. This is cheap and strengthens the robustness score.",
   "how_to_test_tonight": "From the existing result.json rows, compute per fold the threshold that gives ≤0.5 idle triggers per minute and ≤5% feint triggers. Report throw recall and first-crossing latency at that point. Compare smoothing windows W=5/10/20 (--eval-wins) at the same false-alarm budget.",
   "confidence": "high (read the full ar5iv text)"
  },
  {
   "title": "Micro-power keyword spotting on Xylo Audio 2: an SNN streaming detector trained with Rockpool's PeakLoss and deployed at 8-bit weights / 16-bit state (Bos & Muir, SynSense, 2024)",
   "url": "https://arxiv.org/abs/2406.15112",
   "year": "2024",
   "what_they_did": "A small LIF SNN (341-461 neurons) with one readout neuron detects a wake word on neuromorphic hardware. Training works on the readout membrane potential. For target clips, the loss is the MSE between a target g and the mean membrane over a window M right after the peak. For non-target clips, the whole membrane trace is pushed to 0, with a weight above 1. At test time any readout spike counts as a detection, and the threshold is swept for ROC curves. Deployment uses 8-bit weights and 16-bit state, with no special QAT stage described.",
   "key_numbers": "About 2,000 utterances from 96 speakers (3:1 non-target to target). M=140 ms, g=1.5, non-target weight 1.4. Float model: 98.4% (TPR 100%, FPR 3.1%). Deployed quantized model: 95.3% (TPR 91.7%, FPR 1.0%). 291 µW dynamic power.",
   "idea_for_us": "This is the closest SNN analogue to our task: a binary streaming detector trained on little data, at bit widths like our Q2.6 weights and Q4.12 potentials. Two takeaways. First, anchor the positive loss on the peak, so only the best moment after the event has to be high. Second, push down the whole negative trace with weight >1, because false triggers cost more than late hits. It is a variant of the max-pool loss above. Their deployment also suggests our fixed-point formats may be enough without heavy QAT, provided the fused weights fit the range.",
   "how_to_test_tonight": "Run it as a variant of the max-pool experiment. Positive loss = (g - mean of z_t over [t*, t*+M])^2, with t* the peak in the post-release window, M = 5-10 frames and g matched to the temp scale. Negative loss = 1.4·mean(relu(z_t)^2) over negative frames. Same folds and metrics.",
   "confidence": "medium-high (read the HTML full text through a summarizer; the PeakLoss form is as reported there)"
  },
  {
   "title": "Output population plus voting layer instead of one output neuron per class (Fang et al., PLIF, ICCV 2021)",
   "url": "https://arxiv.org/abs/2007.05785",
   "year": "2021",
   "what_they_did": "The last spiking layer has M·C neurons (M=10 per class), followed by average pooling with kernel M as a 'voting' layer. They used it on DVS128 Gesture, N-MNIST and CIFAR10-DVS, together with learnable membrane time constants.",
   "key_numbers": "M=10 neurons per class. DVS128 Gesture: about 96.5-97.6% (two table columns).",
   "idea_for_us": "Our conv head has only 2 LIF neurons, so each can spike only 0 or 1 per 10 ms frame. The evidence S(t) is very coarse, and we need W = 10-20 frames (100-200 ms) of summing to get a usable score, which costs latency. A head with 2·M output channels stays a single conv over the 5x5 map, so it is FPGA-exportable (M=8-10 means about 26-32k head weights). Summing per class on the host gives up to M spikes per class per frame. That graded per-frame evidence lets W shrink to 3-5 frames and makes the threshold and ROC smoother.",
   "how_to_test_tonight": "In a copy, call reset_head(2*M) (with M=8), sum the spikes over each class's M channels before win_logits and clip_scores, and rescale temp by 1/M. Train folds A,B,C and compare rec@f10, idle_per_min and median trigger delay at W=3/5/10 against the M=1 baseline.",
   "confidence": "medium (voting-layer details come from search snippets of the paper and its supplement, not a full read)"
  },
  {
   "title": "SPRT-TANDEM: sequential log-likelihood-ratio accumulation for early classification (Ebihara et al., ICLR 2021), plus neural CUSUM for change-point detection",
   "url": "https://arxiv.org/abs/2006.05587",
   "year": "2021 (NN-CUSUM: 2022)",
   "what_they_did": "A per-frame CNN plus a short-memory temporal integrator outputs posteriors. These are turned into a running log-likelihood ratio (the TANDEM formula, Markov order N), and the net decides as soon as the LLR crosses an upper or lower threshold (Wald's SPRT). Sweeping the threshold gives a speed-accuracy curve, and an LLLR loss trains the LLR estimate. A related paper, NN-CUSUM (Lee et al., arXiv 2210.17312), feeds neural likelihood-ratio estimates into the CUSUM recursion and analyses false-alarm run length vs detection delay.",
   "key_numbers": "UCF101 two-class task with only 1,026 training videos: 94.4% balanced accuracy at a mean of 2 frames, 97.3% at 49 frames. SiW: 99.8% at 1.19 frames.",
   "idea_for_us": "The ball release is a change point. Instead of a box sum over W frames, the host can run a CUSUM or SPRT-type accumulator over the per-frame evidence d_t = throw spikes minus no-throw spikes: g_t = max(0, g_{t-1} + d_t - k), trigger when g_t > h, then a refractory period. CUSUM is the classical rule for the shortest detection delay at a given false-alarm rate. It fires fast on strong evidence and needs longer on weak evidence, which is exactly the early-but-safe trade-off we need. It is a host-only change and needs no retraining.",
   "how_to_test_tonight": "In a copy of evaluate(), save the per-frame head spikes sp of all held-out clips to an npz. Offline, sweep (k, h), a leaky integrator with decay λ, and the current box sums W=10/20. For each setting compute feint FP rate, idle triggers per minute, throw recall and first-trigger frame after release, then plot latency against false-alarm rate.",
   "confidence": "high for SPRT-TANDEM (read the paper's main part); low-medium for NN-CUSUM (abstract and snippets only)"
  },
  {
   "title": "Early exit for SNNs from accumulated-output confidence with a Pareto-chosen threshold: Dynamic Confidence (Li, Jones, Furber, ICCV 2023) and SEENN (Li et al., NeurIPS 2023)",
   "url": "https://arxiv.org/abs/2303.10276",
   "year": "2023",
   "what_they_did": "SNN inference stops at the first timestep where the confidence decoded from the accumulated outputs passes a threshold. Confidence is the max softmax of the accumulated logits divided by a scale α; without the scale the softmax saturates early. The threshold comes from a cheap Pareto-front search on a validation split, keeping the lowest-latency threshold with no accuracy loss. SEENN-I uses the same confidence thresholding, and SEENN-II learns the exit timestep with RL.",
   "key_numbers": "26-58% fewer timesteps with no accuracy loss on CIFAR-10 and ImageNet (about 40% on average). Threshold grid at 0.1 resolution. SEENN-II ResNet-19 on CIFAR-10: 96.1% at an average of 1.08 timesteps.",
   "idea_for_us": "Treat the trigger as an early exit. Calibrate the scale of the windowed logit, then choose the threshold together with W (or CUSUM k,h) by a Pareto search over latency vs false triggers on the held-out folds, instead of one hand-picked number. The resulting Pareto front is also a clear 'robustness/control' figure for the jury.",
   "how_to_test_tonight": "Use the same per-frame spike dump as for CUSUM. Grid W in {3,5,10,20} by threshold, and for each point compute median trigger delay after release, feint FP rate and idle triggers per minute. Keep the Pareto set and pick the point that meets the false-alarm budget.",
   "confidence": "high for Dynamic Confidence (read the HTML text); medium for SEENN (abstract and snippets)"
  },
  {
   "title": "Per-timestep losses for early SNN decisions: EEvAct loss and readout ablation (Neumeier et al. 2025) and Temporal Efficient Training (TET, Deng et al., ICLR 2022, arXiv 2202.11946)",
   "url": "https://arxiv.org/abs/2507.07734",
   "year": "2025 / 2022",
   "what_they_did": "EEvAct compares training losses for early decisions on event streams with two-stream SNNs: CE on the time-averaged membrane, TET (CE at many timesteps), and their sum. It also compares mean-membrane and last-membrane readouts, and proposes a two-threshold 'early recognition neuron' that thresholds both the instantaneous membrane and its running mean. It includes a Prophesee tennis-motion dataset as a real-world trigger test. TET defines the loss as L = (1-λ)·mean_t CE(O_t, y) + λ·mean_t MSE(O_t, V_th) and argues that it reaches flatter minima.",
   "key_numbers": "EEvAct on THU EACT-50 (~10.5k recordings): top-1 at 0.3 s is 7.9% with CE-on-mean vs 23.7% with TET vs 25.2% with TET + last-membrane readout. TET + last-membrane falls at long horizons (66.9% vs 89.7% at 2.0 s). On SpikingBody (Prophesee, ~1,100 train and 400 test 1-s streams), top-3 is ≈80% within the first 50 ms. TET: λ=0.001 for DVS data; DVS-CIFAR10 77.3% vs 73.3% without augmentation and 83.2% with augmentation.",
   "idea_for_us": "Per-timestep losses buy early decisions at the cost of later stability. That fits our case, because only the ~250 ms after release matters and the head must stay quiet otherwise. Our win loss is already per-timestep, so two things remain to take. First, a two-threshold rule: fire on a strong short-window burst OR a moderate but sustained long-window mean. This matches the repo's two-stage reflex idea. Second, TET's tiny MSE term (λ≈1e-3) to keep the head neurons from dying or saturating when positive frames are rare.",
   "how_to_test_tonight": "Offline on the spike dump: test the rule S_W3 > h1 OR S_W10 > h2 against the best single-window rule at the same feint FP and idle triggers per minute. Optionally add λ·mean((z_t - c)^2) with λ=1e-3 to the loss on folds A,B,C (low priority).",
   "confidence": "high (read the full HTML of EEvAct and the ar5iv text of TET)"
  },
  {
   "title": "QAT for fixed-point SNNs: periodic cosine LR for quantized SNNs (Eshraghian et al., AICAS 2022), MINT shared weight/membrane scale (Yin et al., ASP-DAC 2024), and firing-distribution drift after quantization (Smith et al., arXiv 2604.14487, 2026)",
   "url": "https://arxiv.org/abs/2202.07221",
   "year": "2022-2026",
   "what_they_did": "Eshraghian et al.: quantized SNNs trained with straight-through estimators get stuck in local minima, and a periodic, restarting cosine-annealed learning rate helps (tested at 4 bits; code at github.com/jeshraghian/QSNNs). MINT quantizes both weights and membrane potentials with a shared scale, so inference needs no multipliers, with potentials down to 2 bits. Smith et al. show that quantizers with equal accuracy can still change the firing distributions a lot, and they measure the shift as the EMD between float and quantized firing distributions.",
   "key_numbers": "QSNN: 4-bit with a cosine-restart LR. MINT: membrane state grows from <20% to >60% of the memory as weight precision drops, and is quantized to 2 bits. Smith et al.: uniform quantization causes distribution drift even when accuracy holds.",
   "idea_for_us": "Our hardware fixes the formats: weights Q2.6 (±2, step 1/64) and potentials Q4.12 (±8, step 1/4096). The risks are that BN-fused weights above 2 get clipped, potentials saturate at ±8, and firing near threshold shifts. Any of these can move the trigger timing or raise false triggers while accuracy still looks fine. Check the trigger timing and the per-layer firing rates in float vs fixed point. If they drift, fine-tune the last ~20-30% of steps with fake-quant weights (rounded and clamped) and saturating potentials, using a restarting cosine LR.",
   "how_to_test_tonight": "On the final model: fuse BN, print max |w| per layer (it must stay below 2; otherwise rescale weights and threshold together), round to Q2.6 and rerun the held-out eval. Report the change in rec@f10, idle triggers per minute and first-trigger frame, plus per-layer spike-rate histograms for float vs quantized. If the changes are large, run riverops prepare_qat(fused_bn=True) for a few hundred steps.",
   "confidence": "medium (abstracts and snippets; none of the three read in full)"
  },
  {
   "title": "Transfer with little data: linear probe first, then fine-tune (LP-FT, Kumar et al., ICLR 2022), and DVS-Gesture-pretrained few-shot SNN learning on Loihi (SOEL, Stewart et al. 2020)",
   "url": "https://arxiv.org/abs/2202.10054",
   "year": "2022 / 2020",
   "what_they_did": "Kumar et al. show that fully fine-tuning a good pretrained model distorts its features and does worse out of distribution than training only the head. Training the head first and then fine-tuning everything (LP-FT) gets the best of both. Stewart et al. (arXiv 2008.01151) pretrain an SNN offline with SLAYER on DVS-Gesture-like data. Then only the output layer is learned on the Loihi chip, from a few shots of new gestures, with a local three-factor surrogate-gradient rule, including from a live DVS stream.",
   "key_numbers": "LP-FT vs full fine-tuning on 10 distribution-shift benchmarks: about +1% in-distribution and +10% out-of-distribution, with features changing 10-100x less. SOEL: few-shot learning on chip, comparable to off-chip simulation (exact numbers not checked).",
   "idea_for_us": "Our held-out folds are other sessions and people, so they are out of distribution. With 77 throws, first train the head with the DVS-Gesture backbone frozen (--freeze 4). Then unfreeze with a low backbone learning rate, keeping 1-2 early layers frozen throughout. This should generalize better to unseen people than fine-tuning everything from a random head.",
   "how_to_test_tonight": "Folds A,B,C, same seeds: (a) current full fine-tuning; (b) head only (--freeze 4); (c) LP-FT: 400 head-only steps, then 800 steps on all layers with lr 3e-4. (c) needs a small change so the phase-1 head is loaded (build_model drops head.* today), or a body-LR ramp from 0 after step 400. Compare per-fold auc_all, rec@f10 and idle triggers per minute, and look at the spread across folds.",
   "confidence": "medium-high for LP-FT (abstract and talk material); medium for SOEL (abstract and snippets)"
  },
  {
   "title": "EV-Catcher: low-latency catching of fast balls with Binary Event History Images and uncertainty-weighted predictions, trained on 235 real samples (Wang et al., RA-L 2022)",
   "url": "https://arxiv.org/abs/2304.07200",
   "year": "2022",
   "what_they_did": "Every 10 ms they build a binary event image: a pixel is 1 if it had any event in the window, with no counts. A small 4-layer stride-2 CNN predicts the ball's x position, its uncertainty σ and the time to collision. The predictions are fused with 1/σ² weights into a trajectory fit, and the actuator moves once when the predicted time to collision drops below 210 ms. They trained on real data only, because simulators did not match the sensor.",
   "key_numbers": "~371k parameters, 250x250 input, inference within 20 ms on a Jetson NX. 81% catches at up to 13 m/s. Without background motion, the binary input beat event volumes and grayscale (ball position error 7.8 vs 18.3 vs 32.6 mm). Removing the uncertainty weighting raised impact error from 19.0 to 30.1 mm. 235 training samples.",
   "idea_for_us": "(1) Binary inputs work well for small, fast balls and do not change with event rate. Our model trains on binarized cells (bank counts > 0), so the FPGA path must binarize the despot counts too (count > 0). Otherwise the first layer can see 16 or more per cell, its potentials get over-driven and false triggers follow. (2) A small set of real samples can be enough when the target is simple and the outputs are weighted by confidence.",
   "how_to_test_tonight": "Eval only, no training: run the current final model on held-out clips with three inputs: binary, raw counts (Bank.gather returns uint8 counts before the >0) and min(count, 2). This is a missing stress test in robust_cv.py. If raw counts change rec@f10 or the idle triggers, binarize on the Kria, or add count-clipping augmentation to training.",
   "confidence": "high (read the ar5iv full text)"
  },
  {
   "title": "SpikeMS: deep SNN for event-based motion segmentation with a spike-count loss and incremental predictions (Parameshwara et al., IROS 2021)",
   "url": "https://arxiv.org/abs/2105.06562",
   "year": "2021",
   "what_they_did": "An encoder-decoder SNN labels each event as an independently moving object or as ego-motion. The loss is BCE on per-pixel spike counts plus a van Rossum spike-distance term on foreground spikes (trained with SLAYER), on synthetic (MOD) and real (EV-IMO) data. Trained on 10 ms windows, it still gives useful outputs from shorter windows (1-9 ms), which they call incremental prediction.",
   "key_numbers": "A Kalman-filtered 3 ms prediction reaches ≈64% of the 10 ms accuracy with ~70% less latency. The authors claim ~50x lower power than a comparable ANN.",
   "idea_for_us": "Dense supervision of small moving objects works in SNNs, and we already have per-frame ball cells from the classical tracker (tracks.npz). Add a training-only auxiliary head on the 20x20 or 10x10 feature spikes that predicts a 'flying ball' map. The target is 1 at the tracker's ball cells and 0 everywhere in feints and idle, including the ball still in the hand. This pushes the backbone to encode the ball leaving the hand rather than the arm swing that feints share. The aux head is dropped at export, so the FPGA net is unchanged.",
   "how_to_test_tonight": "In a copy, run the backbone layers one by one and put a 1x1 conv on the layer-2 or layer-3 spikes, summed over 3 frames. Use BCE against the max-pooled track mask (pos_weight ≈ 20) with λ in {0.1, 0.3}. Only throw clips with tracks and pasted synthetic throws get nonzero masks. Compare auc_tf and rec@f10 on folds A,B,C.",
   "confidence": "medium-high (read the ar5iv text; the auxiliary-head transfer to our setup is our own idea)"
  },
  {
   "title": "Synthetic moving objects for event data: ShapeAug (2024) and sim-trained dodging of thrown objects (EVDodgeNet, ICRA 2020)",
   "url": "https://arxiv.org/abs/2401.02274",
   "year": "2024 / 2020",
   "what_they_did": "ShapeAug adds events from synthetic, randomly moving shapes into real event streams, keeping them consistent over time, and improves classification and Gen1 detection. EVDodgeNet (arXiv 1906.02919) trains shallow nets entirely in simulation to separate thrown objects from ego-motion in a quadrotor's event camera, and they transfer to the real world without fine-tuning. Domain context: Forrai et al. (ICRA 2023) catch objects with an event camera on an ANYmal quadruped.",
   "key_numbers": "ShapeAug: up to +5% on pedestrian detection on Gen1. EVDodgeNet: ~70% real-world dodge success, including unknown shapes and low light. Forrai: up to 15 m/s from 4 m, 83% success, 100 Hz on a Jetson Orin.",
   "idea_for_us": "The current --synth-throw pastes only the ~33 real tracker tracks. Add procedural tracks: a blob of 2-5 cells rising 1-2 cells per frame on a slight parabola, with random size, speed and density and polarity patterns copied from real tracks, pasted at the hand location. Also paste distractor blobs that move down, move sideways or do not start at a hand into feint and idle clips, labelled no-throw. The net then learns 'leaves the hand upwards' rather than 'any small blob'.",
   "how_to_test_tonight": "In a copy of Sampler.paste: draw 50% real and 50% procedural tracks, and give 20% of negatives a distractor blob. Compare rec@f10 and especially feint and idle triggers on folds A,B,C (fold D stays unsolvable because the ball is invisible there).",
   "confidence": "low-medium (ShapeAug and EVDodgeNet from abstracts and snippets only)"
  },
  {
   "title": "Time-to-first-spike training with exact gradients (Göltz et al., Nature Machine Intelligence 2021)",
   "url": "https://www.nature.com/articles/s42256-021-00388-x",
   "year": "2021",
   "what_they_did": "Each neuron encodes its output as the time of its first spike. They derive exact gradients of first-spike times in LIF networks, train hierarchical SNNs so the correct output neuron fires first, and emulate the method on BrainScaleS-2 with accuracy close to software.",
   "key_numbers": "Accuracy on BrainScaleS-2 is similar to software simulation (numbers not checked).",
   "idea_for_us": "In principle this is the most natural reflex code: decision = first output spike. But it needs an exact-gradient, event-based training framework that does not match riverops surrogate-gradient training tonight. The max-pool loss with the latency shift gives the same 'fire early, fire once' incentive inside our existing pipeline.",
   "how_to_test_tonight": "Not tonight: it needs a different training framework and output coding. Note it as background for the jury's originality question.",
   "confidence": "medium (abstract and snippets)"
  }
 ],
 "top_ideas": [
  "1. Train the trigger the way it fires. Replace or mix the mean-over-frames 'win' CE with a keyword-spotting max-pool (MIL) loss. It penalizes the single most throw-like window in feint, idle and wind-up frames, and rewards the best window within 250 ms after release. Add the Bernoulli 'one frame earlier' latency shift (b≈0.25), or a PeakLoss-style variant with non-target weight about 1.4, and start from the current CE weights. Our false triggers and our eval are both max-over-time, so this targets them directly. It is about 30 lines in a copy of ball_cv.py, to be tested on folds A,B,C against rec@f10, idle_per_min and trigger delay (Jose et al. 2022; Sun et al. 2016; Bos & Muir 2024).",
  "2. A host-side sequential decision rule calibrated to a false-alarm budget, with no retraining. Dump the per-frame head spikes of the held-out clips. Compare the box sum against a leaky or CUSUM accumulator (g = max(0, g + d_t - k) > h) and a two-threshold fast/slow rule, and pick the operating point by Pareto search over trigger delay, feint FP and idle triggers per minute (SPRT-TANDEM, Dynamic Confidence, EEvAct, Coucke). Before the demo, verify the threshold on 2-3 minutes of venue idle. In the same pass, run the cheap eval with binary vs raw-count input to confirm the Kria input must be binarized (count > 0) as in training (EV-Catcher BEHI).",
  "3. A population-coded output head. Use 2·M output channels (M=8-10) in the same single conv head and sum per class on the host (Fang et al. voting layer). Each 10 ms frame then carries graded evidence instead of 0/1 per class, so the readout window can shrink from 10-20 frames to 3-5. That lowers latency and smooths the threshold, and it stays FPGA-exportable.",
  "4. Teach the backbone the ball, not the arm. Add a training-only auxiliary 'flying-ball map' loss driven by the tracker's tracks.npz (SpikeMS-style spike-count BCE; target 0 for ball-in-hand, feints and idle). Add procedural ball tracks and distractor blobs to the paste augmentation (ShapeAug/EVDodgeNet spirit), and use an LP-FT schedule (head first, then a gentle unfreeze) to generalize to unseen sessions and people. Finish with a fixed-point check: fused weights within Q2.6 range, and float vs quantized trigger-time and firing-rate drift, with a short riverops QAT plus cosine-restart LR fine-tune if needed."
 ]
}