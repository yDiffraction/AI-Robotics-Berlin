# research / research:augment_synth (phase Research)

{
 "findings": [
  {
   "title": "ShapeAug / ShapeAug++: procedural moving shapes rendered as events (Bendig, Schuster, Stricker)",
   "url": "https://arxiv.org/abs/2401.02274",
   "year": "2024",
   "what_they_did": "They draw 1 to 5 random circles, rectangles or ellipses (size 3 px up to s_max) that move in straight lines at a random speed and angle. Events are the sign of the difference between consecutive rendered frames, written into the ON/OFF channels of a (T,2,H,W) histogram, and 20% of the synthetic events are dropped. The shapes are added on top of real samples, and real events behind a shape are removed (occlusion). They tested a spiking PreAct-ResNet34 with PLIF neurons on datasets resized to 80x80, and also Gen1 detection. ShapeAug++ (arXiv 2409.11075) adds polygons, curved paths and rotation, reported up to +3.7% top-1 over ShapeAug.",
   "key_numbers": "Top-1, geometric only vs geometric+ShapeAug: N-Caltech101 62.2->68.7, DVS-Gesture 89.8->91.7, DVS-CIFAR10 73.8->75.7, N-CARS 97.1->97.3. Gen1 mAP 29.26->29.60 (pedestrian AP 17.37->18.30). Shapes of 100-150 px made results worse.",
   "idea_for_us": "Use the same recipe for the one object we care about: a procedural 'ball after release' generator in our exact format (80x80, 10 ms, 2 polarities, binary). Ball radius about 1-3 cells. It starts at the hand centroid at a sampled release frame, rises 0.8-2.5 cells per frame with a small sideways drift, and overlaps the hand blob for its first 1-3 frames. The leading edge gets one polarity and the trailing edge the other (random sign), and each cell is kept with probability 0.4-1.0. Paste it into feint/idle windows as positives. The current --synth-throw only reuses about 20-30 real tracks per training fold, so the net can memorise them. A generator gives unlimited, exactly labelled variation in size, speed, contrast and distance, which also covers jury 'unseen inputs' such as a different ball or distance. A second, label-preserving use in the ShapeAug style: add random clutter shapes to all classes.",
   "how_to_test_tonight": "Add a 'proc' kind next to 'paste' in ball_cv.Sampler.batch (about 40 lines). Render on a 4x finer grid and OR-pool down to 80x80 so motion can be sub-cell. Run ball_cv.py --loss win --test-folds A,B,C with 2 seeds and three arms: (a) the current real-track paste with --synth-throw 0.5, (b) procedural balls only, (c) 50/50. Compare recall at 10% feint false-trigger rate and idle triggers/min, keeping evaluation purely real. Before that, check realism with debug_synth.py renders and a tiny real-vs-synthetic track classifier: an AUC near 0.5 means the synthetic balls look real.",
   "confidence": "high (read the paper HTML)"
  },
  {
   "title": "EVDodgeNet: dodging thrown objects with shallow event networks trained only in simulation (Sanket et al.)",
   "url": "https://arxiv.org/abs/1906.02919",
   "year": "2020 (ICRA)",
   "what_they_did": "A quadrotor with one event camera runs shallow networks for deblurring, ego-motion (homography) and segmentation/flow of independently moving objects. All of them were trained on a simulated Moving Object Dataset and deployed on the real drone without fine-tuning, while balls, toy cars and planes were thrown at it.",
   "key_numbers": "70% overall dodging success: 86% for a spherical ball, 76% for unknown shapes, 70% in low light. Perception lag 60 ms: about 10 ms of network time per frame pair plus 2 ms post-processing, and the object must be detected over 5 consecutive frames.",
   "idea_for_us": "This is evidence that a small event network keyed on 'independently moving small object' can transfer from synthetic to real data. In our case the hard parts (people, wind-ups, background) stay real and only the ball is synthetic. Proposed ablation, which also works as a jury story: train with no real throw windows as positives (only real-track or synthetic balls pasted into real feint/idle windows) and test on real throws. If recall holds, the SNN demonstrably keys on the ball and not the arm, which is exactly the robustness argument against feints.",
   "how_to_test_tonight": "--synth-throw 1.0 already makes every throw slot a pasted track in a feint/idle window, so no real throw window is used as a positive. Run it on folds A,B,C with 2 seeds and compare recall@10% feint FPR to the normal run. Repeat with procedural balls only once the generator exists. A small drop means the model is free of the shortcut; a large drop means it was relying on body cues.",
   "confidence": "medium (read the ar5iv text; how the simulated dataset was generated is in a supplement I did not read)"
  },
  {
   "title": "Sim-to-real for event data: randomise contrast widely and add noise (Stoffregen et al., Reducing the Sim-to-Real Gap; E2VID)",
   "url": "https://arxiv.org/abs/2003.09078",
   "year": "2020 (ECCV)",
   "what_they_did": "They studied why networks trained on simulated events degrade on some real datasets. E2VID (Rebecq et al. 2019) was trained purely on ESIM synthetic events and still worked on real cameras. The main factor is the simulator's contrast threshold, i.e. the event rate per pixel. Training over a wide random threshold range (0.1-1.5), plus Gaussian noise, hot pixels and a refractory period, gave the best or near-best results on every dataset.",
   "key_numbers": "Retraining with their data recipe: +20-40% video reconstruction quality and +10-16% optic flow. Noise augmentation alone gave up to about 10%.",
   "idea_for_us": "For synthetic balls, the counterpart of the contrast threshold is the ball's event density per cell and its edge thickness, which depend on ball-vs-background contrast, distance and lighting. Do not tune one 'nice' ball. Randomise density (thinning), blob size and polarity balance over a range that covers the real tracks (8-25 cells, 1-2 cells per 10 ms), and add hot pixels and isolated noise so the net cannot detect synthetic crispness.",
   "how_to_test_tonight": "From tracks.npz, compute per-frame cell counts, polarity ratio, rise speed and frames to the top edge. Set the generator ranges to cover the 5th-95th percentiles. Then A/B a narrow versus a wide randomisation inside the procedural-paste run from the ShapeAug finding.",
   "confidence": "medium-high (read Stoffregen on ar5iv; the E2VID claim comes from a search snippet)"
  },
  {
   "title": "Copy-paste for small or rare objects: artifacts, context and distractors (Dwibedi ICCV 2017; Dvornik ECCV 2018; Kisantal 2019; Ghiasi CVPR 2021)",
   "url": "https://arxiv.org/abs/1708.01642",
   "year": "2017-2021",
   "what_they_did": "Dwibedi: cut object instances out and paste them onto backgrounds. Subtle paste-boundary artifacts let the detector cheat; training on several blending variants of the same paste makes it invariant to them, and 10% real data plus synthetic matched 100% real. Dvornik (https://arxiv.org/abs/1807.07428): random placement hurts detection unless a context model chooses plausible locations; gains with few labels on VOC'12. Kisantal (https://arxiv.org/abs/1902.07296): oversample images that contain small objects and paste the small objects several times. Ghiasi (https://arxiv.org/abs/2012.07177): simple random copy-paste is strong for rare classes and adds on top of pseudo-label self-training.",
   "key_numbers": "Dwibedi: +8 AP from blending. Kisantal: +7.1% (detection) and +9.7% (segmentation) relative on small COCO objects. Ghiasi: +6.1 mask AP on rare LVIS classes.",
   "idea_for_us": "Our paste/erase already follows the context rule (the ball is placed above the hand). What is missing is artifact balancing: a pasted ball only ever appears in positives and an erased hole only ever in negatives, so any synthesis artifact predicts the label. Fixes: (a) punch the same kind of holes along random fake paths in all classes; (b) paste decoy tracks into negatives that break the throw rule, such as tracks that start far from any hand, move down or sideways, last only 2-3 frames, or appear mid-air; (c) vary how pasted tracks are blended (thin to 30-100%, jitter by 1 frame, OR versus overwrite). This is a cheap way to make 'a small blob that leaves the hand and keeps rising' the only signal that carries the label.",
   "how_to_test_tonight": "Add --hole-share and --decoy-share to the Sampler, with decoys labelled no-throw. Run folds A,B,C with 2 seeds against the current --synth-throw/--erase-share setting. Besides recall@10% FPR, add two probes on held-out data: feints with decoys pasted in, and throws with the ball erased, must both stay below the threshold. Caveat: a time-reversed (falling) decoy also suppresses a late trigger on the incoming ball, so use it only if you want release-only triggering.",
   "confidence": "medium-high (abstracts, poster and snippets; full texts not read)"
  },
  {
   "title": "Gradient supervision on counterfactual pairs (Teney et al. ECCV 2020) and Right for the Right Reasons (Ross et al. IJCAI 2017)",
   "url": "https://arxiv.org/abs/2004.09034",
   "year": "2020",
   "what_they_did": "They take pairs of inputs that differ minimally but carry different labels. An auxiliary loss, 1 - cos(grad_x f(x_i), x_j - x_i), makes the input gradient of the decision point along the pair difference, i.e. toward what actually changes the label. Ross et al. (https://arxiv.org/abs/1703.03717) instead penalise input-gradient mass on regions marked as irrelevant.",
   "key_numbers": "Out-of-distribution gains: IMDb edited test 88.7->91.2, SNLI edited 57.8->61.2, COCO edited images 64.0->65.2 mAP. Random, non-counterfactual pairs did worse than the baseline. Costs about 2 extra backward passes; the loss weight ranged from 0.01 to 20 depending on the task.",
   "idea_for_us": "We get perfect counterfactual pairs for free: a throw clip x and the same clip with its ball track erased, x' (labelled no-throw). x - x' is exactly the ball's events. Add the gradient-supervision term on the throw readout over the post-release window, so the SNN's decision depends on the ball cells and not on the wind-up or follow-through (the feint shortcut). Variant in the style of Ross et al.: penalise gradient outside the dilated ball mask in post-release frames.",
   "how_to_test_tonight": "Only after the copy-paste and heatmap ideas. It needs a double backward pass through the surrogate gradients (create_graph=True); keep it cheap with T of about 40 frames around the release and batch size 8 on the 8 GB GPU. Compare --erase-share 0.3 with and without the term (weight 0.1 and 1). Main metric: feint false-trigger rate at fixed recall. Probe: erased throws should score like feints.",
   "confidence": "high for the method (read on ar5iv); medium that it works with LIF surrogate gradients"
  },
  {
   "title": "Tiny event networks that localise balls from automatically labelled throws: Ziegler et al. 2024 (SNN on neuromorphic hardware) and EV-Catcher (RA-L 2022)",
   "url": "https://arxiv.org/abs/2403.10677",
   "year": "2024 / 2022",
   "what_they_did": "Ziegler: a 4-layer SNN on 64x64 binary 11 ms event frames (a pixel is 1 if it saw any event; polarity dropped) outputs the ball's x and y as 64-bin population codes trained with soft targets. It used 8.6k training samples, 7.5k of them labelled automatically by triangulating frame cameras, and ran on Akida, DynapCNN and Loihi 2 inside a table-tennis robot loop. EV-Catcher (https://arxiv.org/abs/2304.07200): binary event-history images every 10 ms, 4 conv layers (64 channels, stride 2) plus a linear layer, 371k parameters. It was trained only on real data: 235 data points labelled with motion capture, with trajectories filled in by a polynomial fit, and no simulator. Its only augmentation merged flying-ball events with recorded background motion of people. 1000 Rallies (arXiv 2606.25620, June 2026) similarly trains a ball position and velocity CNN, robust to player motion, on pseudo ground truth from 14 high-speed cameras.",
   "key_numbers": "Ziegler: 1.44-1.59 px error on the neuromorphic chips vs 3.55 px for a GPU CNN baseline, 0.6-0.9 ms inference, 15 of 15 balls returned online. EV-Catcher: 81% catch rate at up to 13 m/s, 19 +/- 15 mm impact error, on a Jetson NX.",
   "idea_for_us": "Distil our classical tracker into the SNN through an auxiliary conv-only head: a per-frame ball heatmap at 20x20 or 10x10 (a 1x1 conv on an existing layer). Targets come from tracks.npz for real throws and from the exact masks of pasted or synthetic balls, and are zero for feints and idle. Each throw then gives 6-15 supervised frames of 'where is the ball' instead of one clip label, which is the regime where these tiny networks worked with a few hundred throws. Trigger rule: throw readout AND a heatmap peak that rises over 2-3 frames. This is interpretable and can be shown live to the jury. The head can be dropped at export if RIVER output wiring is a problem.",
   "how_to_test_tonight": "Add the head plus BCE (pos_weight about 20, loss weight 0.5) to the ball_cv.py loss, and rasterise tracks.npz onto the head's grid. Compare folds A,B,C with and without the head on the same seeds. Also report how often the heatmap peak lands within 2 cells of the tracker on held-out throws. If the heatmap works, try the combined trigger rule as well.",
   "confidence": "high for Ziegler and EV-Catcher (read on arXiv HTML and ar5iv); 1000 Rallies from the abstract only"
  },
  {
   "title": "Outlier Exposure: train against an extra pool of 'everything else' (Hendrycks et al. ICLR 2019)",
   "url": "https://arxiv.org/abs/1812.04606",
   "year": "2019",
   "what_they_did": "An auxiliary dataset of outliers is added during training so the model learns to give low confidence to inputs unlike its in-distribution data. This clearly improved anomaly and out-of-distribution detection in vision and NLP. Follow-up work (Liznerski et al., TMLR 2022) finds it helps less when anomalies are very subtle.",
   "key_numbers": "Large out-of-distribution detection gains reported, but no single number transfers to our setting; caveat for subtle anomalies.",
   "idea_for_us": "For a trigger, 'throw' is the in-distribution class and every other human motion is an outlier. DVS-Gesture (arm rolls, claps, air guitar, waving; 29 people) is exactly the family of false triggers a jury will try, and the team already has a converter to our 80x80, 10 ms format (runs/own_throw/pretrain/prep_dvsgesture.py). Use those clips as extra no-throw windows, and paste real or synthetic balls into some of them as positives so arm-waving no longer predicts the label. This also gives an honest 'unseen inputs' metric: triggers per minute on held-out DVS-Gesture users.",
   "how_to_test_tonight": "The dvsg_*.npz files are not on this laptop because pretraining ran on the Intel Arc machine; copy data/dvsg_train.npz and dvsg_test.npz over. Add them as a third negative pool (about 20% of the no-throw slots). A/B on folds A,B,C and report idle triggers/min and triggers/min on dvsg_test at the threshold that gives 10% feint FPR.",
   "confidence": "medium (Outlier Exposure from the abstract; using DVS-Gesture this way is our own inference)"
  },
  {
   "title": "Event-specific augmentation family: NDA (ECCV 2022), EventDrop (IJCAI 2021), EventMix (2023), EventRPG (ICLR 2024), polarity flip",
   "url": "https://arxiv.org/abs/2203.06145",
   "year": "2021-2024",
   "what_they_did": "NDA: geometric augmentations applied consistently over time, for SNNs on small DVS datasets; also the first contrastive pretraining for SNNs. EventDrop (https://arxiv.org/abs/2106.05836): drop events at random, within a time window, or within an area. EventMix (https://arxiv.org/abs/2205.12054): mix two streams with 3D masks shaped by a Gaussian mixture model. EventRPG (https://arxiv.org/abs/2403.09274): saliency maps from spiking relevance propagation decide where to drop or mix. MEM trains with a random polarity flip at probability 0.5.",
   "key_numbers": "NDA: +10.1% on CIFAR10-DVS, +13.7% on N-Caltech101. EventRPG: 85.62% N-Caltech101, 85.55% CIFAR10-DVS, 91.59% SL-Animals.",
   "idea_for_us": "The harness already has geometric, thinning, noise, overlay, swap and mix, so expect only small extra gains from this family. Two real gaps remain. (1) Polarity swap at p=0.5 makes the net invariant to whether the ball is brighter or darker than the background; a different ball or wall at the jury venue flips which edge produces ON events. (2) Drop-by-time: blank 1-3 whole frames (despot or USB hiccups) so the per-frame trigger survives frame loss. EventRPG can also be used the other way round: SNN relevance maps as a debugging check that throw decisions sit on the ball cells.",
   "how_to_test_tonight": "Add one flag each in Sampler.batch: x = x[:, :, [1,0]] with p=0.5, and zero out 1-3 random frames. A/B on folds A,B,C. Stress test without retraining: evaluate the current model on polarity-swapped held-out clips; a large drop shows it depends on the contrast sign.",
   "confidence": "medium (mostly abstract-level; the polarity flip in MEM comes from a snippet)"
  },
  {
   "title": "Self-supervised SNN pretraining on unlabelled events (SpikeCLR 2026, MEM WACV 2024, Hagenaars et al. NeurIPS 2021)",
   "url": "https://arxiv.org/abs/2603.16338",
   "year": "2021-2026",
   "what_they_did": "SpikeCLR (March 2026): contrastive pretraining of SNNs with spatial, temporal and polarity augmentations, then fine-tuning. It beats supervised training in few-shot and semi-supervised settings on CIFAR10-DVS, N-Caltech101, N-MNIST and DVS-Gesture, and using spatial and temporal augmentations together was critical. MEM (https://arxiv.org/abs/2212.10368): masked event modelling on unlabelled events beats supervised RGB pretraining. Hagenaars (https://arxiv.org/abs/2106.01862): an SNN learns optical flow self-supervised from raw events.",
   "key_numbers": "Only abstracts read; no numbers.",
   "idea_for_us": "Pretrain the conv-only SNN on all our own unlabelled GenX320 recordings (same sensor, same despot format) instead of only on DVS-Gesture (DVS128, different noise), using a contrastive or next-frame-prediction objective.",
   "how_to_test_tonight": "Not tonight: it needs hours of GPU time plus tuning, and with about 77 throws the gain over the existing DVS-Gesture initialisation is uncertain. Mention it as future work.",
   "confidence": "medium (abstracts)"
  },
  {
   "title": "v2e: video frames to realistic DVS events (Hu, Liu, Delbruck, CVPRW 2021)",
   "url": "https://arxiv.org/abs/2006.07722",
   "year": "2021",
   "what_they_did": "A toolbox that turns intensity video into DVS events with a realistic pixel model: temporal noise, leak events, intensity-dependent bandwidth and threshold mismatch. Pretraining on v2e data under varied lighting improved transfer to real DVS data, and a night-driving car detector trained on v2e events beat a frame-trained YOLOv3 by about 40%.",
   "key_numbers": "+40% average night-driving detection accuracy vs YOLOv3 trained on frames.",
   "idea_for_us": "A source of extra throws and feints from phone slow-motion (240 fps) videos of other people and places, or from the repo's Gazebo virtual-thrower scene, to diversify bodies and backgrounds.",
   "how_to_test_tonight": "Not tonight. Ten minutes with the real GenX320 gives 50+ real throws and feints with no domain gap, which beats v2e conversion. Use v2e only to emulate a venue you cannot record in.",
   "confidence": "medium-high"
  }
 ],
 "top_ideas": [
  "1. Make the ball the only signal that carries the label. Build a procedural ball generator in the style of ShapeAug, in our 80x80/10 ms binary format, with wide randomisation of size, density and polarity as Stoffregen et al. recommend. Combine it with artifact-balanced paste/erase (holes in all classes; decoy tracks that start away from a hand, fall, move sideways or are too short, labelled no-throw), following Dwibedi and Dvornik. This builds directly on the existing --synth-throw/--erase-share code: about 1 h of coding, then 2 seeds x folds A,B,C. Include the EVDodgeNet-style ablation '--synth-throw 1.0, no real throw windows as positives' as a shortcut test. If recall holds, it is also a strong jury slide: the SNN keys on the ball, not the arm.",
  "2. Distil the classical tracker into the SNN with an auxiliary conv-only ball-heatmap head (20x20 or 10x10). Targets come from tracks.npz plus the exact masks of pasted or synthetic balls. This gives 6-15 supervised frames per throw instead of one clip label, the regime where Ziegler et al. 2024 (SNN on neuromorphic hardware) and EV-Catcher (235 throws) worked. Trigger = throw readout AND a heatmap peak rising over 2-3 frames: interpretable, and it can be shown live.",
  "3. Robustness to unseen inputs: Outlier Exposure with DVS-Gesture as extra no-throw windows. Copy dvsg_train/test.npz from the Arc laptop; they are already in board format via prep_dvsgesture.py. Paste balls into some of those windows, and add polarity-swap (p=0.5) plus 1-3-frame drop augmentation. Report triggers/min on held-out DVS-Gesture users and accuracy on polarity-swapped clips as the 'unseen inputs' metric for the robustness score (25 points).",
  "4. If time remains: gradient supervision (Teney et al. 2020) on the free counterfactual pairs (throw vs the same throw with the ball erased), so the input gradient of the throw readout points at the ball cells. It needs a double backward pass, so use T of about 40 and a small batch. Note: none of my searches found prior work on event-camera detection of a throw's release moment, which supports the originality score (20 points)."
 ]
}