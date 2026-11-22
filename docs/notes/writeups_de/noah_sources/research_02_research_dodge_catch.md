# research / research:dodge_catch (phase Research)

{
 "findings": [
  {
   "title": "Dynamic obstacle avoidance for quadrotors with event cameras (Falanga, Kleber, Scaramuzza, Science Robotics)",
   "url": "https://doi.org/10.1126/scirobotics.aaz9712",
   "year": "2020",
   "what_they_did": "A quadrotor dodges balls thrown at it. Learning-free pipeline: IMU-based rotational ego-motion compensation, then a normalized mean-timestamp image (per pixel, how recent its events are), a threshold to split dynamic from static pixels, clustering into obstacles, and a fast repulsive-field avoidance command. Ran on an Insightness SEEM1 camera and an Intel UP board.",
   "key_numbers": "3.5 ms overall perception latency. The press release says over 90% avoidance of a ball thrown from 3 m at about 10 m/s. No training data.",
   "idea_for_us": "Our camera is static before the dodge, so ego-motion compensation is not needed and every event comes from something moving. A recency or mean-timestamp map over the last 3-5 frames tells freshly moving small blobs (the ball) apart from body parts that have been moving for a while. A classical detector costs almost no latency and needs no data.",
   "how_to_test_tonight": "CPU, about 20 min: from the 10 ms frames build a per-cell recency map (frame index of last activity) and a per-blob newness score (share of the blob's cells inactive 2 frames earlier). Add it as a gate or feature to the blob-chain tracker and count tracked throws (of 77) and hits on the 104 feints and on idle clips.",
   "confidence": "medium: the 3.5 ms figure (abstract) and the >90% / 3 m / 10 m/s figures (UZH press release) were seen via search, not in the paper; pipeline details confirmed only indirectly through Zhu 2023 and Forrai 2023, which reuse and describe it. Full paper not opened (PDF link 404)."
  },
  {
   "title": "Event-based Agile Object Catching with a Quadrupedal Robot (Forrai, Miki, Gehrig, Hutter, Scaramuzza, ICRA)",
   "url": "https://arxiv.org/abs/2303.17479",
   "year": "2023",
   "what_they_did": "ANYmal catches thrown balls with a net. Detection is classical, with no CNN: an IMU rotation-compensated mean-timestamp image, an adaptive threshold that rises with angular rate, and DBSCAN over position, optical-flow similarity and timestamp. Depth comes from the known ball size. A RANSAC fit of a ballistic parabola with fixed gravity rejects outlier points, and a median filter over successive impact estimates runs before the catch is triggered. Control is a PPO policy trained in Isaac Gym.",
   "key_numbers": "VGA DVXplorer, about 100 Hz on a Jetson Orin, 83% catch success up to 15 m/s from about 4 m, 22 real throws evaluated, no training data for detection.",
   "idea_for_us": "Use physics as the false-trigger filter. A released ball keeps rising at a near-constant velocity over the 40-80 ms we see it, while a hand in a feint slows down and reverses. A RANSAC constant-velocity or parabola fit plus a median over successive estimates is a cheap, explainable separator.",
   "how_to_test_tonight": "CPU, about 30 min: make the blob chain from ball_tracks.py causal (drop the condition that the blob reaches the top band). Fit vy over the last 4-6 centroids with RANSAC and fire when there are at least 4 inliers, the residual is under 1 cell and vy is 0.8-2.5 cells per frame upward. Report recall on the 77 throws, frames after release, hits on the 104 feints, and idle triggers per minute.",
   "confidence": "high: full text read (ar5iv)"
  },
  {
   "title": "Dynamic Object Avoidance using Event-Data for a Quadruped Robot (Zhu, Perera, Yu, Hwang, Kim, IROS workshop)",
   "url": "https://ippc-iros23.github.io/papers/zhu.pdf",
   "year": "2023",
   "what_they_did": "A Mini-Cheetah dodges thrown or rolled balls. With the robot standing still, only moving objects produce events. They use a Falanga-style normalized time image with a dual threshold: a high threshold for seed regions, then a lower threshold inside a +-10 px window around each seed. This keeps the object complete without adding noise. Two detections are enough for a collision check and five give the velocity. The robot dodges by changing its body height.",
   "key_numbers": "24 trials, 83% success (11/13 aerial, 9/11 ground), balls up to 5 m/s launched from 8 m. Event plus depth camera. No learning.",
   "idea_for_us": "Hysteresis thresholding recovers faint, sparse balls that a single threshold loses, which may matter for weak sessions. Commit early: a threat decision after about 2 detections. A single fast posture change is enough for a quadruped dodge, so use the fastest Go2 sport-API motion.",
   "how_to_test_tonight": "CPU, about 15 min: in the blob step, take as seeds the cells active in 2 consecutive frames or in both polarities, then grow into any active cell within +-2 cells. Rerun the release-chain logic and check whether more than the current 33 throws get tracked while feint hits stay at 0.",
   "confidence": "high: full PDF read"
  },
  {
   "title": "Robotic goalie with 3 ms reaction time at 4% CPU load using event-based dynamic vision sensor (Delbruck and Lang, Frontiers in Neuroscience)",
   "url": "https://www.frontiersin.org/articles/10.3389/fnins.2013.00223/full",
   "year": "2013",
   "what_they_did": "A 128x128 DVS with an event-driven cluster tracker of about 20 clusters. A cluster becomes visible after about 10 events and a minimum speed, is pruned after about 10 ms without support, and overlapping clusters are merged. Velocity comes from a rolling least-squares fit over 10-30 ms. Only the cluster predicted to reach the goal next is treated as a threat, so balls rolling away and the shooter's hand are ignored. After a few seconds without a definite ball the goalie goes into a sleep state and stops reacting to every movement.",
   "key_numbers": "2.2 +- 2 ms from ball movement to motor command, 550 Hz median update rate, under 4% CPU, about 80% of shots from 1 m blocked. No learning for detection.",
   "idea_for_us": "A sensor resolution close to our 80x80 grid was enough. Borrow the false-trigger controls: visibility needs support plus speed, prune on missing support, select threats by direction, and use an armed/sleep state with a refractory period. For idle scenes, arm the ball trigger only after wind-up-level activity (or SNN context spikes) in the last ~1 s, and hold it 2 s after firing.",
   "how_to_test_tonight": "CPU: add the armed state and refractory period to the causal tracker, plus a rule that ignores blobs connected to the large arm/body component. Measure idle triggers per minute on the idle recordings with and without arming, and check that throw recall does not drop.",
   "confidence": "high: full text read (PMC)"
  },
  {
   "title": "EV-Catcher: High-Speed Object Catching Using Low-latency Event-based Neural Networks (Wang et al., RA-L)",
   "url": "https://arxiv.org/abs/2304.07200",
   "year": "2022",
   "what_they_did": "Binary Event History Image (BEHI): a pixel is 1 if it had any event since the start, updated every 10 ms, with about 120 ms collected before the decision, so the ball's path shows up as a streak. A small CNN (4 stride-2 3x3 conv layers with 64 channels, average pooling, one linear layer) regresses ball x, its uncertainty sigma and time to collision. Predictions are combined in an uncertainty-weighted least-squares trajectory fit. A linear actuator catches ping-pong balls.",
   "key_numbers": "235 real samples with mocap labels, 20 held-out trajectories, 250x250 input, 371k parameters, 12 BEHIs inferred within about 20 ms, Jetson NX. 81% catch success at up to 13 m/s, 1.9 cm impact error. BEHI beat a 10-bin event volume (location 7.8 vs 18.3 mm, TTC 9.0 vs 39.4 ms). Uncertainty weighting cut impact error from 30.1 to 19.0 mm.",
   "idea_for_us": "A binary history channel lets a tiny conv net read the whole ball streak, detached above the hand, in a single time step, and it worked with about 200 real samples. To keep our FPGA input at 2x80x80, drop polarity (Ziegler 2024 also did): channel 0 = OR of both polarities at t, channel 1 = OR of the last ~8 frames.",
   "how_to_test_tonight": "GPU, about 1 h, for the parent: in a scratch copy of the ball_cv.py input prep, build the [current, history OR(t-7..t)] channels and rerun the w1_win / w2_win_paste configs on folds A,B,C. Compare AUC, recall at 10% feint false-trigger rate, and idle triggers per minute. Only the frame prep on the Kria side would change.",
   "confidence": "high: full text read (ar5iv); the paper does not say which hardware the 20 ms inference figure was measured on"
  },
  {
   "title": "Detection of Fast-Moving Objects with Neuromorphic Hardware (Ziegler, Vetter, Gossard, Tebbe, Otte, Zell)",
   "url": "https://arxiv.org/abs/2403.10677",
   "year": "2024",
   "what_they_did": "An SNN detects a table-tennis ball in binary, polarity-free 11 ms event frames (a 64x64 crop around the last ball position). Layers: conv 4ch 5x5 stride 2, conv 4ch 3x3, FC 64, then FC 128 split into an x population and a y population of 64 bins each. Trained with MSE on soft targets (1 for the true bin, 0.5 for its neighbours) because cross-entropy gave large activations and more spikes. Deployed on Akida, DynapCNN and Loihi 2.",
   "key_numbers": "8,630 training frames (7,569 auto-labelled by projecting 3D positions from frame cameras, 1,061 manual). Error about 1.4-1.6 px. Chip inference 0.6-0.9 ms, Akida about 4.5 mW. Returned 15 of 15 balls online.",
   "idea_for_us": "Give our SNN dense localization supervision. Auto-label the ball cell per frame from tracks.npz and from pasted tracks (paste() already knows the cells). Add a conv-only spiking head that outputs x/y populations or a 10x10 ball map, with soft-target MSE and zero targets on feint, idle and pre-release frames. This turns 33 tracked throws into hundreds of labelled frames, ties the features to the ball instead of the arm or session, and keeps spike rates low. It also gives an interpretable overlay for the demo.",
   "how_to_test_tonight": "GPU, 1-2 h: add the auxiliary head on the 10x10 feature map (loss weight 0.3-1.0) to a scratch copy of ball_cv.py and train on folds A,B,C against w2_win_paste. Check throw-vs-feint AUC and whether the predicted ball cell follows the ball on held-out throws.",
   "confidence": "high: HTML v2 read"
  },
  {
   "title": "An Event-Based Perception Pipeline for a Table Tennis Robot (Ziegler, Gossard, Glover, Zell)",
   "url": "https://arxiv.org/abs/2502.00749",
   "year": "2025",
   "what_they_did": "Each event updates an EROS surface (an exponentially decaying time surface: the neighbourhood is decayed and the event pixel set to maximum). A fast Hough circle detector runs on that surface inside a region around the last ball position. An event trail filter and tuned camera biases suppress noise and maximize ball events. An EKF predicts the trajectory.",
   "key_numbers": "4,136 +- 101 detections/s vs 149 Hz frame-based. ROI detection 0.09 ms, full-resolution detection 0.66 ms. Pixel error 1.34 px.",
   "idea_for_us": "A decaying surface makes sparse, fast blobs continuous, which could link faint ball fragments that our per-frame blob test drops. After the first hit, track only a small region above the hand to cut false hits from elsewhere in the scene.",
   "how_to_test_tonight": "CPU, about 10 min: M_t = max(d * M_{t-1}, F_t) with d about 0.6 per 10 ms. Run the chain() blob logic on M_t > 0.3 and count tracked throws and feint hits against the plain binary frames.",
   "confidence": "high: HTML read; the paper gives no end-to-end latency"
  },
  {
   "title": "Fast Trajectory End-Point Prediction with Event Cameras for Reactive Robot Control (Monforte et al., CVPRW)",
   "url": "https://arxiv.org/abs/2302.13796",
   "year": "2023",
   "what_they_did": "An event-driven centre-of-mass tracker in a ROI (about 1 kHz) feeds (x, y, dt) to a stateful LSTM with 350 hidden units that predicts where and when the ball ends. Pretrained on ESIM-simulated trajectories and fine-tuned on real ones. A Panda arm intercepts a bouncing ball. Runs on a CPU only.",
   "key_numbers": "5,400 simulated trajectories plus 310 real for fine-tuning. Tracker about 1 kHz. No end-to-end latency reported.",
   "idea_for_us": "Split the problem: the tracker gives a few interpretable features per frame and a tiny model decides. Simulation pretraining plus a small real fine-tune matches our situation, where a parametric synthetic ball generator is the cheap simulator.",
   "how_to_test_tonight": "CPU, about 30 min: from the causal tracker, extract per-clip features (max rise, chained frames, mean vy, RANSAC residual, blob size, gap to the arm component). Train logistic regression or a small GBDT with the A/B/C folds and compare AUC to the SNN. If it wins, use it as the gate.",
   "confidence": "high: ar5iv full text read"
  },
  {
   "title": "Event-Based Trajectory Prediction Using Spiking Neural Networks (Debat et al., Frontiers in Computational Neuroscience)",
   "url": "https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.658764/full",
   "year": "2021",
   "what_they_did": "A 3-layer convolutional LIF SNN (60/80/100 filters, lateral inhibition, delays for motion direction) learns unsupervised with STDP. A polynomial regression readout predicts where a thrown ball will arrive from the first 15-90% of its trajectory. There is no explicit detector, and the ball is the only moving object.",
   "key_numbers": "297 recorded throws (208 train, 89 test). Already good with only 20% of the training throws, levelling off around 80%. At 15% of the trajectory (about 0.114 s after the throw starts) the mean error is 7.7 px vs 9.4 px for a mean baseline. Left/right direction was never wrong.",
   "idea_for_us": "A calibration point: small SNNs learn ball-trajectory features from a few dozen throws only when the ball is the only moving thing. Our 49 visible-ball throws sit among wind-ups and feints with identical arm motion, so we need ball-centred supervision or augmentation, not more raw clip labels.",
   "how_to_test_tonight": "Not tonight (STDP training is out of scope). Use it as a data-size argument for the jury and for choosing the tracker-first design.",
   "confidence": "medium-high: article read; the training count is inconsistent in the source (208 vs 211)"
  },
  {
   "title": "Learning Agile Humanoid Dodgeball with Onboard Monocular Vision (arXiv, Unitree G1)",
   "url": "https://arxiv.org/abs/2611.01197",
   "year": "2026",
   "what_they_did": "YOLOv8 bounding-box histories go into a temporal estimator that outputs the relative ball position and a binary threat flag, which feed an RL dodge policy. Training happens in simulation: analytic projection plus perception randomization (dropped detections, centre jitter, size noise, 30-50 ms delay) stands in for rendering. Explicit non-threat samples (stationary, low-speed, receding or absent ball) are mixed in at roughly 1:1:2 to avoid bias toward predicting threats.",
   "key_numbers": "Over 85% single-throw dodge success in more than 15 hardware trials. Simulated non-threat recall 0.967-0.993 (sim only). Detector trained on 3,050 augmented images from about 830 labelled frames.",
   "idea_for_us": "Generate ball tracks parametrically instead of only reusing the 33 real tracks: disc radius 1-2.5 cells, 0.8-2.5 cells per frame upward, cell dropout 20-60%, jitter. Add explicit hard negatives labelled no-throw: a blob that rises 1-3 frames then stops or reverses with the hand, drifts sideways or falls. The network then learns the defining contrast, detached plus sustained ballistic rise, and cannot memorize real tracks.",
   "how_to_test_tonight": "GPU, about 1 h: extend Sampler.paste in a scratch copy with the parametric generator and hard negatives (mix roughly paste-throw : real : hard-negative = 1:1:2). Compare with w2_win_paste on folds A,B,C, especially feint false triggers and idle triggers per minute.",
   "confidence": "medium: HTML read but partial; no hardware false-alarm data; non-threat recall is from simulation"
  },
  {
   "title": "REBot: Reflexive Evasion Robot for Instantaneous Dynamic Obstacle Avoidance (Unitree Go2)",
   "url": "https://arxiv.org/abs/2508.06229",
   "year": "2025",
   "what_they_did": "On our platform, the Go2: a finite-state machine (normal, avoidance, recovery) with PPO avoidance and recovery policies trained in Isaac Gym. The robot mostly jumps away or crouches. Obstacle states came from OptiTrack, so perception was assumed. Thrown balls, pokes and kicks were tested.",
   "key_numbers": "Avoidance success by warning time: 0.05 at 0.1-0.5 s, 0.65 at 0.5-1.5 s, 0.81 at 1.5-4 s. Real robot: 56% avoidance, 53% recovery. Peak joint power above 500 W for very short warnings.",
   "idea_for_us": "A timing budget for the Go2: below about 0.5 s of warning a dodge almost never works. With a high arc, the trigger should fire within about 50-100 ms of release, and the dodge should be the fastest sport-API action. Our harness counts a detection up to 250 ms after release (--det-frames 25), which hides latency.",
   "how_to_test_tonight": "Eval only: rerun the evaluation on existing checkpoints with --det-frames 5 and 10 to get recall within 50 and 100 ms. Time the chosen sport-API dodge in the MuJoCo emulator (sim/tools/measure_reflex.py) to set a hard deadline.",
   "confidence": "high: HTML read"
  },
  {
   "title": "Can Spiking Neural Networks play pinball? Learning-free neuromorphic motion detector for target tracking (plus insect STMD models)",
   "url": "https://arxiv.org/abs/2609.24403",
   "year": "2026",
   "what_they_did": "A learning-free network of spiking Time-Difference Encoders jointly estimates the position, speed and direction of a small, fast pinball ball from a DVS. It runs on SpiNNaker and plays closed-loop pinball. Related insect-inspired small-target motion detector (STMD) models, e.g. STMDNet (https://arxiv.org/abs/2501.13054), use local excitation and opposing inhibition to suppress large-object and background motion so only tiny moving targets respond.",
   "key_numbers": "21.7 ms reaction including 5 ms network latency, under 25k neurons, about 148 microwatts estimated, 56.1% hit rate (about twice the human average). No training data.",
   "idea_for_us": "An original, explainable layer 0 for the SNN, which could earn originality points: fixed centre-surround (DoG) kernels that pass ball-sized blobs and suppress the large arm or torso, combined with an upward-correlation (delay-and-correlate) unit. It can be stored as Q2.6 weights and should hold up with unseen people and clothes because it is not learned.",
   "how_to_test_tonight": "CPU, about 20 min: offline, apply a DoG filter (centre 3x3 +1, surround 7x7 -0.25) to the binary frames and AND it with an upward shift (cell active at t and a cell 1-2 rows below active at t-1). Compare the per-frame max response on tracked ball frames vs feint arm frames (AUC). If it separates them well, freeze it as the first conv layer.",
   "confidence": "medium: pinball paper read only at abstract level; STMD details from search snippets (low)"
  },
  {
   "title": "Prophesee GenX320 bias and on-chip event-filter documentation (bias_hpf, AFK, STC/Trail, ERC)",
   "url": "https://docs.prophesee.ai/stable/hw/manuals/biases.html",
   "year": "2025-2026 docs",
   "what_they_did": "Vendor documentation. bias_hpf is the pixel high-pass filter: raising it filters out slow light changes, lowering it makes the sensor react to slower changes. bias_diff_on and bias_diff_off set the contrast thresholds, and bias_refr sets the refractory period. The GenX320 also has on-chip filters: anti-flicker (AFK), STC/Trail filters that remove redundant same-polarity bursts, and an event rate controller (ERC) at the end of the chain that drops events above a rate cap and degrades the signal when active (https://docs.prophesee.ai/stable/hw/manuals/esp.html). Ziegler et al. tuned biases specifically to maximize ball events.",
   "key_numbers": "bias_hpf range 0-127, default 40 on mass-production parts. Sensor latency under 150 microseconds at 1000 lux.",
   "idea_for_us": "(a) The session where the ball is invisible (04/215305) may be caused by ERC clipping during busy arm motion or by low ball-to-background contrast. That is a sensor issue, not a model issue, so keep it out of throw-vs-feint training. (b) For the demo, raising bias_hpf could suppress slow walkers while the fast ball still fires, and STC/Trail removes noise. This would cut idle false triggers with no retraining of the tracker. The SNN would need a little re-recorded data if biases change.",
   "how_to_test_tonight": "CPU, about 10 min: plot the per-10 ms event count for 04/215305 vs a good session; a flat ceiling means ERC clipping. If the camera is free: record 1 min of idle walking plus 5 throws at bias_hpf 40 vs about 80, and compare walker event counts with ball blob cells.",
   "confidence": "medium: biases page read; on-chip AFK/STC/Trail/ERC confirmed only through a search summary of the product brief and ESP docs; effect on our scenes untested"
  }
 ],
 "top_ideas": [
  "1. Make a causal ball tracker the main trigger and use the SNN only as a context gate (AND fusion). Every system that really catches or dodges thrown balls (Delbruck's goalie, Falanga's drone, Forrai's ANYmal, Zhu's Mini-Cheetah, EV-Catcher, the Ziegler table-tennis work) finds the ball itself and checks the physics. None classifies the thrower's motion, and our feints have the same wind-up by design. Concretely: run ball_tracks.chain() causally without the top-band condition (fire at the first frame where 4 or more linked small blobs have risen 6 or more cells). Add a RANSAC constant-velocity check (residual under 1 cell, 0.8-2.5 cells per frame up), Delbruck-style gates (support and speed, prune on a 1-frame gap, ignore blobs attached to the big arm component, arm only after wind-up activity, 2 s refractory) and Zhu-style hysteresis to keep faint balls. Expected trigger is 40-60 ms after release. It runs on CPU in under 1 h, and can be scored on all 77 throws, 104 feints and idle clips (recall, feint hits, idle triggers per minute).",
  "2. Give the SNN dense ball supervision instead of clip labels only, following Ziegler et al. 2024. Add a conv-only spiking localization head (x/y populations or a 10x10 map, soft-neighbour MSE) trained from the tracks.npz cells and the pasted tracks, with zero targets on feint, idle and pre-release frames. Optionally replace the second polarity channel with an EV-Catcher-style binary history (OR of the last ~8 frames), so the ball streak is visible in one step and the FPGA input stays 2x80x80. This turns 33 tracked throws into hundreds of labelled frames and forces features onto the ball instead of the arm or the session.",
  "3. Replace or extend real-track pasting with a parametric ballistic ball generator plus explicit hard negatives, following the humanoid-dodgeball recipe (analytic projection, perception randomization, non-threat samples mixed about 1:1:2). Randomize disc size (1-2.5 cells), upward speed (0.8-2.5 cells per frame), cell dropout (20-60%) and jitter. Hard negatives labelled no-throw: blobs that rise 1-3 frames then stop or reverse with the hand, drift sideways or fall. This prevents memorizing the 33 real tracks and teaches exactly the decisive cue, a detached and sustained ballistic rise.",
  "4. Fix the inputs and the deadline. (a) Check whether the invisible-ball session 04/215305 comes from ERC clipping or low contrast (look for a flat event-rate ceiling) and keep it out of throw-vs-feint training. For the demo, consider raising GenX320 bias_hpf and enabling the on-chip STC/Trail filters to suppress slow walkers and noise (re-record a little if biases change). (b) Evaluate recall within 50-100 ms of release (--det-frames 5 and 10) instead of 250 ms. REBot on the Go2 shows dodge success near zero with under 0.5 s of warning, and our ball leaves the view after 60-150 ms."
 ]
}