# research / research:bio_handcrafted (phase Research)

{
 "findings": [
  {
   "title": "Can Spiking Neural Networks play pinball? A neuromorphic motion detector for target tracking (learning-free TDE network)",
   "url": "https://arxiv.org/abs/2609.24403",
   "year": "2026 (preprint)",
   "what_they_did": "They tracked a fast ball from a Prophesee EVK4 (128x128 region of interest, ON events only) with a spiking network on SpiNNaker that has no trained weights. A fixed 3x3 Gaussian stride-2 conv removes noise and downsamples. Pooled 3x3 receptive fields (22x22) feed 8 direction-selective time-difference encoders (TDEs) per neighbourhood. Each TDE has a facilitation input (2 ms delay) and a trigger input (1 ms delay). A winner-take-all stage and readouts follow for position, direction (36-neuron ring) and speed (firing rate). All weights were set by hand. The only fitted part is a speed regression.",
   "key_numbers": "24.6k neurons; about 5 ms network latency (21.7 ms reaction including 16.7 ms accumulation); about 148 uW estimated; direction accuracy 68.5% with 3x3 receptive fields and 82.7% with 5x5; hit rate 56.1% vs 25.6% for the human average",
   "idea_for_us": "This is the closest published match to what we need: a direction-selective spiking ball detector set analytically, with no training data. Each stage maps to conv+LIF in riverops: a fixed smoothing/downsampling conv, then 'up' units, then integration and winner-take-all. Their larger receptive field gave better direction accuracy. Our 'below' offsets should therefore cover 1-3 cells, because our ball rises about 1-3.2 cells per 10 ms frame (median about 2; measured from tracks.npz tonight).",
   "how_to_test_tonight": "Build the fixed-weight stack from riverops SpikingConv2d layers (learnable_leak=False). Give it 2 outputs: throw = detector, no-throw = 0. Pass it to ball_cv.evaluate() on folds A/B/C and report AUC, recall at 10% feint false triggers and idle triggers per minute. Compare with ThrowSNN v1 and the classical tracker (33 hits, 0 of 68 feints). CPU is enough.",
   "confidence": "medium-high: read about 60% of the HTML full text (architecture and results sections); recent preprint"
  },
  {
   "title": "Retina-Inspired Object Motion Segmentation for Event-Cameras (OMS; Clerico et al.), building on Snyder et al. 2023 'Object Motion Sensitivity'",
   "url": "https://arxiv.org/abs/2408.09454",
   "year": "2024",
   "what_they_did": "They turned events in a time window into binary frames, with both polarities merged into one channel. Two fixed circular averaging kernels are applied: a center of radius 2 and a surround of radius 4. The output is center minus surround, then a threshold. The method has no learning and 80 fixed weights in total, and it removes camera ego-motion and large-field motion so that independently moving objects remain.",
   "key_numbers": "mIoU 80.1% on EV-IMO and 81.6% on MOD; detection rate 92.6% and 100%; EVDodgeNet scores 65.8% and 75% mIoU. The method fails when background event density is much higher than object density.",
   "idea_for_us": "Use one fixed difference-of-Gaussians conv with an LIF threshold (tau about 1, so it has no memory) as layer 1 on merged polarities. It produces a 'small, isolated, moving thing' map that suppresses the arm and body, wide edges, people walking close to the camera and robot ego-motion. Scale the kernels to our ball, which is about 12-38 cells (3-6 cells across). This is the ESTMD/STMD size selectivity, expressed as a plain conv.",
   "how_to_test_tonight": "On CPU, apply conv2d with a DoG kernel to the merged-polarity frames. For several values of (center radius 1-2, surround radius 3-5, threshold), measure the fraction of ball-track cells (tracks.npz) that survive against the fraction of feint and idle event cells that survive. Choose the setting with the best separation.",
   "confidence": "high: read the HTML full text"
  },
  {
   "title": "Parameter Optimization and Learning in a Spiking Neural Network for UAV Obstacle Avoidance targeting Neuromorphic Processors (spiking LGMD, Salt, Howard, Indiveri, Sandamirskaya)",
   "url": "https://arxiv.org/abs/1910.07960",
   "year": "2019 (IEEE TNNLS 2020)",
   "what_they_did": "They built a spiking LGMD looming detector fed by a DVS on a quadrotor. It has a photoreceptor layer, a summing layer with slow and fast distance-weighted lateral inhibition, pooling-like intermediate layers and an LGMD neuron with AEIF adaptation, plus optional STDP. They did not train weights. Instead they tuned 18 hand parameters (decays, gains, inhibition ratios, adaptation) with differential evolution, Bayesian optimisation and self-adaptive DE. Everything ran in Brian2 simulation, aimed at DYNAP-SE.",
   "key_numbers": "Accuracy 0.8-1.0 on simple stimuli; 0.66-1.0 on a rolling ball and 0.7-0.8 on a cup. On a hand stimulus: accuracy 0.5 and specificity 0, so it fired on hand motion. The optimizers needed about 440-1350 evaluations.",
   "idea_for_us": "(a) Do not gradient-train thousands of weights on 77 throws. Write the detector as about 8-10 interpretable parameters and fit them with DE or CMA-ES on the training folds. (b) Caveat: a pure looming/LGMD detector responds to hand movements, so ours needs size selectivity and upward-direction selectivity, not raw motion energy. (c) Borrow their feed-forward inhibition idea to veto whole-field activity.",
   "how_to_test_tonight": "Write objective(params) as -recall at 10% feint false triggers + lambda * idle triggers per minute, computed with the ball_cv metric functions on the training folds. Optimize it with scipy.optimize.differential_evolution (popsize about 10, about 300 evaluations). Report on the held-out folds.",
   "confidence": "high: read the ar5iv full text"
  },
  {
   "title": "SLoN: spiking looming perception network with ON/OFF channels, delayed inhibition and feed-forward inhibition",
   "url": "https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1291053/full",
   "year": "2024",
   "what_they_did": "They built a feed-forward LIF network with all parameters set by hand and no training. ON and OFF channels come from frame differences. Excitation arrives without delay, while inhibition arrives delayed and spread spatially. A feed-forward inhibition (FFI) stage silences the network when population activity is high, as during fast view shifts. One output neuron combines ON and OFF. Input was frames (100x100 at 30 Hz), not events.",
   "key_numbers": "Delay sweep: 0 gave no response; 1 detected approach but could not separate it from receding; 2 separated them best; larger delays lost looming detection. Detected 12 of 12 real crash videos and 4 of 4 ball-collision clips. No false-positive rates are reported.",
   "idea_for_us": "Selectivity comes from the timing offset between excitation and the delayed, spatially offset inhibition, and that offset has a narrow sweet spot. In our conv-LIF design, the corresponding knobs are the per-layer leak (how long the trace from the cell below lasts) and the spatial offset d. Both must match the ball speed of about 1-3 cells per frame. Their FFI is a free robustness feature: global activity inhibits the output.",
   "how_to_test_tonight": "Sweep tau_trace in {2,3,4,6} and d in {1,2,3}. For each, compute a direction-selectivity index: up-detector response on the real throw clips against the same clips time-reversed or vertically flipped. These give the ball moving down, a clean null-direction control. Also measure the response on feint wind-ups.",
   "confidence": "high: read most of the full text"
  },
  {
   "title": "Finding the Gap: Neuromorphic Motion Vision in Cluttered Environments (spiking elementary motion detector sEMD / time-difference encoder)",
   "url": "https://arxiv.org/abs/2102.08417",
   "year": "2021 (Nature Communications 2024)",
   "what_they_did": "This is a spiking Reichardt-type detector. A facilitatory input sets an exponentially decaying gain, and a later trigger input from the neighbouring pixel turns that gain into a burst of spikes. Spike count falls as the travel time between the two pixels grows, and swapping the two roles flips the preferred direction. A coincidence filter comes first: each 2x2 macropixel fires only if more than 50% of its pixels spiked within 20 ms. The paper reports implementations in Brian2, Nengo and NEST, on SpiNNaker and Loihi, and in analog CMOS. The robot ran a simplified version on SpiNNaker.",
   "key_numbers": "About 4k neurons and 300k synapses. Robust from 5 to 5000 lux. 50% response at about 35% contrast. 81% mean success navigating clutter. No latency or power figures are given.",
   "idea_for_us": "(1) Their noise prefilter is just a 2x2 or 3x3 conv with a count threshold, which is cheap and should reduce false triggers on idle scenes. (2) Their facilitation corresponds to our LIF membrane trace, so a 'below-then-here' coincidence unit is an additive TDE. (3) Spike count encodes speed, so use 2-3 offsets to cover the ball's speed range.",
   "how_to_test_tonight": "Insert a 3x3 coincidence layer (threshold of at least 2-3 active cells) in front of the detector, or in front of ThrowSNN. Measure the change in idle triggers per minute and the loss in ball-track recall.",
   "confidence": "high: read the HTML full text (first ~75%)"
  },
  {
   "title": "TDE-3: An improved prior for optical flow computation in spiking neural networks",
   "url": "https://arxiv.org/abs/2402.11662",
   "year": "2024",
   "what_they_did": "They added a third, inhibitory input to the time-difference encoder that resets the facilitation gain. This removes residual responses to null-direction motion in textured scenes. The architecture stays analytic, and its few parameters were trained with BPTT and surrogate gradients on synthetic moving edges. They tested on DAVIS240 recordings.",
   "key_numbers": "Direction-selectivity index 0.36 for TDE-2 vs 1.0 for TDE-3 (synthetic). About 2.6x fewer spikes on real scenes. Angular error about 18-22 degrees.",
   "idea_for_us": "Give the up-detector an explicit null-direction veto: a negative weight from the cell above, the Barlow-Levick term. riverops keeps signed Q4.12 potentials, so this inhibition persists into the next frame. TDE-3 also suggests a middle path: keep the hand-designed wiring and fit only its 4-6 scalars (a, b, c, tau, theta) with surrogate gradients on our clips.",
   "how_to_test_tonight": "Ablation: compare the up-detector with and without the -c (above) term. Measure the direction-selectivity index on flipped throws and the feint false-trigger rate.",
   "confidence": "high: read the HTML full text (first ~70%)"
  },
  {
   "title": "Approaching Retinal Ganglion Cell Modeling and FPGA Implementation for Robotics (approach-sensitivity cell on DVS)",
   "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7512993/",
   "year": "2018",
   "what_they_did": "They modelled a hand-designed retinal approach-sensitive cell. Subunits use OFF events as excitation and ON events as inhibition, are half-rectified and apply local center-surround subtraction; an integrate-and-fire neuron reads them out. It responds to expanding dark objects and stays silent for lateral motion, where ON and OFF cancel, and for receding objects. They implemented it on a Spartan-6 FPGA and drove a mobile robot with it.",
   "key_numbers": "160 ns per event at 50 MHz; 2% of registers and 12% of LUTs; worst-case 90 ms from spike to wheel motion",
   "idea_for_us": "Polarity asymmetry can be a one-frame direction cue. I checked this on our data tonight: across the 33 ball tracks in tracks.npz, ON cells sit above OFF cells in only 56% of frames on average. Only 21% of tracks reach 70% or more, and 15% reach 30% or less. At 4x4 downsampling, polarity layout is therefore not a reliable 'moving up' cue for our ball. Merge polarities for motion detection (the pinball and OMS works also discard polarity) and get direction from timing (Barlow-Levick/TDE).",
   "how_to_test_tonight": "Already done (numpy on tracks.npz, CPU). Optional follow-up: the same statistic on the feint hand blob.",
   "confidence": "high: read the full text; our polarity check was run tonight"
  },
  {
   "title": "HFirst: A Temporal Approach to Object Recognition (hand-coded spiking HMAX on FPGA)",
   "url": "https://arxiv.org/abs/1508.01176",
   "year": "2015",
   "what_they_did": "They built a spiking recognition network for DVS input. The first layer is a fixed 7x7 Gabor bank (12 orientations), followed by temporal winner-take-all pooling (the first neuron to spike resets its neighbours). The S2 templates are built directly from C1 spike counts on training views, and the classifier is a share of spikes or an argmax. There is no gradient training, and per-layer thresholds and leaks are fixed by hand. It runs on a Spartan-6 FPGA.",
   "key_numbers": "Latency of 2 us or less on the FPGA; 97.5% on 4-class poker pips and 84.9% on 36 characters; 150-250 mW",
   "idea_for_us": "A frozen hand-designed spiking conv front end with a cheap head already works on FPGA hardware. For us: freeze the hand-designed motion layers and fit only the head. A spike-count template or a logistic fit on the 5x5 feature counts is enough. That leaves very few free parameters for 77 throws.",
   "how_to_test_tonight": "Freeze layers 1-2 (hand channels) and train only the head, or the head plus one layer, with ball_cv's window loss. Compare held-out throw-vs-feint AUC with v1, which scores about 0.55-0.68.",
   "confidence": "high: read the ar5iv full text"
  },
  {
   "title": "Event-based attention and tracking on neuromorphic hardware (dynamic neural fields on Loihi)",
   "url": "https://arxiv.org/abs/1907.04060",
   "year": "2019",
   "what_they_did": "DAVIS240 events are downsampled to a 64x64 grid in 1 ms bins. Two spiking dynamic-neural-field (DNF) layers follow, with hand-set Mexican-hat lateral kernels (excitation sigma 2, inhibition sigma 4) and a 40-neuron global inhibition group. Layer 1 forms bumps on all objects. Layer 2 keeps a single bump that follows the cued object and ignores equally salient distractors. OFF events inhibit layer 1 so fast objects leave no tails.",
   "key_numbers": "Mean tracking error 3.5 px; about 8k neurons; all weights hand-set",
   "idea_for_us": "Global inhibition or winner-take-all lets the network lock onto one small object and suppress clutter, such as a second person or waving arms. In feedforward riverops form, this is a 'total activity' channel at the 5x5 stage with a negative head weight. riverops also includes a RecurrentSpikingConv2d that could carry the lateral recurrence, if the bitstream supports it.",
   "how_to_test_tonight": "Add a global-activity channel to the hand-designed head (one fixed all-ones kernel through the pyramid with a negative head weight). Measure idle triggers per minute on the busy idle recording, with and without it.",
   "confidence": "high: read the ar5iv full text"
  },
  {
   "title": "Robotic goalie with 3 ms reaction time at 4% CPU load using event-based dynamic vision sensor",
   "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3836084/",
   "year": "2013",
   "what_they_did": "An event-driven cluster tracker runs on a 128x128 DVS. A cluster becomes visible after about 10 supporting events and a speed above 20 px/s. Clusters with no support for about 10 ms are pruned. Velocity comes from a rolling least-squares fit over the last 10 packets (10-30 ms). The goalie targets the cluster predicted to reach the goal next, and a 'sleeping' state ignores hands and people walking by when no ball has been seen for a while.",
   "key_numbers": "Latency 2.2 +/- 2 ms from ball movement to motor command; 550 Hz median update rate; under 4% CPU; about 80% of shots blocked (67% in the results section)",
   "idea_for_us": "Robustness wrapper for the jury's 'false triggers' criterion. Arm the trigger only after a wind-up-like event, sleep otherwise. Require a host-side check, such as a small rising blob in the next 2-3 frames, before the Go2 moves. This is the two-stage reflex that already exists in the repo, now with published precedent.",
   "how_to_test_tonight": "Replay the held-out idle and feint clips through 'SNN trigger AND tracker confirmation within 30 ms'. Report idle triggers per minute and the added latency.",
   "confidence": "high: read the full text"
  },
  {
   "title": "Event-based Agile Object Catching with a Quadrupedal Robot (ANYmal)",
   "url": "https://arxiv.org/abs/2303.17479",
   "year": "2023 (ICRA)",
   "what_they_did": "An event camera on ANYmal detects thrown objects without a learned detector. Events are ego-motion compensated with the IMU, combined into a mean-timestamp image and thresholded. The pixels are then clustered with DBSCAN using distance, flow and timestamp similarity. Depth comes from the known object size, and a RANSAC ballistic parabola gives the impact point for a PPO policy.",
   "key_numbers": "100 Hz on a Jetson Orin; 83% catch rate for objects up to 15 m/s; impact point within the net in 93% of throws under 9 m/s",
   "idea_for_us": "On a legged robot, a geometric, hand-designed detector is the state of practice, which supports our tracker-first approach for the jury. If the camera is mounted on the moving Go2, events need ego-motion compensation or an OMS-style surround (finding 2).",
   "how_to_test_tonight": "Not tonight for the full pipeline. Our ball leaves the image about 9 frames (median, tracks.npz) after release, so a parabola fit has little data. Use only the vertical-speed estimate as an extra confirmation feature.",
   "confidence": "high: read the ar5iv full text"
  },
  {
   "title": "Our FPGA toolchain (riverops, local install) can express hand-designed motion detectors",
   "url": "file:///home/noah/miniconda3/envs/river/lib/python3.10/site-packages/riverops/nn/spiking_conv.py",
   "year": "2026",
   "what_they_did": "I read the installed riverops source (read-only). The LIF step is V = V - V/tau + I, with a spike when V >= threshold, and HARD (V=0) or SOFT (V-threshold) reset. Potentials saturate at the Q4.12 range but stay signed, so inhibition held in the membrane carries into the next frames. Tau and threshold are per layer (scalar leak and v_threshold); conv bias is per channel. Setting tau=1 with learnable_leak=False gives a memoryless layer. The export IR does not record the reset type (README known issue). The hardware graph schema (RiverModelGraph) contains ConvLIF2d, RecurrentConvLIF2d, Slicing2d and Injection2d (Concat is lowered to Injection).",
   "key_numbers": "Weights Q2.6 (step 1/64, range -2 to +1.98); potentials Q4.12 (range -8 to +8)",
   "idea_for_us": "(1) Per-channel bias can act as a per-channel threshold offset, since the threshold is per layer. Steady-state shift = bias x tau. (2) Parallel fixed and learned branches can be concatenated in hardware (Injection2d). (3) RecurrentSpikingConv2d feeds the previous frame's spikes back through its own conv. That is an exact 1-frame delay line, enough for a true Reichardt/TDE unit if the bitstream implements it. (4) Validate every design under the reset mode the FPGA actually uses.",
   "how_to_test_tonight": "Ask the organizers or check DEPLOY.md and KRIA.md for (a) the reset mode, (b) RecurrentConvLIF2d support in the KV260 bitstream and (c) biases in backbone layers. In simulation, run the up-detector with fixed_point enabled to confirm that negative potentials persist across frames.",
   "confidence": "high: read the source code directly"
  }
 ],
 "top_ideas": [
  "1. Build a training-free release detector inspired by insect vision, in riverops conv+LIF, that copies the tracker's rules (blob of 3-45 cells, rising at least 6 cells over at least 4 frames, reaching the top band). Layer 1: OMS/ESTMD difference-of-Gaussians on merged polarities (center about 3x3 positive, 7x7 ring negative), tau=1, threshold so that blobs of 12-38 cells fire but the arm and body do not. Layer 2: Barlow-Levick 'up' units at offsets d=1,2,3: +a from the small-blob map at y+d (the cell below, remembered in the membrane), +b at y, -c at y-d (above, null-direction veto); tau about 3-4. Starting point a=0.5, b=0.7, c=1.0, threshold 1.0: by hand calculation this fires for upward motion but not for downward motion or a static blob. Layers 3-5: stride-2 integrators with tau about 8 that need at least 3-4 up-detections within 40-60 ms. Head (5x5): weight the top rows (the ball reaches the top band about 9 frames after release) minus an LGMD-style global-activity veto. Fit the 8-10 scalars with DE or CMA-ES on the training folds using the ball_cv metrics. The aim is tracker-level performance (33 hits, 0 of 68 feints) running on the FPGA, which also scores on originality, robustness and reproducibility.",
  "2. Hybrid model: freeze those hand-designed layers 1-2 as a parallel branch (Concat, exported as Injection2d) or as fixed channels inside ThrowSNN layers 1-2 (zero their gradients). Train only the downstream layers and the head with ball_cv's window loss. With 77 throws, a fixed motion prior plus a small head (HFirst, pinball TDE, TDE-3 precedent) should beat the current held-out throw-vs-feint accuracy of 0.55-0.68. Compare on folds A/B/C, with fold D reported separately.",
  "3. Exact delay lines: if the KV260 bitstream supports RecurrentConvLIF2d (it is in riverops and in the hardware graph schema, but must be confirmed with the organizers), use the recurrent path as a 1-frame delayed copy. That gives true sEMD/TDE units with TDE-3-style inhibition from the cell above, and allows DNF-style lateral global inhibition to lock onto one object. Prototype in simulation first.",
  "4. Cheap robustness additions with published precedent: a 3x3 coincidence prefilter against noise (sEMD), LGMD feed-forward inhibition and a global-activity veto against people walking or waving, and goalie-style arming plus host-side tracker confirmation within 30 ms before the Go2 moves. Measure each on the idle and feint clips as triggers per minute. Do not use polarity layout as a direction cue: ON is above OFF in only 56% of ball frames in tracks.npz."
 ]
}