# Which uncertainty should decide when to ask the expert? Deferral under distribution shift on BloodMNIST

Maryam Tavasoli. Technical report, November 2026. Code: github.com/maryamtavasoli98/expert-deferral-under-shift

## Abstract
We compare five predictive-uncertainty methods (single softmax, temperature scaling, MC-dropout, a five-member deep ensemble and a from-scratch last-layer Laplace approximation) for a deployment question: when a classifier meets shifted data, which uncertainty signal should route cases to a fallible expert under a fixed review budget? On BloodMNIST with 25 controlled corruptions, two near-OOD sets and a noise set, 15 CNNs over three seeds, all methods are badly overconfident under severe shift. The ensemble gives the best system accuracy at equal workload (about 1.5 points above a single softmax at a 10 % budget); temperature scaling fixes calibration but hardly changes the case ranking; confidence-based OOD scores fall below chance on skin-lesion images. This is a small simulation study, not a clinical one.

## 1. Introduction
Models deployed outside their training distribution fail silently. One response, studied in the ERC project ODD-ML (Out-of-Distribution Deployable Machine Learning, CORDIS 101201120) [3], is to keep a domain expert in the loop with minimal workload; recent work from that group studies adaptation with noisy expert feedback [1] and human-machine collaboration under shift [2]. We study a much simpler, deployment-time question with calibrated baselines: given an expert who answers correctly with probability r and a review budget b, which uncertainty estimate gives the best system? We also report calibration and OOD detection, because the threshold rule "defer if confidence < r" is only sound for calibrated probabilities.

## 2. Related work
Calibration of deep networks and temperature scaling [5]; deep ensembles [6]; MC-dropout [7]; Laplace approximations, in particular last-layer variants [8, 9]; benchmarking predictive uncertainty under dataset shift [10] and corruption benchmarks [12]; the max-softmax OOD baseline [11]; selective classification [13] and learning to defer to an expert [14]. MedMNIST [4] provides the data. We read the abstracts of all cited works; we do not reproduce their methods except where stated.

## 3. Method
Data, model, the five methods, metrics and the expert simulation are as in the repository README (sections Data and Method). In short: a two-block CNN with a 128-d penultimate layer, AdamW with one-cycle schedule for 20 epochs, checkpoint by clean validation NLL; temperature and Laplace prior precision are fitted on clean validation data only. The Laplace approximation uses the full generalised Gauss-Newton Hessian of the 8 x 129 head weights; its Hessian is tested against the autograd Hessian of the softmax cross-entropy. Shifts: Gaussian noise, blur, brightness, contrast, pixelation at five severities and a mixed stream with one random corruption per image. Expected system accuracy sends the b most uncertain cases to an expert with reliability r (expert errors count as errors); an oracle reviewing the model's errors first and random deferral are the bounds.

## 4. Experiments
All values mean ± std over three seeds.

### 4.1 Setup
<!-- table:setup -->
- 15 CNNs (3 seeds x 5 members), mean clean test accuracy 0.937, 29 s per model on `mps`.
- Fitted temperatures per seed: 1.45, 1.45, 1.37. Laplace prior precision chosen on clean validation NLL per seed: 100, 100, 100.
- Software: torch 2.15.0 on arm64.
<!-- /table:setup -->

### 4.2 Calibration and selective risk
<!-- table:calibration -->
| method | condition | accuracy | ECE ↓ | NLL ↓ | AURC ↓ |
|---|---|---|---|---|---|
| single softmax | clean test | 0.942 ± 0.002 | 0.017 ± 0.004 | 0.191 ± 0.010 | 0.008 ± 0.000 |
| temperature scaling | clean test | 0.942 ± 0.002 | 0.010 ± 0.002 | 0.180 ± 0.008 | 0.008 ± 0.000 |
| MC-dropout (30) | clean test | 0.941 ± 0.004 | 0.010 ± 0.004 | 0.179 ± 0.007 | 0.008 ± 0.001 |
| deep ensemble (5) | clean test | 0.945 ± 0.001 | 0.008 ± 0.003 | 0.165 ± 0.003 | 0.007 ± 0.000 |
| last-layer Laplace | clean test | 0.942 ± 0.002 | 0.011 ± 0.003 | 0.180 ± 0.009 | 0.007 ± 0.001 |
| single softmax | mixed shift, severity 3 | 0.569 ± 0.023 | 0.314 ± 0.056 | 3.460 ± 0.864 | 0.185 ± 0.027 |
| temperature scaling | mixed shift, severity 3 | 0.569 ± 0.023 | 0.266 ± 0.059 | 2.526 ± 0.532 | 0.184 ± 0.027 |
| MC-dropout (30) | mixed shift, severity 3 | 0.571 ± 0.022 | 0.232 ± 0.073 | 2.424 ± 0.621 | 0.171 ± 0.025 |
| deep ensemble (5) | mixed shift, severity 3 | 0.575 ± 0.005 | 0.219 ± 0.018 | 2.539 ± 0.301 | 0.159 ± 0.005 |
| last-layer Laplace | mixed shift, severity 3 | 0.569 ± 0.022 | 0.229 ± 0.072 | 2.465 ± 0.690 | 0.168 ± 0.025 |
| single softmax | mixed shift, severity 5 | 0.446 ± 0.016 | 0.420 ± 0.089 | 5.038 ± 1.451 | 0.351 ± 0.066 |
| temperature scaling | mixed shift, severity 5 | 0.446 ± 0.016 | 0.368 ± 0.099 | 3.749 ± 0.966 | 0.349 ± 0.066 |
| MC-dropout (30) | mixed shift, severity 5 | 0.446 ± 0.016 | 0.336 ± 0.110 | 3.458 ± 0.974 | 0.328 ± 0.057 |
| deep ensemble (5) | mixed shift, severity 5 | 0.465 ± 0.010 | 0.300 ± 0.029 | 3.680 ± 0.292 | 0.300 ± 0.013 |
| last-layer Laplace | mixed shift, severity 5 | 0.447 ± 0.017 | 0.329 ± 0.116 | 3.518 ± 0.961 | 0.325 ± 0.043 |
<!-- /table:calibration -->

### 4.3 OOD detection
<!-- table:ood -->
| method | score | near-OOD: DermaMNIST | near-OOD: OrganAMNIST | far-OOD: uniform noise |
|---|---|---|---|---|
| single softmax | max_prob | 0.267 ± 0.024 | 0.615 ± 0.118 | 0.250 ± 0.063 |
| single softmax | entropy | 0.151 ± 0.030 | 0.611 ± 0.132 | 0.183 ± 0.119 |
| temperature scaling | max_prob | 0.210 ± 0.011 | 0.610 ± 0.132 | 0.201 ± 0.106 |
| temperature scaling | entropy | 0.150 ± 0.029 | 0.605 ± 0.134 | 0.182 ± 0.119 |
| MC-dropout (30) | max_prob | 0.267 ± 0.047 | 0.903 ± 0.028 | 0.364 ± 0.146 |
| MC-dropout (30) | entropy | 0.235 ± 0.055 | 0.906 ± 0.032 | 0.359 ± 0.153 |
| MC-dropout (30) | mutual_info | 0.262 ± 0.060 | 0.975 ± 0.015 | 0.398 ± 0.168 |
| deep ensemble (5) | max_prob | 0.360 ± 0.011 | 0.917 ± 0.010 | 0.481 ± 0.211 |
| deep ensemble (5) | entropy | 0.271 ± 0.015 | 0.911 ± 0.012 | 0.471 ± 0.219 |
| deep ensemble (5) | mutual_info | 0.310 ± 0.015 | 0.972 ± 0.009 | 0.518 ± 0.249 |
| last-layer Laplace | max_prob | 0.442 ± 0.063 | 0.939 ± 0.027 | 0.579 ± 0.097 |
| last-layer Laplace | entropy | 0.432 ± 0.065 | 0.950 ± 0.031 | 0.581 ± 0.100 |
| last-layer Laplace | mutual_info | 0.517 ± 0.069 | 0.994 ± 0.005 | 0.702 ± 0.110 |
<!-- /table:ood -->

### 4.4 Deferral at equal workload (r = 0.9)
<!-- table:deferral -->
| stream | expert budget | no expert | random deferral | single softmax | temperature scaling | MC-dropout (30) | deep ensemble (5) | last-layer Laplace | oracle |
|---|---|---|---|---|---|---|---|---|---|
| clean | 10 % | 0.942 | 0.938 | 0.970 ± 0.001 | 0.969 ± 0.001 | 0.970 ± 0.001 | **0.973 ± 0.001** | 0.970 ± 0.001 | 0.990 |
| clean | 20 % | 0.942 | 0.933 | 0.973 ± 0.000 | 0.973 ± 0.001 | 0.974 ± 0.000 | **0.975 ± 0.001** | 0.974 ± 0.001 | 0.980 |
| shift_s3 | 10 % | 0.569 | 0.602 | 0.632 ± 0.026 | 0.632 ± 0.028 | 0.636 ± 0.031 | **0.647 ± 0.005** | 0.632 ± 0.032 | 0.661 |
| shift_s3 | 20 % | 0.569 | 0.635 | 0.695 ± 0.029 | 0.695 ± 0.028 | 0.701 ± 0.033 | **0.720 ± 0.004** | 0.702 ± 0.030 | 0.750 |
| shift_s5 | 10 % | 0.446 | 0.491 | 0.515 ± 0.025 | 0.515 ± 0.024 | 0.516 ± 0.024 | **0.530 ± 0.015** | 0.516 ± 0.026 | 0.540 |
| shift_s5 | 20 % | 0.446 | 0.537 | 0.579 ± 0.034 | 0.581 ± 0.032 | 0.583 ± 0.034 | **0.597 ± 0.011** | 0.583 ± 0.032 | 0.630 |
<!-- /table:deferral -->

### 4.5 Threshold policy (severity 5)
<!-- table:threshold -->
| method | r = 0.8: system acc / deferred | r = 0.9: system acc / deferred | r = 0.95: system acc / deferred | r = 1.0: system acc / deferred |
|---|---|---|---|---|
| single softmax | 0.604 / 28 % | 0.676 / 36 % | 0.734 / 42 % | 0.979 / 88 % |
| temperature scaling | 0.643 / 36 % | 0.753 / 49 % | 0.829 / 58 % | 0.989 / 93 % |
| MC-dropout (30) | 0.687 / 43 % | 0.785 / 54 % | 0.872 / 64 % | 0.996 / 96 % |
| deep ensemble (5) | 0.751 / 52 % | 0.857 / 63 % | 0.909 / 69 % | 0.995 / 92 % |
| last-layer Laplace | 0.684 / 42 % | 0.793 / 53 % | 0.883 / 64 % | 0.998 / 98 % |
<!-- /table:threshold -->

### 4.6 Ensemble size
<!-- table:ensemble -->
| members | ECE clean | ECE shift 5 | NLL shift 5 | AURC clean | AURC shift 5 |
|---|---|---|---|---|---|
| 1 | 0.017 ± 0.004 | 0.420 ± 0.089 | 5.038 ± 1.451 | 0.008 ± 0.000 | 0.351 ± 0.066 |
| 2 | 0.012 ± 0.005 | 0.367 ± 0.083 | 4.440 ± 0.948 | 0.007 ± 0.000 | 0.335 ± 0.038 |
| 3 | 0.008 ± 0.001 | 0.325 ± 0.058 | 4.048 ± 0.435 | 0.007 ± 0.000 | 0.316 ± 0.016 |
| 4 | 0.009 ± 0.001 | 0.335 ± 0.033 | 3.813 ± 0.172 | 0.007 ± 0.000 | 0.315 ± 0.006 |
| 5 | 0.008 ± 0.003 | 0.300 ± 0.029 | 3.680 ± 0.292 | 0.007 ± 0.000 | 0.300 ± 0.013 |
<!-- /table:ensemble -->

Figures are in the repository under results/figures (calibration versus severity, risk-coverage, expert-budget curves).

## 5. Limitations
Synthetic shifts on a 28 x 28 benchmark with a small CNN; three seeds and no significance tests; a constant-reliability expert whose errors are not modelled further and whose cost is only the reviewed fraction; a six-value prior grid for the Laplace approximation; DermaMNIST differs from BloodMNIST in task and appearance, so near-OOD here is entangled with covariate shift.

## 6. Conclusion
Under severe shift no method restores calibration, but ensembles and (for some scores) Laplace and MC-dropout give better review queues and better OOD scores than a single softmax, while temperature scaling mainly changes how many cases a threshold rule would defer. The gap to the oracle remains large, which motivates methods that use the expert's answers.

## References

[1] L. Mäkinen, J. Loría, S. Kaski. Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings. ICML 2026. arXiv:2602.19788.
[2] M. Parakkal Unni, S. Kaski. Human-Machine Collaboration on Generative Meta-Learning: Model and Algorithm. arXiv:2607.00926.
[3] ERC project ODD-ML, Out-of-Distribution Deployable Machine Learning, CORDIS grant 101201120. https://cordis.europa.eu/project/id/101201120
[4] J. Yang et al. MedMNIST v2. arXiv:2110.14795.
[5] C. Guo, G. Pleiss, Y. Sun, K. Weinberger. On Calibration of Modern Neural Networks. arXiv:1706.04599.
[6] B. Lakshminarayanan, A. Pritzel, C. Blundell. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. arXiv:1612.01474.
[7] Y. Gal, Z. Ghahramani. Dropout as a Bayesian Approximation. arXiv:1506.02142.
[8] E. Daxberger et al. Laplace Redux. arXiv:2106.14806.
[9] A. Kristiadi, M. Hein, P. Hennig. Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks. arXiv:2002.10118.
[10] Y. Ovadia et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. arXiv:1906.02530.
[11] D. Hendrycks, K. Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples. arXiv:1610.02136.
[12] D. Hendrycks, T. Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. arXiv:1903.12261.
[13] Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. arXiv:1705.08500.
[14] H. Mozannar, D. Sontag. Consistent Estimators for Learning to Defer to an Expert. arXiv:2006.01862.

