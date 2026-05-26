# Conformal-EADC: Probe-Agnostic Streaming LLM Safety Detection

Code for the paper *"Beyond Fixed Thresholds: Conformal-EADC for Probe-Agnostic Streaming LLM Safety Detection"*.

## Directory Structure

```
src/                          # Core algorithm and experiment scripts
  eprocess.py                 # E-process engine, conformal calibration, drift profiler, ACI
  baselines.py                # FixedThreshold, SWMA, Delay-K baselines
  data_utils.py               # Data loading utilities
  offline_profiler.py         # Offline DKW drift profiler
  train_probe.py              # RoBERTa probe training
  run_m1_aci.py               # M1: ACI validation experiment
  run_m2_ablation.py          # M2: EADC+Baseline ablation
  run_m3_beavertails_v2.py    # M3: Cross-dataset (BeaverTails) validation
  run_m4_multiseed.py         # M4: Multi-seed robustness (5 seeds)
  run_m5_sensitivity.py       # M5: Hyperparameter sensitivity
outputs/                      # Experiment results
  m1_aci_results.json         # ACI comparison results
  m2_ablation_results.json    # EADC ablation results
  m3_beavertails_results.json # BeaverTails zero-shot transfer results
  m4_multiseed_results.json   # Multi-seed robustness results
  best_probe/                 # Trained RoBERTa probe (git-ignored)
  score_cache/                # Cached calibration/test scores (git-ignored)
data-FineHarm/                # FineHarm dataset (git-ignored)
data-BeaverTails/             # BeaverTails test set (git-ignored)
paper/                        # LaTeX paper (git-ignored until published)
```

## Setup

Requires Python 3.8+ with PyTorch and transformers:

```bash
pip install torch transformers scikit-learn scipy numpy
```

## Reproducing Experiments

1. Train the probe: `python src/train_probe.py`
2. Run offline profiler: `python src/offline_profiler.py`
3. Run individual experiments:
   - `python src/run_m1_aci.py` — ACI validation
   - `python src/run_m2_ablation.py` — EADC ablation
   - `python src/run_m3_beavertails_v2.py` — BeaverTails cross-dataset
   - `python src/run_m4_multiseed.py` — Multi-seed robustness
   - `python src/run_m5_sensitivity.py` — Hyperparameter sensitivity

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\alpha$ | 0.10 | Target FPR level |
| $B$ | 10 | Number of calibration buckets |
| $\epsilon_t$ mult. | 3 | DKW relaxation multiplier |
| $C_{\max}$ | 10 | Max EADC skip interval |
| $\rho$ | 2.0 | EADC exponent |
| $\beta$ | 0.9 | EWMA decay |
| $\gamma$ | 1.0 | EWMA exponent |
| $\eta$ | 0.005 | ACI learning rate |

## Citation

```bibtex
@article{qiu2025conformal,
  title={Beyond Fixed Thresholds: Conformal-EADC for Probe-Agnostic Streaming LLM Safety Detection},
  author={Qiu, Jianmin},
  year={2025}
}
```
