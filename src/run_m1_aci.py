"""
M1 Theory Fix: ACI (Adaptive Conformal Inference) vs ε_t × multiplier.

Compares three approaches:
  1. Baseline: ε_t with multiplier=1 (original DKW bound)
  2. Hacky: ε_t with multiplier=3 (current best)
  3. ACI: Adaptive α_t via Gibbs & Candès 2021

Only requires: numpy, scipy, eprocess.py (with ACI module)
"""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprocess import (
    EProcessEngine, DriftProfiler, ConformalCalibrator,
    evaluate_batch,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "outputs" / "score_cache"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_cached_data():
    """Load cached scores and metadata."""
    print("Loading cached data...")
    cal_safe_scores = np.load(CACHE_DIR / "cal_safe_scores.npy")
    cal_safe_lengths = np.load(CACHE_DIR / "cal_safe_lengths.npy")

    with open(CACHE_DIR / "drift_profiler.json", "r") as f:
        drift_data = json.load(f)
    drift_profiler = DriftProfiler()
    drift_profiler.epsilon_table = {int(k): v for k, v in drift_data["epsilon_table"].items()}
    drift_profiler.bucket_boundaries = np.array(drift_data["bucket_boundaries"])
    drift_profiler.n_buckets = drift_data["n_buckets"]
    drift_profiler.confidence_delta = drift_data.get("confidence_delta", 0.05)

    with open(CACHE_DIR / "test_scores.json", "r") as f:
        test_data = json.load(f)

    all_scores = []
    all_is_harmful = []
    all_harmful_onset = []

    items = test_data.values() if isinstance(test_data, dict) else test_data
    for item in items:
        scores = np.array(item["scores"], dtype=np.float32)
        all_scores.append(scores)
        all_is_harmful.append(item["is_harmful"])
        all_harmful_onset.append(item.get("harmful_onset"))

    print(f"  Calibration: {len(cal_safe_scores)} scores")
    print(f"  Test: {len(all_scores)} sequences "
          f"({sum(all_is_harmful)} harmful, {len(all_is_harmful) - sum(all_is_harmful)} safe)")
    return cal_safe_scores, cal_safe_lengths, drift_profiler, all_scores, all_is_harmful, all_harmful_onset


def run_aci_sweep(cal_scores, cal_lengths, drift_profiler,
                  all_scores, all_is_harmful, all_harmful_onset):
    """Sweep ACI configurations and compare with epsilon multiplier."""
    print("\n=== M1: ACI vs ε_t × multiplier ===\n")

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
    aci_gammas = [0.001, 0.005, 0.01, 0.02, 0.05]
    eps_multipliers = [1, 2, 3]

    results = {}

    for alpha in alphas:
        print(f"\n--- α = {alpha} ---")
        results[str(alpha)] = {}

        # 1. ε_t × 1 (original DKW)
        engine_dkw = EProcessEngine(
            cal_scores_safe=cal_scores,
            cal_lengths=cal_lengths,
            alpha=alpha,
            drift_profiler=drift_profiler,
            epsilon_multiplier=1,
            use_aci=False,
        )
        r_dkw = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine_dkw)
        results[str(alpha)]["eps_x1"] = {
            "prefix_fpr": r_dkw.prefix_fpr,
            "power": r_dkw.empirical_power,
            "leakage": r_dkw.mean_leakage,
            "pir": r_dkw.mean_pir,
        }
        print(f"  ε_t × 1:  FPR={r_dkw.prefix_fpr:.4f} Power={r_dkw.empirical_power:.4f} "
              f"Leak={r_dkw.mean_leakage:.1f}")

        # 2. ε_t × 3 (current best hack)
        engine_eps3 = EProcessEngine(
            cal_scores_safe=cal_scores,
            cal_lengths=cal_lengths,
            alpha=alpha,
            drift_profiler=drift_profiler,
            epsilon_multiplier=3,
            use_aci=False,
        )
        r_eps3 = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine_eps3)
        results[str(alpha)]["eps_x3"] = {
            "prefix_fpr": r_eps3.prefix_fpr,
            "power": r_eps3.empirical_power,
            "leakage": r_eps3.mean_leakage,
            "pir": r_eps3.mean_pir,
        }
        print(f"  ε_t × 3:  FPR={r_eps3.prefix_fpr:.4f} Power={r_eps3.empirical_power:.4f} "
              f"Leak={r_eps3.mean_leakage:.1f}")

        # 3. ACI with different learning rates
        for gamma in aci_gammas:
            engine_aci = EProcessEngine(
                cal_scores_safe=cal_scores,
                cal_lengths=cal_lengths,
                alpha=alpha,
                drift_profiler=drift_profiler,
                epsilon_multiplier=1,  # Use base ε_t
                use_aci=True,
                aci_gamma=gamma,
            )
            r_aci = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine_aci)
            label = f"aci_g{gamma}"
            results[str(alpha)][label] = {
                "prefix_fpr": r_aci.prefix_fpr,
                "power": r_aci.empirical_power,
                "leakage": r_aci.mean_leakage,
                "pir": r_aci.mean_pir,
                "final_alpha_t": engine_aci.aci.alpha_t if engine_aci.aci else None,
                "aci_empirical_fpr": engine_aci.aci.empirical_fpr if engine_aci.aci else None,
            }
            status = "✓" if r_aci.prefix_fpr <= alpha else "✗"
            print(f"  ACI(γ={gamma}): FPR={r_aci.prefix_fpr:.4f} Power={r_aci.empirical_power:.4f} "
                  f"Leak={r_aci.mean_leakage:.1f} α_T={engine_aci.aci.alpha_t:.4f} {status}")

    return results


def main():
    t0 = time.time()
    cal_scores, cal_lengths, drift_profiler, all_scores, all_is_harmful, all_harmful_onset = \
        load_cached_data()

    results = run_aci_sweep(cal_scores, cal_lengths, drift_profiler,
                            all_scores, all_is_harmful, all_harmful_onset)

    # Save
    output_file = OUTPUT_DIR / "m1_aci_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    print(f"\nResults saved to {output_file}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: FPR Control Comparison")
    print("=" * 80)
    print(f"{'α':<6} {'Method':<15} {'FPR':>8} {'Power':>8} {'Controlled?':>12}")
    print("-" * 52)
    for alpha_str, methods in results.items():
        alpha = float(alpha_str)
        for method, data in methods.items():
            ctrl = "YES" if data["prefix_fpr"] <= alpha else "NO"
            print(f"{alpha_str:<6} {method:<15} {data['prefix_fpr']:>8.4f} "
                  f"{data['power']:>8.4f} {ctrl:>12}")
        print()


if __name__ == "__main__":
    main()
