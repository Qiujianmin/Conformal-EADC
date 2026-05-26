"""
M5: Hyperparameter sensitivity analysis.

Sweeps each hyperparameter individually while holding others at default,
reporting FPR, Power, Leakage for each value.

Parameters: β, γ, C_max, ρ, κ_min, B (buckets), m (min_bucket_size)
"""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprocess import EProcessEngine, DriftProfiler, evaluate_batch

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "outputs" / "score_cache"
OUTPUT_DIR = BASE_DIR / "outputs"

# Default config (best from previous experiments)
DEFAULTS = {
    "alpha": 0.10,
    "epsilon_multiplier": 3,
    "kappa_min": 0.1,
    "ewma_beta": 0.9,
    "ewma_gamma": 1.0,
    "eadc_C_max": 10,
    "eadc_rho": 2.0,
    "t_min": 1,
}

# Parameter sweep ranges
SWEEPS = {
    "ewma_beta": [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    "ewma_gamma": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "eadc_C_max": [3, 5, 10, 15, 20, 30],
    "eadc_rho": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "kappa_min": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    "epsilon_multiplier": [1, 2, 3, 4, 5],
    "t_min": [1, 3, 5, 10, 15, 20],
}


def load_cached_data():
    print("Loading cached data...")
    cal_safe_scores = np.load(CACHE_DIR / "cal_safe_scores.npy")
    cal_safe_lengths = np.load(CACHE_DIR / "cal_safe_lengths.npy")

    with open(CACHE_DIR / "drift_profiler.json") as f:
        dd = json.load(f)
    dp = DriftProfiler()
    dp.epsilon_table = {int(k): v for k, v in dd["epsilon_table"].items()}
    dp.bucket_boundaries = np.array(dd["bucket_boundaries"])
    dp.n_buckets = dd["n_buckets"]
    dp.confidence_delta = dd.get("confidence_delta", 0.05)

    with open(CACHE_DIR / "test_scores.json") as f:
        td = json.load(f)

    all_scores, all_is_harmful, all_harmful_onset = [], [], []
    for item in td.values():
        all_scores.append(np.array(item["scores"], dtype=np.float32))
        all_is_harmful.append(item["is_harmful"])
        all_harmful_onset.append(item.get("harmful_onset"))

    print(f"  Test: {len(all_scores)} sequences")
    return cal_safe_scores, cal_safe_lengths, dp, all_scores, all_is_harmful, all_harmful_onset


def run_sensitivity(param_name, values, cal_scores, cal_lengths, dp,
                    all_scores, all_is_harmful, all_harmful_onset):
    """Sweep a single parameter."""
    results = []
    for val in values:
        kwargs = dict(DEFAULTS)
        # Map parameter name to constructor arg
        if param_name == "epsilon_multiplier":
            kwargs["epsilon_multiplier"] = val
        elif param_name == "eadc_C_max":
            kwargs["eadc_C_max"] = int(val)
        elif param_name == "eadc_rho":
            kwargs["eadc_rho"] = val
        elif param_name == "kappa_min":
            kwargs["kappa_min"] = val
        elif param_name == "ewma_beta":
            kwargs["ewma_beta"] = val
        elif param_name == "ewma_gamma":
            kwargs["ewma_gamma"] = val
        elif param_name == "t_min":
            kwargs["t_min"] = int(val)

        engine = EProcessEngine(
            cal_scores_safe=cal_scores,
            cal_lengths=cal_lengths,
            drift_profiler=dp,
            **kwargs,
        )
        r = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine)
        results.append({
            "value": val,
            "fpr": r.prefix_fpr,
            "power": r.empirical_power,
            "leakage": r.mean_leakage,
            "pir": r.mean_pir,
        })
        print(f"    {param_name}={val}: FPR={r.prefix_fpr:.4f} Pow={r.empirical_power:.4f} "
              f"Leak={r.mean_leakage:.1f} PIR={r.mean_pir:.3f}")
    return results


def main():
    t0 = time.time()
    cal_scores, cal_lengths, dp, all_scores, all_is_harmful, all_harmful_onset = \
        load_cached_data()

    print(f"\n=== M5: Hyperparameter Sensitivity Analysis ===\n")

    all_results = {}
    for param_name, values in SWEEPS.items():
        print(f"--- Sweeping {param_name} (values={values}) ---")
        results = run_sensitivity(param_name, values, cal_scores, cal_lengths, dp,
                                  all_scores, all_is_harmful, all_harmful_onset)
        all_results[param_name] = results
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Parameter':<20} {'Range':>20} {'FPR range':>15} {'Power range':>15}")
    print("-" * 70)
    for param_name, results in all_results.items():
        fprs = [r["fpr"] for r in results]
        pows = [r["power"] for r in results]
        vals = [str(r["value"]) for r in results]
        print(f"{param_name:<20} {vals[0]+'..'+vals[-1]:>20} "
              f"{min(fprs):.4f}..{max(fprs):.4f}   "
              f"{min(pows):.4f}..{max(pows):.4f}")

    # Save
    with open(OUTPUT_DIR / "m5_sensitivity_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    print(f"\nResults saved to {OUTPUT_DIR / 'm5_sensitivity_results.json'}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
