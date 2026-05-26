"""
M2 Ablation Experiment: Baseline + EADC (fair compute comparison).

This standalone script tests whether EADC skipping alone can match
Conformal-EADC's compute reduction, or if the conformal e-process is essential.

Only requires: numpy, baselines.py
"""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import (
    FixedThreshold, SlidingWindowMean, DelayK, NaiveSPRT,
    evaluate_baseline_batch, grid_search_baseline, EADCWrappedBaseline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "outputs" / "score_cache"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_cached_data():
    """Load cached scores and metadata."""
    print("Loading cached data...")
    cal_safe_scores = np.load(CACHE_DIR / "cal_safe_scores.npy")
    cal_safe_lengths = np.load(CACHE_DIR / "cal_safe_lengths.npy")

    with open(CACHE_DIR / "test_scores.json", "r") as f:
        test_data = json.load(f)

    all_scores = []
    all_is_harmful = []
    all_harmful_onset = []

    # test_scores.json is a dict keyed by sample index
    items = test_data.values() if isinstance(test_data, dict) else test_data
    for item in items:
        scores = np.array(item["scores"], dtype=np.float32)
        all_scores.append(scores)
        all_is_harmful.append(item["is_harmful"])
        all_harmful_onset.append(item.get("harmful_onset"))

    print(f"  Calibration: {len(cal_safe_scores)} scores")
    print(f"  Test: {len(all_scores)} sequences "
          f"({sum(all_is_harmful)} harmful, {len(all_is_harmful) - sum(all_is_harmful)} safe)")
    return all_scores, all_is_harmful, all_harmful_onset


def run_baselines(all_scores, all_is_harmful, all_harmful_onset):
    """Run standard baselines with grid search."""
    print("\n=== Standard Baselines (Grid Search) ===")
    bl_configs = {
        "FixedThreshold": (FixedThreshold, {"threshold": [0.3, 0.5, 0.7, 0.9]}),
        "SWMA": (SlidingWindowMean, {"threshold": [0.3, 0.5, 0.7], "window": [3, 5, 7, 10]}),
        "DelayK": (DelayK, {"threshold": [0.3, 0.5, 0.7, 0.9], "k": [2, 3, 5]}),
    }
    bl_results = {}
    for name, (cls, grid) in bl_configs.items():
        search = grid_search_baseline(cls, grid, all_scores, all_is_harmful, all_harmful_onset)
        best = search["best"]
        bl_results[name] = {"best": best, "frontier": search["frontier"]}
        print(f"  {name}: FPR={best['fpr']:.4f} Power={best['power']:.4f} "
              f"Leakage={best['mean_leakage']:.1f} params={best['params']}")
    return bl_configs, bl_results


def run_eadc_ablation(all_scores, all_is_harmful, all_harmful_onset,
                      bl_configs, bl_results):
    """M2 Ablation: wrap baselines with EADC skipping."""
    print("\n=== M2 Ablation: Baseline + EADC ===")

    # Also test with multiple EADC parameter combinations
    eadc_param_grid = [
        {"C_max": 5, "rho": 1.5},
        {"C_max": 10, "rho": 2.0},
        {"C_max": 15, "rho": 2.0},
        {"C_max": 10, "rho": 3.0},
    ]

    eadc_bl_results = {}

    for name, (cls, grid) in bl_configs.items():
        best_params = bl_results[name]["best"]["params"]

        for eadc_params in eadc_param_grid:
            base_bl = cls(**best_params)
            eadc_bl = EADCWrappedBaseline(base_bl, **eadc_params)
            label = f"EADC(C={eadc_params['C_max']},ρ={eadc_params['rho']})+{name}"
            r = evaluate_baseline_batch(eadc_bl, all_scores, all_is_harmful, all_harmful_onset)

            # Compute PIR (Probe Invocation Ratio)
            total_tokens = sum(len(s) for s in all_scores)
            n_evals = 0
            for i, scores in enumerate(all_scores):
                # Re-run to count evals (approximate)
                T = len(scores)
                t = 1
                max_s = 0.0
                local_evals = 0
                while t <= T:
                    max_s = max(max_s, scores[t - 1])
                    local_evals += 1
                    delta = max(1, int(eadc_params['C_max'] * ((1.0 - max_s) ** eadc_params['rho'])))
                    t += delta
                n_evals += local_evals
            pir = n_evals / total_tokens

            eadc_bl_results[label] = {**r, "pir": pir}
            print(f"  {label}({best_params}): FPR={r['fpr']:.4f} Power={r['power']:.4f} "
                  f"Leakage={r['mean_leakage']:.1f} PIR={pir:.3f}")

    return eadc_bl_results


def main():
    t0 = time.time()
    all_scores, all_is_harmful, all_harmful_onset = load_cached_data()

    # Step 1: Standard baselines
    bl_configs, bl_results = run_baselines(all_scores, all_is_harmful, all_harmful_onset)

    # Step 2: EADC ablation
    eadc_results = run_eadc_ablation(all_scores, all_is_harmful, all_harmful_onset,
                                     bl_configs, bl_results)

    # Save results
    output = {
        "baselines": bl_results,
        "eadc_ablation": eadc_results,
        "elapsed_seconds": time.time() - t0,
    }

    # Custom JSON serializer for numpy types
    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_file = OUTPUT_DIR / "m2_ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=default_serializer)

    print(f"\nResults saved to {output_file}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Baseline vs EADC+Baseline")
    print("=" * 60)
    print(f"{'Method':<40} {'FPR':>8} {'Power':>8} {'Leakage':>8} {'PIR':>8}")
    print("-" * 72)
    for name, data in bl_results.items():
        b = data["best"]
        print(f"{name:<40} {b['fpr']:>8.4f} {b['power']:>8.4f} {b['mean_leakage']:>8.1f} {'1.000':>8}")
    for name, data in eadc_results.items():
        short_name = name[:40]
        print(f"{short_name:<40} {data['fpr']:>8.4f} {data['power']:>8.4f} "
              f"{data['mean_leakage']:>8.1f} {data['pir']:>8.3f}")


if __name__ == "__main__":
    main()
