"""
Combined eps×3 + ACI experiment to fill missing data for Table 2 revision.

Runs the e-process with epsilon_multiplier=3 AND use_aci=True simultaneously,
which is the recommended deployment configuration.
"""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprocess import (
    EProcessEngine, DriftProfiler,
    evaluate_batch,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "outputs" / "score_cache"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_cached_data():
    print("Loading cached data...")
    cal_safe_scores = np.load(CACHE_DIR / "cal_safe_scores.npy")
    cal_safe_lengths = np.load(CACHE_DIR / "cal_safe_lengths.npy")

    with open(CACHE_DIR / "drift_profiler.json", "r") as f:
        drift_data = json.load(f)
    drift_profiler = DriftProfiler()
    drift_profiler.epsilon_table = {int(k): v for k, v in drift_data["epsilon_table"].items()}
    drift_profiler.bucket_boundaries = np.array(drift_data["bucket_boundaries"])
    drift_profiler.n_buckets = drift_data["n_buckets"]

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


def main():
    t0 = time.time()
    cal_scores, cal_lengths, drift_profiler, all_scores, all_is_harmful, all_harmful_onset = load_cached_data()

    results = {}

    configs = [
        # (alpha, eps_mult, aci_gamma, label)
        (0.05, 3, 0.05, "a0.05_eps3_aci_g0.05"),
        (0.05, 3, 0.01, "a0.05_eps3_aci_g0.01"),
        (0.10, 3, 0.01, "a0.10_eps3_aci_g0.01"),
        (0.10, 3, 0.05, "a0.10_eps3_aci_g0.05"),
        (0.10, 2, 0.05, "a0.10_eps2_aci_g0.05"),
        # Also standalone for Traditional FPR
        (0.05, 3, None, "a0.05_eps3"),
        (0.10, 3, None, "a0.10_eps3"),
        # Additional: α=0.05 ACI with different η for ACI table
        (0.05, 3, 0.01, "a0.05_eps3_aci_g0.01"),
    ]

    print("\n=== Combined eps×3 + ACI Experiments ===\n")
    print(f"{'Config':<30} {'Prefix-FPR':>12} {'Trad-FPR':>12} {'Power':>8} {'Leak':>8} {'PIR':>8}")
    print("-" * 80)

    for alpha, eps_mult, aci_gamma, label in configs:
        use_aci = aci_gamma is not None
        engine = EProcessEngine(
            cal_scores_safe=cal_scores,
            cal_lengths=cal_lengths,
            alpha=alpha,
            drift_profiler=drift_profiler,
            epsilon_multiplier=eps_mult,
            use_aci=use_aci,
            aci_gamma=aci_gamma if use_aci else 0.005,
        )
        r = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine)

        results[label] = {
            "alpha": alpha,
            "eps_mult": eps_mult,
            "aci_gamma": aci_gamma,
            "prefix_fpr": r.prefix_fpr,
            "traditional_fpr": r.traditional_fpr,
            "power": r.empirical_power,
            "leakage": r.mean_leakage,
            "pir": r.mean_pir,
            "n_fp": r.n_fp,
            "n_tp": r.n_tp,
            "n_fn": r.n_fn,
            "n_safe": r.n_safe,
            "n_harmful": r.n_harmful,
        }

        aci_str = f"ACI(η={aci_gamma})" if use_aci else "No ACI"
        ctrl = "✓" if r.prefix_fpr <= alpha else "✗"
        print(f"  α={alpha} ε×{eps_mult} {aci_str:<15} "
              f"{r.prefix_fpr:>8.4f}   {r.traditional_fpr:>8.4f}   "
              f"{r.empirical_power:>6.4f}  {r.mean_leakage:>6.1f}  {r.mean_pir:>6.3f}  {ctrl}")

    # Save
    output_file = OUTPUT_DIR / "combined_aci_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    print(f"\nResults saved to {output_file}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
