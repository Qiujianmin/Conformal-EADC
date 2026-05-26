"""
M4: Multi-seed experiments with statistical significance.

Runs the full pipeline with 5 different random seeds for:
  1. Data split (train/calibration)
  2. Reports mean±std for FPR, Power, Leakage
  3. McNemar test comparing Conformal-EADC vs best baseline

Requires: numpy, scipy, eprocess.py, baselines.py
"""

import sys, json, time, random
from pathlib import Path
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprocess import EProcessEngine, DriftProfiler, evaluate_batch
from baselines import (
    FixedThreshold, SlidingWindowMean, DelayK,
    evaluate_baseline_batch, grid_search_baseline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "outputs" / "score_cache"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_cached_data():
    """Load cached scores and metadata."""
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

    print(f"  Calibration: {len(cal_safe_scores)} scores")
    print(f"  Test: {len(all_scores)} sequences")
    return cal_safe_scores, cal_safe_lengths, dp, all_scores, all_is_harmful, all_harmful_onset


def mcnemar_test(predictions_a, predictions_b, labels):
    """
    McNemar's test for comparing two classifiers.

    Args:
        predictions_a: binary predictions from method A (correct/incorrect per sample)
        predictions_b: binary predictions from method B
        labels: ground truth labels

    Returns: (statistic, p_value)
    """
    correct_a = (predictions_a == labels)
    correct_b = (predictions_b == labels)

    # Discordant pairs
    n_01 = np.sum(~correct_a & correct_b)   # A wrong, B right
    n_10 = np.sum(correct_a & ~correct_b)   # A right, B wrong

    if n_01 + n_10 == 0:
        return 0.0, 1.0

    # Continuity-corrected McNemar
    statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return float(statistic), float(p_value)


def run_single_seed(seed, cal_scores, cal_lengths, dp,
                    all_scores, all_is_harmful, all_harmful_onset):
    """Run full evaluation with a specific seed (controls data shuffle)."""
    rng = np.random.RandomState(seed)

    # Shuffle test samples (simulates different random data splits)
    indices = list(range(len(all_scores)))
    rng.shuffle(indices)

    scores_shuffled = [all_scores[i] for i in indices]
    harmful_shuffled = [all_is_harmful[i] for i in indices]
    onsets_shuffled = [all_harmful_onset[i] for i in indices]

    # Subsample calibration scores (simulates different calibration split)
    n_cal = len(cal_scores)
    cal_indices = rng.choice(n_cal, size=int(n_cal * 0.8), replace=False)
    cal_sub = cal_scores[cal_indices]
    cal_len_sub = cal_lengths[cal_indices]

    # 1. Conformal-EADC (eps_x3, α=0.10)
    engine = EProcessEngine(
        cal_scores_safe=cal_sub, cal_lengths=cal_len_sub,
        alpha=0.10, drift_profiler=dp, epsilon_multiplier=3,
    )
    r_ours = evaluate_batch(scores_shuffled, harmful_shuffled, onsets_shuffled, engine)

    # 2. Best baseline: DelayK - run per-sample to get individual predictions
    delay_k = DelayK(threshold=0.9, k=5)
    bl_results = []
    for i in range(len(scores_shuffled)):
        r_bl = delay_k(scores_shuffled[i], harmful_shuffled[i], onsets_shuffled[i], i)
        bl_results.append(r_bl)

    # Build per-sample predictions for McNemar
    n = len(r_ours.results)
    our_preds = np.zeros(n, dtype=bool)
    bl_preds = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=bool)

    for i in range(n):
        res_ours = r_ours.results[i]
        res_bl = bl_results[i]
        # "Correct" = stopped on harmful and not on safe
        if harmful_shuffled[i]:
            labels[i] = True
            our_preds[i] = res_ours.is_true_positive
            bl_preds[i] = res_bl.is_true_positive
        else:
            labels[i] = False
            our_preds[i] = not res_ours.stopped
            bl_preds[i] = not res_bl.stopped

    return {
        "seed": seed,
        "ours": {
            "fpr": r_ours.prefix_fpr,
            "power": r_ours.empirical_power,
            "leakage": r_ours.mean_leakage,
            "pir": r_ours.mean_pir,
        },
        "baseline": {
            "fpr": sum(1 for r in bl_results if r.is_false_positive) / max(len(bl_results), 1),
            "power": sum(1 for r in bl_results if r.is_true_positive) / max(sum(1 for r in bl_results if r.is_harmful), 1),
            "leakage": float(np.mean([r.token_leakage for r in bl_results if r.is_harmful and r.is_true_positive and r.token_leakage is not None])) if any(r.is_harmful for r in bl_results) else None,
        },
        "mcnemar_predictions": {
            "ours": our_preds.tolist(),
            "baseline": bl_preds.tolist(),
            "labels": labels.tolist(),
        },
    }


def main():
    t0 = time.time()
    cal_scores, cal_lengths, dp, all_scores, all_is_harmful, all_harmful_onset = \
        load_cached_data()

    seeds = [42, 123, 456, 789, 2024]
    results = []

    print(f"\n=== M4: Multi-Seed Experiments (seeds={seeds}) ===\n")

    for seed in seeds:
        r = run_single_seed(seed, cal_scores, cal_lengths, dp,
                            all_scores, all_is_harmful, all_harmful_onset)
        results.append(r)
        print(f"  Seed {seed}: Ours FPR={r['ours']['fpr']:.4f} Pow={r['ours']['power']:.4f} "
              f"Leak={r['ours']['leakage']:.1f} | BL FPR={r['baseline']['fpr']:.4f} "
              f"Pow={r['baseline']['power']:.4f} Leak={r['baseline']['leakage']:.1f}")

    # Aggregate: mean ± std
    ours_fpr = [r["ours"]["fpr"] for r in results]
    ours_pow = [r["ours"]["power"] for r in results]
    ours_leak = [r["ours"]["leakage"] for r in results]
    ours_pir = [r["ours"]["pir"] for r in results]
    bl_fpr = [r["baseline"]["fpr"] for r in results]
    bl_pow = [r["baseline"]["power"] for r in results]
    bl_leak = [r["baseline"]["leakage"] for r in results]

    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (mean ± std over 5 seeds)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'FPR':>12} {'Power':>12} {'Leakage':>12} {'PIR':>12}")
    print("-" * 68)
    print(f"{'Conformal-EADC':<20} {np.mean(ours_fpr):.4f}±{np.std(ours_fpr):.4f} "
          f"{np.mean(ours_pow):.4f}±{np.std(ours_pow):.4f} "
          f"{np.mean(ours_leak):.1f}±{np.std(ours_leak):.1f} "
          f"{np.mean(ours_pir):.3f}±{np.std(ours_pir):.3f}")
    print(f"{'DelayK (best BL)':<20} {np.mean(bl_fpr):.4f}±{np.std(bl_fpr):.4f} "
          f"{np.mean(bl_pow):.4f}±{np.std(bl_pow):.4f} "
          f"{np.mean(bl_leak):.1f}±{np.std(bl_leak):.1f} "
          f"{'1.000':>12}")

    # McNemar test (pooled across seeds)
    all_ours_preds = np.concatenate([np.array(r["mcnemar_predictions"]["ours"]) for r in results])
    all_bl_preds = np.concatenate([np.array(r["mcnemar_predictions"]["baseline"]) for r in results])
    all_labels = np.concatenate([np.array(r["mcnemar_predictions"]["labels"]) for r in results])

    stat, p_val = mcnemar_test(all_ours_preds, all_bl_preds, all_labels)
    print(f"\nMcNemar test (Ours vs DelayK): χ²={stat:.2f}, p={p_val:.4f}")
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"Significance: {sig}")

    # Per-seed McNemar
    print(f"\nPer-seed McNemar:")
    for r in results:
        s, p = mcnemar_test(
            np.array(r["mcnemar_predictions"]["ours"]),
            np.array(r["mcnemar_predictions"]["baseline"]),
            np.array(r["mcnemar_predictions"]["labels"]),
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  Seed {r['seed']}: χ²={s:.2f}, p={p:.4f} {sig}")

    # Save
    output = {
        "seeds": seeds,
        "results": [{k: v for k, v in r.items() if k != "mcnemar_predictions"} for r in results],
        "aggregated": {
            "ours": {
                "fpr_mean": float(np.mean(ours_fpr)), "fpr_std": float(np.std(ours_fpr)),
                "power_mean": float(np.mean(ours_pow)), "power_std": float(np.std(ours_pow)),
                "leakage_mean": float(np.mean(ours_leak)), "leakage_std": float(np.std(ours_leak)),
                "pir_mean": float(np.mean(ours_pir)), "pir_std": float(np.std(ours_pir)),
            },
            "baseline": {
                "fpr_mean": float(np.mean(bl_fpr)), "fpr_std": float(np.std(bl_fpr)),
                "power_mean": float(np.mean(bl_pow)), "power_std": float(np.std(bl_pow)),
                "leakage_mean": float(np.mean(bl_leak)), "leakage_std": float(np.std(bl_leak)),
            },
        },
        "mcnemar": {"statistic": stat, "p_value": p_val, "significance": sig},
    }

    with open(OUTPUT_DIR / "m4_multiseed_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'm4_multiseed_results.json'}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
