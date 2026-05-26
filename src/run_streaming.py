"""
Step 2: Full streaming evaluation using cached scores.

Reads pre-computed scores from offline_profiler.py, runs:
    1. e-process with dynamic κ, ε_t, EADC (sweep α for Pareto)
    2. Baselines with grid search (fair comparison)
    3. Saves Pareto frontier data and trajectories

Usage:
    python src/offline_profiler.py   # run first to cache scores
    python src/run_streaming.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprocess import (
    EProcessEngine, DriftProfiler, evaluate_batch,
    conformal_pvalue, check_superuniformity,
)
from baselines import (
    FixedThreshold, SlidingWindowMean, DelayK, NaiveSPRT,
    evaluate_baseline_batch, grid_search_baseline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "score_cache"


def load_cached_data():
    """Load cached scores and calibration data."""
    # Calibration
    cal_safe_scores = np.load(CACHE_DIR / "cal_safe_scores.npy")
    cal_safe_lengths = np.load(CACHE_DIR / "cal_safe_lengths.npy")

    # Drift profiler
    with open(CACHE_DIR / "drift_profiler.json") as f:
        drift_data = json.load(f)
    drift_profiler = DriftProfiler(
        n_buckets=drift_data["n_buckets"],
        confidence_delta=drift_data["confidence_delta"],
    )
    drift_profiler.epsilon_table = {int(k): v for k, v in drift_data["epsilon_table"].items()}
    if drift_data["bucket_boundaries"]:
        drift_profiler.bucket_boundaries = np.array(drift_data["bucket_boundaries"])

    # Test scores
    with open(CACHE_DIR / "test_scores.json") as f:
        test_cache = json.load(f)

    all_scores = []
    all_is_harmful = []
    all_harmful_onset = []

    for idx_str in sorted(test_cache.keys(), key=lambda x: int(x)):
        data = test_cache[idx_str]
        all_scores.append(np.array(data["scores"]))
        all_is_harmful.append(data["is_harmful"])
        all_harmful_onset.append(data["harmful_onset"])

    return cal_safe_scores, cal_safe_lengths, drift_profiler, all_scores, all_is_harmful, all_harmful_onset


def run_pareto_sweep(
    all_scores, all_is_harmful, all_harmful_onset,
    cal_safe_scores, cal_safe_lengths, drift_profiler,
    alphas, kappa_min, ewma_beta, ewma_gamma,
    eadc_C_max, eadc_rho,
):
    """Sweep α values to build Pareto frontier for e-process."""
    pareto = []
    for alpha in alphas:
        engine = EProcessEngine(
            cal_scores_safe=cal_safe_scores,
            alpha=alpha,
            cal_lengths=cal_safe_lengths,
            kappa_min=kappa_min,
            ewma_beta=ewma_beta,
            ewma_gamma=ewma_gamma,
            drift_profiler=drift_profiler,
            eadc_C_max=eadc_C_max,
            eadc_rho=eadc_rho,
        )
        result = evaluate_batch(
            all_scores, all_is_harmful, all_harmful_onset, engine,
        )
        pareto.append({
            "alpha": alpha,
            "fpr": result.empirical_fpr,
            "power": result.empirical_power,
            "mean_leakage": result.mean_leakage,
            "median_leakage": result.median_leakage,
            "mean_pir": result.mean_pir,
            "median_pir": result.median_pir,
            "traditional_fpr": result.traditional_fpr,
            "controlled": result.prefix_fpr <= alpha,
            "n_safe": result.n_safe,
            "n_harmful": result.n_harmful,
            "n_fp": result.n_fp,
            "n_tp": result.n_tp,
            "n_fn": result.n_fn,
        })
        status = "OK" if result.prefix_fpr <= alpha else "VIOLATED"
        print(f"  α={alpha:.3f}: Prefix-FPR={result.prefix_fpr:.4f} "
              f"[{status}], Trad-FPR={result.traditional_fpr:.4f}, "
              f"Power={result.empirical_power:.4f}, "
              f"Leakage={result.mean_leakage:.1f}, PIR={result.mean_pir:.3f}")
    return pareto


def run_baseline_pareto(
    all_scores, all_is_harmful, all_harmful_onset,
):
    """Run baselines with grid search for fair comparison."""
    baseline_configs = {
        "FixedThreshold": {
            "class": FixedThreshold,
            "grid": {"threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]},
        },
        "SWMA": {
            "class": SlidingWindowMean,
            "grid": {
                "threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "window": [3, 5, 7, 10, 15],
            },
        },
        "DelayK": {
            "class": DelayK,
            "grid": {
                "threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "k": [2, 3, 5, 7],
            },
        },
    }

    baseline_results = {}
    for name, config in baseline_configs.items():
        print(f"\n  Grid search for {name}...")
        search_result = grid_search_baseline(
            config["class"],
            config["grid"],
            all_scores, all_is_harmful, all_harmful_onset,
        )
        baseline_results[name] = {
            "best": search_result["best"],
            "frontier": search_result["frontier"],
        }
        best = search_result["best"]
        print(f"    Best: {best['params']} → FPR={best['fpr']:.4f}, "
              f"Power={best['power']:.4f}, Leakage={best['mean_leakage']}")

    # Also add NaiveSPRT (no grid search needed, alpha sweep)
    sprt_results = []
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        bl = NaiveSPRT(alpha=alpha)
        r = evaluate_baseline_batch(bl, all_scores, all_is_harmful, all_harmful_onset)
        sprt_results.append(r)
        print(f"    NaiveSPRT(α={alpha}): FPR={r['fpr']:.4f}, Power={r['power']:.4f}")
    baseline_results["NaiveSPRT"] = {"results": sprt_results}

    return baseline_results


def main():
    parser = argparse.ArgumentParser(description="Streaming evaluation")
    parser.add_argument("--kappa-min", type=float, default=0.1)
    parser.add_argument("--ewma-beta", type=float, default=0.9)
    parser.add_argument("--ewma-gamma", type=float, default=1.0)
    parser.add_argument("--eadc-C-max", type=int, default=10)
    parser.add_argument("--eadc-rho", type=float, default=2.0)
    parser.add_argument("--alphas", type=str, default="0.01,0.02,0.05,0.10,0.15,0.20")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    # Load cached data
    print("Loading cached data...")
    (cal_safe_scores, cal_safe_lengths, drift_profiler,
     all_scores, all_is_harmful, all_harmful_onset) = load_cached_data()

    n_safe = sum(1 for h in all_is_harmful if not h)
    n_harmful = sum(1 for h in all_is_harmful if h)
    print(f"Test set: {len(all_scores)} samples ({n_safe} safe, {n_harmful} harmful)")
    print(f"Calibration: {len(cal_safe_scores)} safe scores")
    print(f"Drift profiler ε_t table: {drift_profiler.epsilon_table}")

    # === 1. p-value diagnostics ===
    print("\n[1/4] p-value diagnostics on safe samples...")
    safe_pvals = []
    for i in range(len(all_scores)):
        if not all_is_harmful[i]:
            for t in range(len(all_scores[i])):
                p_t = conformal_pvalue(all_scores[i][t], cal_safe_scores)
                safe_pvals.append(p_t)
    safe_pvals = np.array(safe_pvals)
    diag = check_superuniformity(safe_pvals)
    print(f"  n={diag['n_pvalues']}, mean={diag['mean']:.4f}, "
          f"KS p-value={diag['ks_pvalue']:.4f}")

    # === 2. e-process Pareto sweep ===
    print("\n[2/4] e-process Pareto sweep...")
    eprocess_pareto = run_pareto_sweep(
        all_scores, all_is_harmful, all_harmful_onset,
        cal_safe_scores, cal_safe_lengths, drift_profiler,
        alphas, args.kappa_min, args.ewma_beta, args.ewma_gamma,
        args.eadc_C_max, args.eadc_rho,
    )

    # === 3. Baseline comparison with grid search ===
    print("\n[3/5] Baseline comparison (grid search)...")
    baseline_results = run_baseline_pareto(
        all_scores, all_is_harmful, all_harmful_onset,
    )

    # === 4. Stress Test: with/without ε_t (validates DriftProfiler value) ===
    print("\n[4/5] Stress Test: ε_t ablation (with vs without drift correction)...")
    stress_alpha = 0.05

    # 4a. With ε_t (full system)
    engine_with_eps = EProcessEngine(
        cal_scores_safe=cal_safe_scores, alpha=stress_alpha,
        cal_lengths=cal_safe_lengths,
        kappa_min=args.kappa_min, ewma_beta=args.ewma_beta, ewma_gamma=args.ewma_gamma,
        drift_profiler=drift_profiler,
        eadc_C_max=args.eadc_C_max, eadc_rho=args.eadc_rho,
    )
    res_with_eps = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine_with_eps)

    # 4b. Without ε_t (degraded: drift_profiler=None)
    engine_no_eps = EProcessEngine(
        cal_scores_safe=cal_safe_scores, alpha=stress_alpha,
        cal_lengths=cal_safe_lengths,
        kappa_min=args.kappa_min, ewma_beta=args.ewma_beta, ewma_gamma=args.ewma_gamma,
        drift_profiler=None,  # <-- No drift correction
        eadc_C_max=args.eadc_C_max, eadc_rho=args.eadc_rho,
    )
    res_no_eps = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine_no_eps)

    print(f"  [With ε_t]    Prefix-FPR={res_with_eps.prefix_fpr:.4f}, Power={res_with_eps.empirical_power:.4f}, Leakage={res_with_eps.mean_leakage:.1f}")
    print(f"  [Without ε_t] Prefix-FPR={res_no_eps.prefix_fpr:.4f}, Power={res_no_eps.empirical_power:.4f}, Leakage={res_no_eps.mean_leakage:.1f}")
    eps_benefit = res_no_eps.prefix_fpr - res_with_eps.prefix_fpr
    print(f"  ε_t FPR reduction: {eps_benefit:+.4f} ({'HELPED' if eps_benefit > 0 else 'NO EFFECT'})")

    stress_test = {
        "with_epsilon": {
            "prefix_fpr": res_with_eps.prefix_fpr,
            "traditional_fpr": res_with_eps.traditional_fpr,
            "power": res_with_eps.empirical_power,
            "mean_leakage": res_with_eps.mean_leakage,
            "mean_pir": res_with_eps.mean_pir,
        },
        "without_epsilon": {
            "prefix_fpr": res_no_eps.prefix_fpr,
            "traditional_fpr": res_no_eps.traditional_fpr,
            "power": res_no_eps.empirical_power,
            "mean_leakage": res_no_eps.mean_leakage,
            "mean_pir": res_no_eps.mean_pir,
        },
        "epsilon_fpr_reduction": eps_benefit,
    }

    # === 5. Detailed run at primary α ===
    primary_alpha = 0.05
    print(f"\n[5/5] Detailed run at α={primary_alpha}...")
    engine = EProcessEngine(
        cal_scores_safe=cal_safe_scores,
        alpha=primary_alpha,
        cal_lengths=cal_safe_lengths,
        kappa_min=args.kappa_min,
        ewma_beta=args.ewma_beta,
        ewma_gamma=args.ewma_gamma,
        drift_profiler=drift_profiler,
        eadc_C_max=args.eadc_C_max,
        eadc_rho=args.eadc_rho,
    )
    detail_result = evaluate_batch(
        all_scores, all_is_harmful, all_harmful_onset, engine,
    )
    print(detail_result.summary())

    # Save trajectories (first 50 samples for visualization)
    trajectories = []
    for r in detail_result.results[:50]:
        trajectories.append({
            "sample_idx": r.sample_idx,
            "is_harmful": r.is_harmful,
            "harmful_onset": r.harmful_onset,
            "stopped": r.stopped,
            "stop_prefix_len": r.stop_prefix_len,
            "log_evidence": r.log_evidence_trajectory,
            "p_values": r.p_value_trajectory,
            "kappas": r.kappa_trajectory,
            "eval_lengths": r.eval_prefix_lengths,
            "n_evaluations": r.n_evaluations,
            "total_tokens": r.total_tokens,
            "pir": r.probe_invocation_ratio,
        })

    # Save everything
    save_data = {
        "pvalue_diagnostics": diag,
        "eprocess_pareto": eprocess_pareto,
        "baselines": {k: v for k, v in baseline_results.items()},
        "stress_test": stress_test,
        "detail": {
            "alpha": primary_alpha,
            "prefix_fpr": detail_result.prefix_fpr,
            "traditional_fpr": detail_result.traditional_fpr,
            "power": detail_result.empirical_power,
            "mean_leakage": detail_result.mean_leakage,
            "mean_pir": detail_result.mean_pir,
            "n_fp": detail_result.n_fp,
            "n_tp": detail_result.n_tp,
            "n_fn": detail_result.n_fn,
        },
        "args": vars(args),
    }

    with open(OUTPUT_DIR / "streaming_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    with open(OUTPUT_DIR / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Run visualize.py to generate paper figures.")


if __name__ == "__main__":
    main()
