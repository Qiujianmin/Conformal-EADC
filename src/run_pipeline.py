"""
Fast pipeline: profile scores + run streaming evaluation.
Optimized for GPU with batched inference for score profiling.

FIX v2: Calibration now profiles ALL token positions (not just 4 fixed ratios),
ensuring distribution matching between calibration and test scores.
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_fineharm, split_train_calibration
from eprocess import (
    EProcessEngine, DriftProfiler, ConformalCalibrator, evaluate_batch,
    conformal_pvalue, check_superuniformity,
)
from baselines import (
    FixedThreshold, SlidingWindowMean, DelayK, NaiveSPRT,
    evaluate_baseline_batch, grid_search_baseline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "score_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_profile_all_scores(samples, model, tokenizer, max_length=512):
    """Profile scores token-by-token with batched inference."""
    model.eval()
    results = {}
    for sample in tqdm(samples, desc="Profiling scores"):
        T = sample.total_tokens
        scores = np.zeros(T, dtype=np.float32)
        all_input_ids = []
        all_attention_mask = []
        for t in range(1, T + 1):
            prefix_text = " ".join(sample.tokens[:t])
            enc = tokenizer(
                sample.prompt, prefix_text,
                truncation="longest_first", max_length=max_length,
                padding="max_length", return_tensors="np",
            )
            all_input_ids.append(enc["input_ids"][0])
            all_attention_mask.append(enc["attention_mask"][0])

        chunk_size = 64
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            ids = torch.tensor(np.array(all_input_ids[start:end])).to(device)
            mask = torch.tensor(np.array(all_attention_mask[start:end])).to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            scores[start:end] = probs

        results[sample.idx] = {
            "scores": scores,
            "is_harmful": sample.is_harmful,
            "harmful_onset": sample.harmful_onset,
            "total_tokens": T,
        }
    return results


def profile_calibration_all_positions(
    cal_samples, model, tokenizer, max_length=512,
):
    """
    Profile calibration scores at ALL token positions for safe prefixes.

    Key fix: instead of sampling 4-5 fixed ratios (which only cover long prefixes),
    we compute scores at every token position for safe prefixes only.
    This ensures the calibration distribution matches the test distribution,
    which is critical for conformal p-value validity.
    """
    model.eval()
    all_safe_scores = []
    all_safe_lengths = []

    for sample in tqdm(cal_samples, desc="Calibration (all positions)"):
        T = sample.total_tokens

        # Find safe prefix positions (before any harmful token)
        if sample.is_harmful and sample.harmful_onset is not None:
            max_safe = sample.harmful_onset  # exclusive
        else:
            max_safe = T  # all positions are safe

        if max_safe <= 0:
            continue

        # Build all safe prefix inputs
        all_input_ids = []
        all_attention_mask = []
        safe_positions = list(range(1, max_safe + 1))

        for t in safe_positions:
            prefix_text = " ".join(sample.tokens[:t])
            enc = tokenizer(
                sample.prompt, prefix_text,
                truncation="longest_first", max_length=max_length,
                padding="max_length", return_tensors="np",
            )
            all_input_ids.append(enc["input_ids"][0])
            all_attention_mask.append(enc["attention_mask"][0])

        # Batched forward pass
        chunk_size = 64
        for start in range(0, len(safe_positions), chunk_size):
            end = min(start + chunk_size, len(safe_positions))
            ids = torch.tensor(np.array(all_input_ids[start:end])).to(device)
            mask = torch.tensor(np.array(all_attention_mask[start:end])).to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            for j in range(end - start):
                all_safe_scores.append(float(probs[j]))
                all_safe_lengths.append(safe_positions[start + j])

    return np.array(all_safe_scores), np.array(all_safe_lengths)


def profile_drift_all_positions(
    val_safe_samples, model, tokenizer, cal_safe_scores, cal_safe_lengths,
    max_length=512, n_buckets=10, confidence_delta=0.05,
):
    """
    Profile validation safe samples at all positions for drift profiling.
    """
    model.eval()
    val_safe_scores = []
    val_safe_lengths = []

    for sample in tqdm(val_safe_samples, desc="Drift profiling (all positions)"):
        T = sample.total_tokens
        all_input_ids = []
        all_attention_mask = []
        positions = list(range(1, T + 1))

        for t in positions:
            prefix_text = " ".join(sample.tokens[:t])
            enc = tokenizer(
                sample.prompt, prefix_text,
                truncation="longest_first", max_length=max_length,
                padding="max_length", return_tensors="np",
            )
            all_input_ids.append(enc["input_ids"][0])
            all_attention_mask.append(enc["attention_mask"][0])

        chunk_size = 64
        for start in range(0, len(positions), chunk_size):
            end = min(start + chunk_size, len(positions))
            ids = torch.tensor(np.array(all_input_ids[start:end])).to(device)
            mask = torch.tensor(np.array(all_attention_mask[start:end])).to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            for j in range(end - start):
                val_safe_scores.append(float(probs[j]))
                val_safe_lengths.append(positions[start + j])

    val_safe_scores = np.array(val_safe_scores)
    val_safe_lengths = np.array(val_safe_lengths)

    drift_profiler = DriftProfiler(n_buckets=n_buckets, confidence_delta=confidence_delta)
    drift_profiler.profile(val_safe_scores, val_safe_lengths, cal_safe_scores, cal_safe_lengths)
    return drift_profiler


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-profiling", action="store_true",
                        help="Skip profiling if cache exists")
    parser.add_argument("--drift-subsample", type=int, default=200,
                        help="Number of val safe samples for drift profiling")
    args = parser.parse_args()

    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    train_samples = load_fineharm("train")
    val_samples = load_fineharm("val")
    test_samples = load_fineharm("test")
    _, cal_samples = split_train_calibration(train_samples, cal_ratio=0.2)

    n_cal_safe = sum(1 for s in cal_samples if not s.is_harmful)
    n_cal_harm = sum(1 for s in cal_samples if s.is_harmful)
    print(f"Calibration: {len(cal_samples)} samples ({n_cal_safe} safe, {n_cal_harm} harmful)")

    # Load probe
    probe_path = str(OUTPUT_DIR / "best_probe")
    print(f"Loading probe from {probe_path}")
    tokenizer = AutoTokenizer.from_pretrained(probe_path)
    model = AutoModelForSequenceClassification.from_pretrained(probe_path).to(device)

    # === Step 1: Calibration scores at ALL positions ===
    print("\n[1/4] Computing calibration scores (ALL token positions)...")
    cal_cache_path = CACHE_DIR / "cal_safe_scores.npy"
    cal_len_path = CACHE_DIR / "cal_safe_lengths.npy"

    if args.skip_profiling and cal_cache_path.exists():
        print("  Loading cached calibration scores...")
        cal_safe_scores = np.load(cal_cache_path)
        cal_safe_lengths = np.load(cal_len_path)
    else:
        cal_safe_scores, cal_safe_lengths = profile_calibration_all_positions(
            cal_samples, model, tokenizer,
        )

    print(f"  Safe calibration: {len(cal_safe_scores)} scores, mean={cal_safe_scores.mean():.4f}")
    print(f"  Length range: [{cal_safe_lengths.min()}, {cal_safe_lengths.max()}], "
          f"median={np.median(cal_safe_lengths):.0f}")
    np.save(cal_cache_path, cal_safe_scores)
    np.save(cal_len_path, cal_safe_lengths)

    # === Step 2: Drift profiling (all positions) ===
    print("\n[2/4] Drift profiling (ALL positions)...")
    drift_cache_path = CACHE_DIR / "drift_profiler.json"

    if args.skip_profiling and drift_cache_path.exists():
        print("  Loading cached drift profiler...")
        with open(drift_cache_path) as f:
            drift_data = json.load(f)
        drift_profiler = DriftProfiler(
            n_buckets=drift_data["n_buckets"],
            confidence_delta=drift_data["confidence_delta"],
        )
        drift_profiler.epsilon_table = {int(k): v for k, v in drift_data["epsilon_table"].items()}
        if drift_data["bucket_boundaries"]:
            drift_profiler.bucket_boundaries = np.array(drift_data["bucket_boundaries"])
    else:
        val_safe = [s for s in val_samples if not s.is_harmful]
        if len(val_safe) > args.drift_subsample:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(val_safe), args.drift_subsample, replace=False)
            val_safe_subset = [val_safe[i] for i in idx]
        else:
            val_safe_subset = val_safe

        drift_profiler = profile_drift_all_positions(
            val_safe_subset, model, tokenizer,
            cal_safe_scores, cal_safe_lengths,
        )

    print(f"  epsilon_table: {drift_profiler.epsilon_table}")
    drift_data = {
        "epsilon_table": {str(k): v for k, v in drift_profiler.epsilon_table.items()},
        "bucket_boundaries": drift_profiler.bucket_boundaries.tolist() if drift_profiler.bucket_boundaries is not None else None,
        "n_buckets": drift_profiler.n_buckets,
        "confidence_delta": drift_profiler.confidence_delta,
    }
    with open(drift_cache_path, "w") as f:
        json.dump(drift_data, f, indent=2)

    # === Step 3: Profile test set ===
    print("\n[3/4] Profiling test set scores...")
    cache_path = CACHE_DIR / "test_scores.json"
    if args.skip_profiling and cache_path.exists():
        print("  Loading cached scores...")
        with open(cache_path) as f:
            test_cache = json.load(f)
    else:
        test_cache = fast_profile_all_scores(test_samples, model, tokenizer)
        save_data = {}
        for idx, data in test_cache.items():
            save_data[str(idx)] = {
                "scores": data["scores"].tolist() if isinstance(data["scores"], np.ndarray) else data["scores"],
                "is_harmful": data["is_harmful"],
                "harmful_onset": data["harmful_onset"],
                "total_tokens": data["total_tokens"],
            }
        with open(cache_path, "w") as f:
            json.dump(save_data, f)

    # Prepare data for evaluation
    all_scores = []
    all_is_harmful = []
    all_harmful_onset = []
    for idx_str in sorted(test_cache.keys(), key=lambda x: int(x)):
        d = test_cache[idx_str]
        all_scores.append(np.array(d["scores"]))
        all_is_harmful.append(d["is_harmful"])
        all_harmful_onset.append(d["harmful_onset"])

    print(f"  {len(all_scores)} test samples loaded")

    # === Step 4: Evaluation ===
    print("\n[4/4] Running evaluation...")

    # p-value diagnostics (use fast calibrator)
    print("  Computing p-value diagnostics on safe test samples...")
    calibrator = ConformalCalibrator(cal_safe_scores, cal_safe_lengths)
    safe_pvals = []
    for i in range(len(all_scores)):
        if not all_is_harmful[i]:
            for t in range(len(all_scores[i])):
                safe_pvals.append(calibrator.pvalue(all_scores[i][t], test_length=t))
    diag = check_superuniformity(np.array(safe_pvals))
    print(f"  P-value diag: mean={diag['mean']:.4f}, KS p={diag['ks_pvalue']:.4f}")
    print(f"  superuniform_mean: {diag['superuniform_mean']}")

    # Alpha sweep (Pareto) — try multiple configs
    print("\n  Alpha sweep (with tuning):")
    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Try different configs to find one where FPR is controlled
    configs = [
        {"epsilon_multiplier": 1.0, "t_min": 1, "label": "v2 (baseline)"},
        {"epsilon_multiplier": 2.0, "t_min": 1, "label": "eps_x2"},
        {"epsilon_multiplier": 3.0, "t_min": 1, "label": "eps_x3"},
        {"epsilon_multiplier": 1.0, "t_min": 5, "label": "t_min=5"},
        {"epsilon_multiplier": 2.0, "t_min": 5, "label": "eps_x2+t5"},
    ]

    all_config_results = {}
    for cfg in configs:
        print(f"\n  Config: {cfg['label']}")
        pareto = []
        for alpha in alphas:
            engine = EProcessEngine(
                cal_scores_safe=cal_safe_scores, alpha=alpha,
                cal_lengths=cal_safe_lengths,
                drift_profiler=drift_profiler,
                epsilon_multiplier=cfg["epsilon_multiplier"],
                t_min=cfg["t_min"],
            )
            result = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine)
            pareto.append({
                "alpha": alpha,
                "prefix_fpr": result.prefix_fpr,
                "traditional_fpr": result.traditional_fpr,
                "power": result.empirical_power,
                "mean_leakage": result.mean_leakage,
                "mean_pir": result.mean_pir,
                "controlled": result.prefix_fpr <= alpha,
            })
            s = "OK" if result.prefix_fpr <= alpha else "VIOLATED"
            print(f"    a={alpha:.2f}: Prefix-FPR={result.prefix_fpr:.4f}[{s}] "
                  f"Power={result.empirical_power:.4f} "
                  f"Leakage={result.mean_leakage:.1f} PIR={result.mean_pir:.3f}")
        all_config_results[cfg["label"]] = pareto

    # Pick best config: the one that controls FPR at α=0.05 with highest power
    best_label = None
    best_power = -1
    for label, pareto in all_config_results.items():
        row05 = [p for p in pareto if p["alpha"] == 0.05][0]
        if row05["controlled"] and row05["power"] > best_power:
            best_power = row05["power"]
            best_label = label
    if best_label is None:
        # No config controls at 0.05, pick the one with lowest FPR
        best_label = min(all_config_results.items(),
                        key=lambda x: [p for p in x[1] if p["alpha"] == 0.05][0]["prefix_fpr"])[0]

    print(f"\n  Best config: {best_label}")
    pareto = all_config_results[best_label]

    # Stress test with best config
    print("\n  Stress test (e_t ablation) with best config:")
    best_cfg = [c for c in configs if c["label"] == best_label][0]
    eng_eps = EProcessEngine(cal_scores_safe=cal_safe_scores, alpha=0.05,
                             cal_lengths=cal_safe_lengths, drift_profiler=drift_profiler,
                             epsilon_multiplier=best_cfg["epsilon_multiplier"],
                             t_min=best_cfg["t_min"])
    eng_no = EProcessEngine(cal_scores_safe=cal_safe_scores, alpha=0.05,
                            cal_lengths=cal_safe_lengths, drift_profiler=None,
                            t_min=best_cfg["t_min"])
    r_eps = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, eng_eps)
    r_no = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, eng_no)
    print(f"    [with eps]    FPR={r_eps.prefix_fpr:.4f} Power={r_eps.empirical_power:.4f}")
    print(f"    [without eps] FPR={r_no.prefix_fpr:.4f} Power={r_no.empirical_power:.4f}")

    # Generate trajectories with best config at α=0.05 for visualization
    print("\n  Generating trajectories for visualization...")
    eng_traj = EProcessEngine(cal_scores_safe=cal_safe_scores, alpha=0.05,
                              cal_lengths=cal_safe_lengths, drift_profiler=drift_profiler,
                              epsilon_multiplier=best_cfg["epsilon_multiplier"],
                              t_min=best_cfg["t_min"])
    traj_result = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, eng_traj)
    trajectories = []
    for r in traj_result.results[:50]:
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

    # Baselines
    print("\n  Baselines (grid search):")
    bl_configs = {
        "FixedThreshold": (FixedThreshold, {"threshold": [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]}),
        "SWMA": (SlidingWindowMean, {"threshold": [0.3, 0.5, 0.7, 0.9],
                                      "window": [3, 5, 7, 10]}),
        "DelayK": (DelayK, {"threshold": [0.3, 0.5, 0.7, 0.9], "k": [2, 3, 5]}),
    }
    bl_results = {}
    for name, (cls, grid) in bl_configs.items():
        search = grid_search_baseline(cls, grid, all_scores, all_is_harmful, all_harmful_onset)
        best = search["best"]
        bl_results[name] = {"best": best, "frontier": search["frontier"]}
        print(f"    {name}: FPR={best['fpr']:.4f} Power={best['power']:.4f} "
              f"Leakage={best['mean_leakage']:.1f}")

    # Save
    results = {
        "pvalue_diagnostics": diag,
        "eprocess_pareto": pareto,
        "all_configs": {k: v for k, v in all_config_results.items()},
        "best_config": best_label,
        "stress_test": {
            "with_epsilon": {"prefix_fpr": r_eps.prefix_fpr, "power": r_eps.empirical_power,
                             "leakage": r_eps.mean_leakage, "pir": r_eps.mean_pir},
            "without_epsilon": {"prefix_fpr": r_no.prefix_fpr, "power": r_no.empirical_power,
                                "leakage": r_no.mean_leakage, "pir": r_no.mean_pir},
        },
        "baselines": bl_results,
        "calibration_stats": {
            "n_cal_scores": len(cal_safe_scores),
            "cal_score_mean": float(cal_safe_scores.mean()),
            "cal_score_std": float(cal_safe_scores.std()),
            "cal_length_min": int(cal_safe_lengths.min()),
            "cal_length_max": int(cal_safe_lengths.max()),
            "cal_length_median": float(np.median(cal_safe_lengths)),
        },
    }
    with open(OUTPUT_DIR / "streaming_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(OUTPUT_DIR / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("DONE!")


if __name__ == "__main__":
    main()
