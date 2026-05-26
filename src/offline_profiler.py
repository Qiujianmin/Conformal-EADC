"""
Offline score profiler: pre-compute all probe risk scores and drift parameters.

This is the first step before running any evaluation. It:
    1. Fine-tunes the probe (or loads a trained one)
    2. Pre-computes probe risk scores for EVERY token position of every sample
    3. Caches scores as .npy files for fast evaluation
    4. Computes drift profiling (ε_t) from benign validation data

Usage:
    # Step A: Train probe first
    python src/train_probe.py

    # Step B: Profile all scores
    python src/offline_profiler.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import (
    load_fineharm, split_train_calibration, StreamingSample, print_dataset_stats,
)
from eprocess import conformal_pvalue, DriftProfiler

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "score_cache"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def profile_single_sample(
    sample: StreamingSample,
    model,
    tokenizer,
    device,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute probe risk score at every token position for one sample.

    Returns array of shape (T,) where T = total_tokens.
    """
    model.eval()
    scores = np.zeros(sample.total_tokens, dtype=np.float32)

    with torch.no_grad():
        for t in range(1, sample.total_tokens + 1):
            prefix_text = " ".join(sample.tokens[:t])
            encoding = tokenizer(
                sample.prompt, prefix_text,
                truncation="longest_first",
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            prob = torch.sigmoid(logits).item()
            scores[t - 1] = prob

    return scores


def profile_batch(
    samples: list,
    model,
    tokenizer,
    device,
    max_length: int = 512,
    desc: str = "Profiling",
) -> dict:
    """
    Profile all samples and return cached results.

    Returns dict: {sample_idx: {"scores": np.ndarray, "is_harmful": bool, "harmful_onset": int|None}}
    """
    results = {}
    for sample in tqdm(samples, desc=desc):
        scores = profile_single_sample(sample, model, tokenizer, device, max_length)
        results[sample.idx] = {
            "scores": scores,
            "is_harmful": sample.is_harmful,
            "harmful_onset": sample.harmful_onset,
            "total_tokens": sample.total_tokens,
        }
    return results


def profile_calibration_scores(
    cal_samples: list,
    model,
    tokenizer,
    device,
    max_length: int = 512,
    n_prefixes_per_sample: int = 5,
) -> tuple:
    """
    Compute calibration scores at multiple prefix lengths for each safe sample.

    Returns:
        safe_scores: array of risk scores from safe calibration samples
        safe_lengths: corresponding prefix lengths
        harmful_scores: array of risk scores from harmful calibration samples
    """
    model.eval()
    safe_scores = []
    safe_lengths = []
    harmful_scores = []

    for sample in tqdm(cal_samples, desc="Calibration scores"):
        for ratio in np.linspace(0.2, 1.0, n_prefixes_per_sample):
            t = max(1, int(sample.total_tokens * ratio))
            prefix_text = " ".join(sample.tokens[:t])

            with torch.no_grad():
                encoding = tokenizer(
                    sample.prompt, prefix_text,
                    truncation="longest_first",
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                logit = model(input_ids=input_ids, attention_mask=attention_mask).logits
                prob = torch.sigmoid(logit).item()

            # Label for this prefix
            prefix_labels = sample.token_labels[:t]
            is_prefix_harmful = any(l == 1 for l in prefix_labels)

            if not is_prefix_harmful:
                safe_scores.append(prob)
                safe_lengths.append(t)
            else:
                harmful_scores.append(prob)

    return (
        np.array(safe_scores),
        np.array(safe_lengths),
        np.array(harmful_scores),
    )


def compute_drift_profiler(
    val_safe_scores: np.ndarray,
    val_safe_lengths: np.ndarray,
    cal_scores_safe: np.ndarray,
    cal_lengths: np.ndarray,
    n_buckets: int = 10,
    confidence_delta: float = 0.05,
) -> DriftProfiler:
    """Build the drift profiler from validation and calibration data."""
    profiler = DriftProfiler(
        n_buckets=n_buckets,
        confidence_delta=confidence_delta,
    )
    profiler.profile(
        val_safe_scores=val_safe_scores,
        val_safe_lengths=val_safe_lengths,
        cal_scores_safe=cal_scores_safe,
        cal_lengths=cal_lengths,
    )
    return profiler


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline score profiling")
    parser.add_argument("--probe-path", default=str(OUTPUT_DIR / "best_probe"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cal-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cal-prefixes", type=int, default=5,
                        help="Number of prefix positions per calibration sample")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("Loading FineHarm...")
    train_samples = load_fineharm("train")
    val_samples = load_fineharm("val")
    test_samples = load_fineharm("test")

    _, cal_samples = split_train_calibration(train_samples, cal_ratio=args.cal_ratio, seed=args.seed)

    # Load probe
    print(f"Loading probe from {args.probe_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.probe_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.probe_path)
    model = model.to(device)
    model.eval()

    # === Step 1: Compute calibration scores ===
    print("\n[Step 1/4] Computing calibration scores...")
    cal_safe_scores, cal_safe_lengths, cal_harmful_scores = profile_calibration_scores(
        cal_samples, model, tokenizer, device, args.max_length, args.n_cal_prefixes,
    )
    print(f"  Safe calibration: {len(cal_safe_scores)} scores")
    print(f"  Harmful calibration: {len(cal_harmful_scores)} scores")

    # Save calibration data
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "cal_safe_scores.npy", cal_safe_scores)
    np.save(CACHE_DIR / "cal_safe_lengths.npy", cal_safe_lengths)
    print(f"  Saved to {CACHE_DIR}/")

    # === Step 2: Compute validation scores (for drift profiling) ===
    print("\n[Step 2/4] Computing validation scores for drift profiling...")
    val_safe_samples = [s for s in val_samples if not s.is_harmful]
    # Sample a subset for efficiency
    rng = np.random.RandomState(args.seed)
    if len(val_safe_samples) > 200:
        idx = rng.choice(len(val_safe_samples), 200, replace=False)
        val_safe_subset = [val_safe_samples[i] for i in idx]
    else:
        val_safe_subset = val_safe_samples

    val_safe_scores_list = []
    val_safe_lengths_list = []
    for sample in tqdm(val_safe_subset, desc="Val safe profiling"):
        for ratio in [0.3, 0.5, 0.7, 1.0]:
            t = max(1, int(sample.total_tokens * ratio))
            prefix_text = " ".join(sample.tokens[:t])
            with torch.no_grad():
                encoding = tokenizer(
                    sample.prompt, prefix_text,
                    truncation="longest_first",
                    max_length=args.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                logit = model(input_ids=input_ids, attention_mask=attention_mask).logits
                prob = torch.sigmoid(logit).item()
            val_safe_scores_list.append(prob)
            val_safe_lengths_list.append(t)

    val_safe_scores = np.array(val_safe_scores_list)
    val_safe_lengths = np.array(val_safe_lengths_list)

    # === Step 3: Build drift profiler ===
    print("\n[Step 3/4] Building drift profiler...")
    drift_profiler = compute_drift_profiler(
        val_safe_scores, val_safe_lengths,
        cal_safe_scores, cal_safe_lengths,
    )
    print(f"  ε_t table: {drift_profiler.epsilon_table}")

    # Save drift profiler
    drift_data = {
        "epsilon_table": {str(k): v for k, v in drift_profiler.epsilon_table.items()},
        "bucket_boundaries": drift_profiler.bucket_boundaries.tolist() if drift_profiler.bucket_boundaries is not None else None,
        "n_buckets": drift_profiler.n_buckets,
        "confidence_delta": drift_profiler.confidence_delta,
    }
    with open(CACHE_DIR / "drift_profiler.json", "w") as f:
        json.dump(drift_data, f, indent=2)

    # === Step 4: Pre-compute test set scores ===
    print("\n[Step 4/4] Pre-computing test set scores (this takes a while)...")
    test_cache = {}
    for sample in tqdm(test_samples, desc="Test set profiling"):
        scores = profile_single_sample(sample, model, tokenizer, device, args.max_length)
        test_cache[str(sample.idx)] = {
            "scores": scores.tolist(),
            "is_harmful": sample.is_harmful,
            "harmful_onset": sample.harmful_onset,
            "total_tokens": sample.total_tokens,
        }

    # Save as compressed numpy + json metadata
    save_path = CACHE_DIR / "test_scores.json"
    print(f"Saving {len(test_cache)} samples to {save_path}...")
    with open(save_path, "w") as f:
        json.dump(test_cache, f)

    # Also save as numpy arrays for faster loading
    all_scores_arrays = []
    all_meta = []
    for idx_str, data in test_cache.items():
        all_scores_arrays.append(np.array(data["scores"]))
        all_meta.append({
            "idx": int(idx_str),
            "is_harmful": data["is_harmful"],
            "harmful_onset": data["harmful_onset"],
            "total_tokens": data["total_tokens"],
        })
    np.savez(CACHE_DIR / "test_scores.npz", **{
        f"s_{m['idx']}": s for s, m in zip(all_scores_arrays, all_meta)
    })
    with open(CACHE_DIR / "test_meta.json", "w") as f:
        json.dump(all_meta, f)

    print(f"\nAll cached data saved to {CACHE_DIR}/")
    print("Ready for streaming evaluation (run_streaming.py)")


if __name__ == "__main__":
    main()
