"""
M3 (Proper): BeaverTails cross-dataset validation with real probe.

Uses the trained RoBERTa probe to score BeaverTails samples at every
token position, then runs the full e-process pipeline.

Key challenge: BeaverTails has NO token-level annotations.
We address this by:
  - For safe samples: is_harmful=False, harmful_onset=None
  - For unsafe samples: is_harmful=True, harmful_onset estimated via
    probe score crossing a threshold (first position where score > 0.5)
"""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprocess import EProcessEngine, DriftProfiler, evaluate_batch
from baselines import (
    FixedThreshold, SlidingWindowMean, DelayK,
    evaluate_baseline_batch, grid_search_baseline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "outputs" / "score_cache"
OUTPUT_DIR = BASE_DIR / "outputs"
BEAVER_DIR = BASE_DIR / "data-BeaverTails"
MODEL_DIR = BASE_DIR / "outputs" / "best_probe"


def load_beavertails():
    """Load BeaverTails balanced test set."""
    print("Loading BeaverTails...")
    path = BEAVER_DIR / "beavertails_test_balanced.json"
    with open(path) as f:
        data = json.load(f)
    safe = sum(1 for d in data if d["is_safe"])
    unsafe = sum(1 for d in data if not d["is_safe"])
    print(f"  {len(data)} samples ({safe} safe, {unsafe} unsafe)")
    return data


def score_with_probe(samples):
    """
    Score each sample at every token position using the trained RoBERTa probe.

    Returns list of dicts with scores, is_harmful, harmful_onset, etc.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading probe model from {MODEL_DIR} (device={device})...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    print(f"  Model loaded. Scoring {len(samples)} samples...")

    scored = []
    for idx, item in enumerate(samples):
        if idx % 50 == 0:
            print(f"  Scoring sample {idx}/{len(samples)}...")

        tokens = item["response"].split()
        T = len(tokens)
        if T == 0:
            continue

        scores = np.zeros(T, dtype=np.float32)

        # Batch inference: score all prefixes
        batch_inputs = []
        batch_positions = []
        for t in range(1, T + 1):
            prefix_text = " ".join(tokens[:t])
            enc = tokenizer(
                item["prompt"], prefix_text,
                truncation="longest_first", max_length=512,
                padding="max_length", return_tensors="np",
            )
            batch_inputs.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
            })
            batch_positions.append(t - 1)

        # Process in mini-batches
        BATCH_SIZE = 64
        for start in range(0, len(batch_inputs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(batch_inputs))
            batch = batch_inputs[start:end]
            positions = batch_positions[start:end]

            input_ids = torch.tensor(np.array([b["input_ids"] for b in batch])).to(device)
            attention_mask = torch.tensor(np.array([b["attention_mask"] for b in batch])).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Model outputs single logit (regression-style), apply sigmoid
                probs = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
                if probs.ndim == 0:
                    probs = np.array([float(probs)])

            for pos, prob in zip(positions, probs):
                scores[pos] = prob

        is_harmful = not item["is_safe"]

        # Estimate harmful_onset from probe scores
        harmful_onset = None
        if is_harmful:
            # Find first position where probe score > 0.5
            onset_candidates = np.where(scores > 0.5)[0]
            if len(onset_candidates) > 0:
                harmful_onset = int(onset_candidates[0])
            else:
                # Probe never flagged — set onset to midpoint as conservative estimate
                harmful_onset = T // 2

        scored.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "scores": scores.tolist(),
            "is_harmful": is_harmful,
            "harmful_onset": harmful_onset,
            "total_tokens": T,
            "categories": item.get("categories", {}),
        })

    print(f"  Scored {len(scored)} samples")
    return scored


def run_experiment(scored, cal_scores, cal_lengths, dp):
    """Run Conformal-EADC and baselines on scored BeaverTails samples."""
    all_scores = [np.array(s["scores"], dtype=np.float32) for s in scored]
    all_is_harmful = [s["is_harmful"] for s in scored]
    all_harmful_onset = [s["harmful_onset"] for s in scored]

    print(f"\n=== BeaverTails Experiment ({len(all_scores)} samples) ===")
    print(f"  Harmful: {sum(all_is_harmful)}, Safe: {len(all_is_harmful) - sum(all_is_harmful)}")

    results = {}

    # Conformal-EADC with multiple configs
    for alpha in [0.05, 0.10, 0.20]:
        for eps_mult in [1, 2, 3]:
            engine = EProcessEngine(
                cal_scores_safe=cal_scores, cal_lengths=cal_lengths,
                alpha=alpha, drift_profiler=dp, epsilon_multiplier=eps_mult,
            )
            r = evaluate_batch(all_scores, all_is_harmful, all_harmful_onset, engine)
            label = f"ours_a{alpha}_eps{eps_mult}"
            results[label] = {
                "fpr": r.prefix_fpr, "power": r.empirical_power,
                "leakage": r.mean_leakage, "pir": r.mean_pir,
                "traditional_fpr": r.traditional_fpr,
            }
            ctrl = "YES" if r.prefix_fpr <= alpha else "NO"
            print(f"  Ours(α={alpha},ε×{eps_mult}): FPR={r.prefix_fpr:.4f} tFPR={r.traditional_fpr:.4f} "
                  f"Pow={r.empirical_power:.4f} Leak={r.mean_leakage:.1f} PIR={r.mean_pir:.3f} {ctrl}")

    # Baselines with grid search
    bl_configs = {
        "FixedThreshold": (FixedThreshold, {"threshold": [0.3, 0.5, 0.7, 0.9]}),
        "SWMA": (SlidingWindowMean, {"threshold": [0.3, 0.5, 0.7], "window": [3, 5, 7, 10]}),
        "DelayK": (DelayK, {"threshold": [0.3, 0.5, 0.7, 0.9], "k": [2, 3, 5]}),
    }
    for name, (cls, grid) in bl_configs.items():
        search = grid_search_baseline(cls, grid, all_scores, all_is_harmful, all_harmful_onset)
        best = search["best"]
        results[f"bl_{name}"] = best
        print(f"  {name}: FPR={best['fpr']:.4f} Pow={best['power']:.4f} "
              f"Leak={best['mean_leakage']:.1f} params={best['params']}")

    return results


def main():
    t0 = time.time()

    # 1. Load BeaverTails
    samples = load_beavertails()

    # 2. Score with real probe
    scored = score_with_probe(samples)

    # Cache scored data
    cache_path = OUTPUT_DIR / "beavertails_scored.json"
    with open(cache_path, "w") as f:
        json.dump(scored, f)
    print(f"  Cached scored data to {cache_path}")

    # 3. Load calibration data (from FineHarm — cross-dataset transfer)
    cal_scores = np.load(CACHE_DIR / "cal_safe_scores.npy")
    cal_lengths = np.load(CACHE_DIR / "cal_safe_lengths.npy")
    print(f"  Calibration: {len(cal_scores)} scores (from FineHarm)")

    with open(CACHE_DIR / "drift_profiler.json") as f:
        dd = json.load(f)
    dp = DriftProfiler()
    dp.epsilon_table = {int(k): v for k, v in dd["epsilon_table"].items()}
    dp.bucket_boundaries = np.array(dd["bucket_boundaries"])
    dp.n_buckets = dd["n_buckets"]

    # 4. Run experiment
    results = run_experiment(scored, cal_scores, cal_lengths, dp)

    # 5. Save
    output = {
        "dataset": "BeaverTails (PKU-Alignment, NeurIPS 2023)",
        "n_samples": len(scored),
        "n_safe": sum(1 for s in scored if not s["is_harmful"]),
        "n_unsafe": sum(1 for s in scored if s["is_harmful"]),
        "scoring_method": "RoBERTa probe (trained on FineHarm, zero-shot transfer)",
        "harmful_onset_method": "probe_estimated (first position where score > 0.5)",
        "calibration_source": "FineHarm training split (cross-dataset)",
        "results": results,
    }

    out_path = OUTPUT_DIR / "m3_beavertails_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    print(f"\nResults saved to {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
