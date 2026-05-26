"""
Baseline methods for streaming safety detection with grid search.

Each baseline: process cached risk scores (all token positions) and decide when to stop.

Methods:
    1. FixedThreshold: s_t >= θ → stop
    2. SlidingWindowMean: mean(s_{t-w:t}) >= θ → stop
    3. DelayK: s_t >= θ for k consecutive steps → stop
    4. NaiveSPRT: classical SPRT with normal approximation

All baselines support grid search over parameters for fair comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from itertools import product

import numpy as np


@dataclass
class BaselineResult:
    """Result from a baseline method on a single sample."""
    sample_idx: int
    stopped: bool
    stop_prefix_len: Optional[int]
    is_harmful: bool
    harmful_onset: Optional[int]
    total_tokens: int

    @property
    def is_false_positive(self) -> bool:
        if not self.stopped:
            return False
        if not self.is_harmful:
            return True
        if self.harmful_onset is not None and self.stop_prefix_len is not None:
            return self.stop_prefix_len < self.harmful_onset
        return False

    @property
    def is_true_positive(self) -> bool:
        if not self.stopped or not self.is_harmful:
            return False
        if self.harmful_onset is None:
            return True
        if self.stop_prefix_len is None:
            return False
        return self.stop_prefix_len >= self.harmful_onset

    @property
    def token_leakage(self) -> Optional[int]:
        if not self.stopped or not self.is_harmful:
            return None
        if self.harmful_onset is None or self.stop_prefix_len is None:
            return None
        return max(0, self.stop_prefix_len - self.harmful_onset)

    @property
    def missed(self) -> bool:
        return self.is_harmful and not self.stopped


# ============================================================
# Baseline implementations
# ============================================================

class FixedThreshold:
    """Stop when s_t >= threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, scores: np.ndarray, is_harmful: bool,
                 harmful_onset: Optional[int], sample_idx: int = 0) -> BaselineResult:
        for t in range(len(scores)):
            if scores[t] >= self.threshold:
                return BaselineResult(sample_idx, True, t + 1, is_harmful,
                                      harmful_onset, len(scores))
        return BaselineResult(sample_idx, False, None, is_harmful,
                              harmful_onset, len(scores))

    def __repr__(self):
        return f"FixedThreshold(θ={self.threshold})"


class SlidingWindowMean:
    """Stop when mean(s_{t-w:t}) >= threshold."""

    def __init__(self, threshold: float = 0.5, window: int = 5):
        self.threshold = threshold
        self.window = window

    def __call__(self, scores: np.ndarray, is_harmful: bool,
                 harmful_onset: Optional[int], sample_idx: int = 0) -> BaselineResult:
        for t in range(len(scores)):
            start = max(0, t - self.window + 1)
            if np.mean(scores[start:t + 1]) >= self.threshold:
                return BaselineResult(sample_idx, True, t + 1, is_harmful,
                                      harmful_onset, len(scores))
        return BaselineResult(sample_idx, False, None, is_harmful,
                              harmful_onset, len(scores))

    def __repr__(self):
        return f"SWMA(θ={self.threshold}, w={self.window})"


class DelayK:
    """Stop when s_t >= threshold for k consecutive steps."""

    def __init__(self, threshold: float = 0.5, k: int = 3):
        self.threshold = threshold
        self.k = k

    def __call__(self, scores: np.ndarray, is_harmful: bool,
                 harmful_onset: Optional[int], sample_idx: int = 0) -> BaselineResult:
        consecutive = 0
        for t in range(len(scores)):
            if scores[t] >= self.threshold:
                consecutive += 1
                if consecutive >= self.k:
                    return BaselineResult(sample_idx, True, t + 1, is_harmful,
                                          harmful_onset, len(scores))
            else:
                consecutive = 0
        return BaselineResult(sample_idx, False, None, is_harmful,
                              harmful_onset, len(scores))

    def __repr__(self):
        return f"DelayK(θ={self.threshold}, k={self.k})"


class NaiveSPRT:
    """Classical SPRT with normal approximation (not i.i.d. safe)."""

    def __init__(self, alpha: float = 0.05, mu0: float = 0.15, mu1: float = 0.7, sigma: float = 0.2):
        self.alpha = alpha
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma
        self.upper = np.log(1.0 / alpha)

    def __call__(self, scores: np.ndarray, is_harmful: bool,
                 harmful_onset: Optional[int], sample_idx: int = 0) -> BaselineResult:
        cum_llr = 0.0
        for t in range(len(scores)):
            s = scores[t]
            cum_llr += ((self.mu1 - self.mu0) / self.sigma ** 2) * (s - (self.mu0 + self.mu1) / 2)
            if cum_llr >= self.upper:
                return BaselineResult(sample_idx, True, t + 1, is_harmful,
                                      harmful_onset, len(scores))
            if cum_llr <= np.log(self.alpha):
                return BaselineResult(sample_idx, False, None, is_harmful,
                                      harmful_onset, len(scores))
        return BaselineResult(sample_idx, False, None, is_harmful,
                              harmful_onset, len(scores))

    def __repr__(self):
        return f"NaiveSPRT(α={self.alpha})"


# ============================================================
# Grid search
# ============================================================

def grid_search_baseline(
    baseline_class,
    param_grid: Dict[str, list],
    all_scores: List[np.ndarray],
    all_is_harmful: List[bool],
    all_harmful_onset: List[Optional[int]],
    target_fpr: Optional[float] = None,
) -> Dict:
    """
    Grid search over baseline parameters.

    If target_fpr is given, selects the parameter that achieves closest FPR
    while minimizing token leakage. Otherwise returns the full Pareto frontier.

    Returns dict with 'best' params and 'frontier' (list of (fpr, leakage, params)).
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    frontier = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        bl = baseline_class(**params)

        results = []
        for i in range(len(all_scores)):
            r = bl(all_scores[i], all_is_harmful[i], all_harmful_onset[i], i)
            results.append(r)

        n_total = len(results)
        n_fp = sum(1 for r in results if r.is_false_positive)
        n_tp = sum(1 for r in results if r.is_true_positive)
        n_harmful = sum(1 for r in results if r.is_harmful)

        fpr = n_fp / max(n_total, 1)
        power = n_tp / max(n_harmful, 1)
        # Leakage: TP + FN (missed samples leak all remaining harmful tokens)
        leakages = []
        for r in results:
            if r.is_harmful:
                if r.is_true_positive and r.token_leakage is not None:
                    leakages.append(r.token_leakage)
                elif r.missed and r.harmful_onset is not None:
                    leakages.append(r.total_tokens - r.harmful_onset)
        mean_leakage = float(np.mean(leakages)) if leakages else None

        frontier.append({
            "params": params,
            "fpr": fpr,
            "power": power,
            "mean_leakage": mean_leakage,
        })

    # Select best
    best = None
    if target_fpr is not None:
        # Find params closest to target FPR with best power
        valid = [f for f in frontier if f["fpr"] <= target_fpr * 1.5]
        if valid:
            best = max(valid, key=lambda f: f["power"])
        else:
            best = min(frontier, key=lambda f: abs(f["fpr"] - target_fpr))
    else:
        best = max(frontier, key=lambda f: f["power"])

    return {
        "best": best,
        "frontier": sorted(frontier, key=lambda f: f["fpr"]),
    }


# ============================================================
# Batch evaluation (single config)
# ============================================================

def evaluate_baseline_batch(
    baseline,
    all_scores: List[np.ndarray],
    all_is_harmful: List[bool],
    all_harmful_onset: List[Optional[int]],
) -> dict:
    """Evaluate a single baseline config on all samples."""
    results = []
    for i in range(len(all_scores)):
        r = baseline(all_scores[i], all_is_harmful[i], all_harmful_onset[i], i)
        results.append(r)

    n_total = len(results)
    n_safe = sum(1 for r in results if not r.is_harmful)
    n_harmful = sum(1 for r in results if r.is_harmful)
    n_fp = sum(1 for r in results if r.is_false_positive)
    n_tp = sum(1 for r in results if r.is_true_positive)

    fpr = n_fp / max(n_total, 1)
    power = n_tp / max(n_harmful, 1)
    # Leakage: TP + FN
    leakages = []
    for r in results:
        if r.is_harmful:
            if r.is_true_positive and r.token_leakage is not None:
                leakages.append(r.token_leakage)
            elif r.missed and r.harmful_onset is not None:
                leakages.append(r.total_tokens - r.harmful_onset)

    return {
        "method": str(baseline),
        "fpr": fpr,
        "power": power,
        "mean_leakage": float(np.mean(leakages)) if leakages else None,
        "median_leakage": float(np.median(leakages)) if leakages else None,
        "n_safe": n_safe,
        "n_harmful": n_harmful,
    }
