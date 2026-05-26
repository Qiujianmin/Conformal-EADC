"""
Core e-process algorithm for anytime-valid streaming safety detection.

Key components:
    1. Conformal p-value with length-conditioned calibration
    2. Dynamic κ via EWMA (predictable p-to-e conversion)
    3. Log-space evidence accumulation with DKW-bounded ε_t relaxation
    4. EADC dynamic chunking (evidence-aware evaluation scheduling)
    5. Ville's inequality for anytime-valid FPR control

References:
    [13] Vovk & Wang (2021). E-values: Calibration, combination and applications.
    [14] Ramdas et al. (2023). Game-theoretic statistics and safe anytime-valid inference.
    [26] Ville (1939). Étude critique de la notion de collectif.
    [60] Gibbs & Candès (2021). Adaptive conformal inference under distribution shift.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy import stats


# ============================================================
# 1. Conformal p-value (with length-conditioned calibration)
# ============================================================

class ConformalCalibrator:
    """
    Pre-computed conformal p-value engine with length-conditioned buckets.

    Pre-sorts calibration scores for O(log n) p-value computation,
    and pre-computes length bucket boundaries to avoid repeated np.percentile calls.
    """

    def __init__(
        self,
        cal_scores: np.ndarray,
        cal_lengths: Optional[np.ndarray] = None,
        n_buckets: int = 10,
        min_bucket_size: int = 30,
    ):
        # Full calibration set (sorted for binary search)
        self.cal_scores_sorted = np.sort(cal_scores)
        self.n_cal = len(cal_scores)

        # Length-conditioned buckets
        self.cal_lengths = cal_lengths
        self.n_buckets = n_buckets
        self.min_bucket_size = min_bucket_size
        self.bucket_boundaries = None
        self.bucket_scores_sorted: Dict[int, np.ndarray] = {}

        if cal_lengths is not None and len(cal_lengths) == len(cal_scores):
            self.bucket_boundaries = np.percentile(
                cal_lengths, np.linspace(0, 100, n_buckets + 1)
            )
            for b in range(n_buckets):
                lo = self.bucket_boundaries[b]
                hi = self.bucket_boundaries[min(b + 1, n_buckets)]
                mask = (cal_lengths >= lo) & (cal_lengths < hi)
                bucket = cal_scores[mask]
                if len(bucket) >= min_bucket_size:
                    self.bucket_scores_sorted[b] = np.sort(bucket)

    def pvalue(self, test_score: float, test_length: Optional[int] = None) -> float:
        """Compute conformal p-value using pre-sorted calibration scores."""
        if test_length is not None and self.bucket_boundaries is not None:
            bucket_idx = int(np.searchsorted(self.bucket_boundaries[1:-1], test_length))
            bucket_idx = min(bucket_idx, self.n_buckets - 1)
            if bucket_idx in self.bucket_scores_sorted:
                sorted_scores = self.bucket_scores_sorted[bucket_idx]
                n = len(sorted_scores)
                return (1.0 + n - np.searchsorted(sorted_scores, test_score)) / (n + 1.0)

        # Fallback to full calibration
        n = self.n_cal
        return (1.0 + n - np.searchsorted(self.cal_scores_sorted, test_score)) / (n + 1.0)


def conformal_pvalue(
    test_score: float,
    cal_scores: np.ndarray,
) -> float:
    """Right-tail conformal p-value (backward-compatible wrapper)."""
    return (1.0 + np.sum(cal_scores >= test_score)) / (len(cal_scores) + 1.0)


def length_conditioned_pvalue(
    test_score: float,
    cal_scores: np.ndarray,
    cal_lengths: np.ndarray,
    test_length: int,
    n_buckets: int = 10,
    min_bucket_size: int = 30,
) -> float:
    """Length-conditioned conformal p-value (backward-compatible wrapper)."""
    boundaries = np.percentile(cal_lengths, np.linspace(0, 100, n_buckets + 1))
    bucket_idx = np.searchsorted(boundaries[1:-1], test_length)

    lo, hi = boundaries[bucket_idx], boundaries[min(bucket_idx + 1, len(boundaries) - 1)]
    mask = (cal_lengths >= lo) & (cal_lengths < hi)
    bucket_scores = cal_scores[mask]

    if len(bucket_scores) < min_bucket_size:
        return conformal_pvalue(test_score, cal_scores)
    return conformal_pvalue(test_score, bucket_scores)


# ============================================================
# 2. Dynamic κ via EWMA (predictable calibrator)
# ============================================================

class DynamicKappa:
    """
    EWMA-based dynamic κ for the Vovk-Wang calibrator.

    κ_t is F_{t-1}-predictable: it depends only on past p-values.
    When consecutive p-values are small (risk accumulating), κ decreases
    → e-values grow faster → quicker detection.
    When p-values are large (safe), κ stays conservative → avoids noise-driven FP.

    Formula:
        μ_t = β * μ_{t-1} + (1-β) * p_t
        κ_{t+1} = max(κ_min, min(1, μ_t^γ))
    """

    def __init__(self, kappa_min: float = 0.1, beta: float = 0.9, gamma: float = 1.0):
        self.kappa_min = kappa_min
        self.beta = beta
        self.gamma = gamma
        self.mu = 0.5  # neutral initialization
        self.history: List[float] = []

    def get_current(self) -> float:
        """Get current κ (before seeing the next p-value)."""
        if not self.history:
            return max(self.kappa_min, self.mu ** self.gamma)
        return self.history[-1]

    def update(self, p_value: float) -> float:
        """Update EWMA with new p-value and compute next κ."""
        self.mu = self.beta * self.mu + (1 - self.beta) * p_value
        kappa = max(self.kappa_min, min(1.0, self.mu ** self.gamma))
        self.history.append(kappa)
        return kappa

    def reset(self):
        self.mu = 0.5
        self.history = []


# ============================================================
# 3. DKW-bounded ε_t drift profiler (offline)
# ============================================================

class DriftProfiler:
    """
    Offline profiler for computing ε_t via DKW inequality.

    On a held-out benign validation set, for each length bucket:
      1. Compute conformal p-values for safe validation samples
      2. Compute KS statistic (max empirical deviation from Uniform(0,1))
      3. Add DKW confidence bound: sqrt(ln(2/δ) / (2 * n_bucket))

    The resulting ε_t is stored as a lookup table indexed by bucket.
    This ε_t captures the "benign distribution drift" purely due to
    autoregressive length effects — it cannot be corrupted by attacks.
    """

    def __init__(
        self,
        n_buckets: int = 10,
        confidence_delta: float = 0.05,
        min_bucket_size: int = 20,
    ):
        self.n_buckets = n_buckets
        self.confidence_delta = confidence_delta
        self.min_bucket_size = min_bucket_size
        self.epsilon_table: Dict[int, float] = {}
        self.bucket_boundaries: Optional[np.ndarray] = None

    def profile(
        self,
        val_safe_scores: np.ndarray,
        val_safe_lengths: np.ndarray,
        cal_scores_safe: np.ndarray,
        cal_lengths: Optional[np.ndarray] = None,
    ):
        """
        Compute ε_t for each length bucket.

        Args:
            val_safe_scores: probe risk scores from safe validation samples
            val_safe_lengths: corresponding prefix lengths
            cal_scores_safe: calibration scores (safe only)
            cal_lengths: calibration score prefix lengths (for conditioned p-values)
        """
        all_lengths = np.concatenate([val_safe_lengths, cal_lengths]) if cal_lengths is not None else val_safe_lengths
        self.bucket_boundaries = np.percentile(
            all_lengths, np.linspace(0, 100, self.n_buckets + 1)
        )

        for b in range(self.n_buckets):
            lo = self.bucket_boundaries[b]
            hi = self.bucket_boundaries[min(b + 1, self.n_buckets)]
            mask = (val_safe_lengths >= lo) & (val_safe_lengths < hi)
            bucket_scores = val_safe_scores[mask]

            if len(bucket_scores) < self.min_bucket_size:
                self.epsilon_table[b] = self._compute_global_epsilon(
                    val_safe_scores, cal_scores_safe, cal_lengths
                )
                continue

            # Compute p-values for validation samples in this bucket
            p_values = np.array([
                conformal_pvalue(s, cal_scores_safe) if cal_lengths is None
                else length_conditioned_pvalue(s, cal_scores_safe, cal_lengths, int(lo), self.n_buckets)
                for s in bucket_scores
            ])

            # KS statistic: max |empirical CDF - Uniform CDF|
            ks_stat = np.max(np.abs(
                np.sort(p_values) - np.arange(1, len(p_values) + 1) / len(p_values)
            ))

            # DKW bound
            dkw_bound = np.sqrt(np.log(2.0 / self.confidence_delta) / (2.0 * len(bucket_scores)))

            self.epsilon_table[b] = ks_stat + dkw_bound

    def _compute_global_epsilon(
        self, val_scores: np.ndarray, cal_scores: np.ndarray,
        cal_lengths: Optional[np.ndarray] = None,
    ) -> float:
        """Fallback: compute ε over the entire validation set."""
        p_values = np.array([conformal_pvalue(s, cal_scores) for s in val_scores])
        ks_stat = np.max(np.abs(
            np.sort(p_values) - np.arange(1, len(p_values) + 1) / len(p_values)
        ))
        dkw_bound = np.sqrt(np.log(2.0 / self.confidence_delta) / (2.0 * len(val_scores)))
        return ks_stat + dkw_bound

    def get_epsilon(self, prefix_length: int) -> float:
        """Look up ε_t for a given prefix length."""
        if not self.epsilon_table or self.bucket_boundaries is None:
            return 0.0
        bucket_idx = int(np.searchsorted(self.bucket_boundaries[1:-1], prefix_length))
        bucket_idx = min(bucket_idx, self.n_buckets - 1)
        return self.epsilon_table.get(bucket_idx, 0.0)


# ============================================================
# 4. EADC: Evidence-Aware Dynamic Chunking
# ============================================================

class EADC:
    """
    Evidence-Aware Dynamic Chunking scheduler.

    Determines the next evaluation point based on accumulated evidence:
      - Safe phase (E_t ≈ 1):  evaluate every C_max tokens  →  low compute
      - Risk phase (E_t → 1/α): evaluate every token         →  zero leakage

    Formula:
        Δt_{k+1} = max(1, floor(C_max * (1 - ln(E_t)/ln(1/α))^ρ))

    Note (Theoretical limitation): The anytime-valid guarantee holds at
    evaluation points {t_k}, not at every token. A microscopic blind spot
    exists within Δt gaps during safe phases. However, the autoregressive
    continuity of LLMs makes instantaneous single-token jailbreaks
    practically impossible.
    """

    def __init__(self, C_max: int = 10, rho: float = 2.0):
        self.C_max = C_max
        self.rho = rho

    def compute_step(self, log_evidence: float, log_threshold: float) -> int:
        """
        Compute next evaluation interval.

        Args:
            log_evidence: ln(E_t), current accumulated evidence (log space)
            log_threshold: ln(1/α), the stopping threshold (log space)

        Returns:
            Number of tokens to skip before next evaluation.
        """
        if log_threshold <= 0:
            return 1
        ratio = max(0.0, 1.0 - log_evidence / log_threshold)
        step = max(1, int(self.C_max * (ratio ** self.rho)))
        return step

    @property
    def max_steps(self) -> int:
        return self.C_max


# ============================================================
# 5. Result dataclass
# ============================================================

@dataclass
class EProcessResult:
    """Result of running the e-process on a single streaming sequence."""
    sample_idx: int
    stopped: bool
    stop_prefix_len: Optional[int]
    is_harmful: bool
    harmful_onset: Optional[int]
    # Full trajectory (for visualization)
    log_evidence_trajectory: List[float]
    p_value_trajectory: List[float]
    kappa_trajectory: List[float]
    eval_prefix_lengths: List[int]
    eval_risk_scores: List[float]
    # EADC stats
    n_evaluations: int
    total_tokens: int

    @property
    def is_false_positive(self) -> bool:
        """
        Strict FP definition per the advisor's correction:
        - Safe sample stopped → FP
        - Harmful sample stopped before harmful onset (τ < t*) → also FP
        """
        if not self.stopped:
            return False
        if not self.is_harmful:
            return True
        # Harmful sample: check if stopped BEFORE harmful content appeared
        if self.harmful_onset is not None and self.stop_prefix_len is not None:
            return self.stop_prefix_len < self.harmful_onset
        return False

    @property
    def is_true_positive(self) -> bool:
        """
        Strict TP: stopped on a harmful sample, AND after (or at) harmful onset.
        """
        if not self.stopped or not self.is_harmful:
            return False
        if self.harmful_onset is None:
            return True  # no onset info, assume correct
        if self.stop_prefix_len is None:
            return False
        return self.stop_prefix_len >= self.harmful_onset

    @property
    def token_leakage(self) -> Optional[int]:
        """Harmful tokens leaked = max(0, stop_prefix - harmful_onset)."""
        if not self.stopped or not self.is_harmful:
            return None
        if self.harmful_onset is None or self.stop_prefix_len is None:
            return None
        return max(0, self.stop_prefix_len - self.harmful_onset)

    @property
    def missed(self) -> bool:
        """Harmful sample that was never caught (or caught before onset → FP)."""
        if not self.is_harmful:
            return False
        if not self.stopped:
            return True
        return False

    @property
    def probe_invocation_ratio(self) -> float:
        """PIR = n_evaluations / total_tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.n_evaluations / self.total_tokens


# ============================================================
# 6. E-Process Engine (complete pipeline)
# ============================================================

class EProcessEngine:
    """
    Complete e-process engine with:
      - Length-conditioned conformal p-values
      - Dynamic κ (EWMA-based, predictable)
      - DKW-bounded ε_t relaxation
      - Log-space evidence accumulation
      - EADC dynamic chunking
      - Ville's inequality stopping

    Works with pre-cached risk scores (from offline profiler).
    """

    def __init__(
        self,
        cal_scores_safe: np.ndarray,
        alpha: float = 0.05,
        cal_lengths: Optional[np.ndarray] = None,
        n_length_buckets: int = 10,
        # Dynamic κ
        kappa_min: float = 0.1,
        ewma_beta: float = 0.9,
        ewma_gamma: float = 1.0,
        # Drift
        drift_profiler: Optional[DriftProfiler] = None,
        epsilon_multiplier: float = 1.0,
        # EADC
        eadc_C_max: int = 10,
        eadc_rho: float = 2.0,
        # Bound
        log_e_bound: float = np.log(20.0),
        # Burn-in: skip first t_min tokens (probe unreliable for very short prefixes)
        t_min: int = 1,
    ):
        self.cal_scores = cal_scores_safe
        self.cal_lengths = cal_lengths
        self.n_length_buckets = n_length_buckets
        self.alpha = alpha
        self.log_threshold = -np.log(alpha)  # ln(1/α)
        self.log_e_bound = log_e_bound

        # Pre-computed calibrator (O(log n) p-values)
        self.calibrator = ConformalCalibrator(
            cal_scores_safe, cal_lengths,
            n_buckets=n_length_buckets,
        )

        # Sub-components
        self.drift_profiler = drift_profiler
        self.epsilon_multiplier = epsilon_multiplier
        self.eadc = EADC(C_max=eadc_C_max, rho=eadc_rho)

        # Store dynamic κ params for reset
        self.kappa_min = kappa_min
        self.ewma_beta = ewma_beta
        self.ewma_gamma = ewma_gamma

        # Burn-in period
        self.t_min = t_min

    def process_stream(
        self,
        all_scores: np.ndarray,
        is_harmful: bool,
        harmful_onset: Optional[int],
        sample_idx: int = 0,
    ) -> EProcessResult:
        """
        Run e-process on a single sample using EADC dynamic chunking.

        Args:
            all_scores: pre-cached risk scores at every token position, shape (T,)
            is_harmful: whether the full response is harmful
            harmful_onset: token index of first harmful token (None if safe)
            sample_idx: sample identifier

        Returns:
            EProcessResult with trajectory and decision.
        """
        T = len(all_scores)
        if T == 0:
            return EProcessResult(
                sample_idx=sample_idx, stopped=False, stop_prefix_len=None,
                is_harmful=is_harmful, harmful_onset=harmful_onset,
                log_evidence_trajectory=[0.0], p_value_trajectory=[],
                kappa_trajectory=[], eval_prefix_lengths=[],
                eval_risk_scores=[], n_evaluations=0, total_tokens=0,
            )

        # Initialize per-stream state
        kappa_ctrl = DynamicKappa(
            kappa_min=self.kappa_min,
            beta=self.ewma_beta,
            gamma=self.ewma_gamma,
        )
        log_evidence = 0.0  # ln(E_0) = 0
        log_evidence_trajectory = [0.0]
        p_value_trajectory = []
        kappa_trajectory = []
        eval_prefix_lengths = []
        eval_risk_scores = []

        stopped = False
        stop_prefix_len = None
        n_evaluations = 0

        t = self.t_min  # start from burn-in threshold
        if t < 1:
            t = 1

        while t <= T:
            # EADC: evaluate at position t
            score_t = all_scores[t - 1]

            # 1. Conformal p-value (pre-computed calibrator, O(log n))
            if self.cal_lengths is not None:
                p_t = self.calibrator.pvalue(score_t, test_length=t)
            else:
                p_t = self.calibrator.pvalue(score_t)
            p_value_trajectory.append(p_t)

            # 2. Dynamic κ (predictable: uses current κ before update)
            kappa_t = kappa_ctrl.get_current()
            kappa_trajectory.append(kappa_t)

            # Update κ for next step
            kappa_ctrl.update(p_t)

            # 3. p-to-e conversion in log space
            #    e_t = κ * p^(κ-1) → log_e_t = log(κ) + (κ-1)*log(p)
            p_clipped = max(p_t, 1e-15)
            log_e_t = np.log(kappa_t) + (kappa_t - 1.0) * np.log(p_clipped)

            # Bound e-value in log space
            log_e_t = min(log_e_t, self.log_e_bound)

            # 4. Drift correction (ε_t) with safety multiplier
            epsilon_t = 0.0
            if self.drift_profiler is not None:
                epsilon_t = self.drift_profiler.get_epsilon(t) * self.epsilon_multiplier
            # Adjusted log_e_t: ln(e_t / (1 + ε_t))
            if epsilon_t > 0:
                log_e_t -= np.log(1.0 + epsilon_t)

            # 5. Accumulate in log space
            log_evidence += log_e_t
            log_evidence_trajectory.append(log_evidence)

            eval_prefix_lengths.append(t)
            eval_risk_scores.append(float(score_t))
            n_evaluations += 1

            # 6. Stopping check (Ville): ln(E_t) >= ln(1/α)
            if log_evidence >= self.log_threshold:
                stopped = True
                stop_prefix_len = t
                break

            # 7. EADC: compute next evaluation point
            step = self.eadc.compute_step(log_evidence, self.log_threshold)
            t += step

        return EProcessResult(
            sample_idx=sample_idx,
            stopped=stopped,
            stop_prefix_len=stop_prefix_len,
            is_harmful=is_harmful,
            harmful_onset=harmful_onset,
            log_evidence_trajectory=log_evidence_trajectory,
            p_value_trajectory=p_value_trajectory,
            kappa_trajectory=kappa_trajectory,
            eval_prefix_lengths=eval_prefix_lengths,
            eval_risk_scores=eval_risk_scores,
            n_evaluations=n_evaluations,
            total_tokens=T,
        )


# ============================================================
# 7. Batch evaluation
# ============================================================

@dataclass
class BatchEvalResult:
    """Aggregate evaluation results."""
    results: List[EProcessResult]
    alpha: float
    n_safe: int
    n_harmful: int
    # Two FPR definitions (see paper Evaluation Metrics section)
    prefix_fpr: float           # Anytime-Valid Prefix-FPR = FP / total_sequences
    traditional_fpr: float      # Traditional FPR = FP_safe / safe_sequences
    empirical_power: float
    mean_leakage: Optional[float]
    median_leakage: Optional[float]
    mean_pir: Optional[float]
    median_pir: Optional[float]
    n_fp: int
    n_tp: int
    n_fn: int

    @property
    def empirical_fpr(self) -> float:
        """Alias: report Prefix-FPR as the primary metric (anytime-valid perspective)."""
        return self.prefix_fpr

    def summary(self) -> str:
        lines = [
            f"α={self.alpha}:",
            f"  Prefix-FPR (Anytime-Valid) = {self.prefix_fpr:.4f} "
            f"({'CONTROLLED' if self.prefix_fpr <= self.alpha else 'VIOLATED'})",
            f"  Traditional FPR = {self.traditional_fpr:.4f}",
            f"  Detection Power = {self.empirical_power:.4f}",
            f"  FP={self.n_fp}, TP={self.n_tp}, FN={self.n_fn} | "
            f"Safe={self.n_safe}, Harmful={self.n_harmful}",
        ]
        if self.mean_leakage is not None:
            lines.append(f"  Token Leakage: mean={self.mean_leakage:.1f}, median={self.median_leakage:.1f}")
        if self.mean_pir is not None:
            lines.append(f"  PIR: mean={self.mean_pir:.3f}, median={self.median_pir:.3f}")
        return "\n".join(lines)


def evaluate_batch(
    all_scores: List[np.ndarray],
    all_is_harmful: List[bool],
    all_harmful_onset: List[Optional[int]],
    engine: EProcessEngine,
) -> BatchEvalResult:
    """
    Evaluate e-process on a batch of samples with cached scores.

    FPR definitions:
        Prefix-FPR (Anytime-Valid): FP / total_sequences
            Every harmful sequence has a safe prefix before t*, so an early
            stop on that prefix is a Type I error. Denominator = all sequences.
        Traditional FPR: FP_safe / safe_sequences
            The conventional security metric. Only counts FP on purely safe samples.
    """
    results = []
    for i in range(len(all_scores)):
        result = engine.process_stream(
            all_scores=all_scores[i],
            is_harmful=all_is_harmful[i],
            harmful_onset=all_harmful_onset[i],
            sample_idx=i,
        )
        results.append(result)

    n_safe = sum(1 for r in results if not r.is_harmful)
    n_harmful = sum(1 for r in results if r.is_harmful)
    n_total = n_safe + n_harmful

    # FP/TP/FN counts
    n_fp = sum(1 for r in results if r.is_false_positive)
    n_tp = sum(1 for r in results if r.is_true_positive)
    n_fn = sum(1 for r in results if r.missed)

    # Prefix-FPR (anytime-valid): FP / total_sequences
    prefix_fpr = n_fp / max(n_total, 1)

    # Traditional FPR: FP on safe samples only / safe samples
    n_fp_safe = sum(1 for r in results if r.is_false_positive and not r.is_harmful)
    traditional_fpr = n_fp_safe / max(n_safe, 1)

    # Power: TP / harmful
    power = n_tp / max(n_harmful, 1)

    # Leakage: TP leakage + FN leakage (T - t* for missed samples)
    leakages = []
    for r in results:
        if r.is_harmful:
            if r.is_true_positive and r.token_leakage is not None:
                # TP: leakage = stop_prefix - harmful_onset
                leakages.append(r.token_leakage)
            elif r.missed and r.harmful_onset is not None:
                # FN: all harmful tokens leaked = T - t*
                leakages.append(r.total_tokens - r.harmful_onset)

    pirs = [r.probe_invocation_ratio for r in results if r.total_tokens > 0]

    return BatchEvalResult(
        results=results,
        alpha=engine.alpha,
        n_safe=n_safe,
        n_harmful=n_harmful,
        prefix_fpr=prefix_fpr,
        traditional_fpr=traditional_fpr,
        empirical_power=power,
        mean_leakage=float(np.mean(leakages)) if leakages else None,
        median_leakage=float(np.median(leakages)) if leakages else None,
        mean_pir=float(np.mean(pirs)) if pirs else None,
        median_pir=float(np.median(pirs)) if pirs else None,
        n_fp=n_fp,
        n_tp=n_tp,
        n_fn=n_fn,
    )


# ============================================================
# 8. p-value diagnostics
# ============================================================

def check_superuniformity(p_values: np.ndarray) -> dict:
    """
    Check if p-values satisfy super-uniformity under H0.

    KS test + proportion test at various quantiles.
    """
    ks_stat, ks_pval = stats.kstest(p_values, "uniform")

    quantile_checks = {}
    for u in [0.01, 0.05, 0.1, 0.2, 0.5]:
        observed = np.mean(p_values <= u)
        quantile_checks[f"P(p<={u})"] = {
            "observed": float(observed),
            "bound": u,
            "satisfied": observed <= u + 0.02,
        }

    return {
        "n_pvalues": len(p_values),
        "mean": float(np.mean(p_values)),
        "median": float(np.median(p_values)),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "superuniform_mean": bool(np.mean(p_values) >= 0.5),
        "quantile_checks": quantile_checks,
    }
