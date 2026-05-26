"""
Generate paper figures from experimental results.

Figures:
    1. p-value histogram + QQ plot
    2. Evidence trajectories (log scale)
    3. Pareto frontier: FPR vs Token Leakage (e-process vs baselines)
    4. Pareto frontier: PIR vs Token Leakage
    5. α sweep: FPR control verification
    6. Detection delay distribution

Usage:
    python src/run_streaming.py   # run first
    python src/visualize.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "ours": "#C44E52",
    "baseline1": "#4C72B0",
    "baseline2": "#55A868",
    "baseline3": "#8172B2",
    "baseline4": "#CCB974",
}


def load_results():
    with open(OUTPUT_DIR / "streaming_results.json") as f:
        results = json.load(f)
    traj_path = OUTPUT_DIR / "trajectories.json"
    if traj_path.exists():
        with open(traj_path) as f:
            trajectories = json.load(f)
    else:
        trajectories = []
        print("Warning: trajectories.json not found, skipping trajectory-based figures.")
    return results, trajectories


def fig1_pvalue_diagnostics(trajectories):
    """p-value histogram + QQ plot for safe samples."""
    safe_pvals = []
    for t in trajectories:
        if not t["is_harmful"]:
            safe_pvals.extend(t["p_values"])

    if not safe_pvals:
        print("No safe p-values for diagnostics.")
        return
    safe_pvals = np.array(safe_pvals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    ax = axes[0]
    ax.hist(safe_pvals, bins=30, density=True, alpha=0.7, color=COLORS["baseline1"],
            edgecolor="white", label="Observed")
    ax.axhline(y=1.0, color="red", linestyle="--", lw=1.5, label="Uniform(0,1)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("(a) P-value Histogram (Safe Samples, $\\mathcal{H}_0$)")
    ax.legend()

    # QQ plot
    ax = axes[1]
    sorted_p = np.sort(safe_pvals)
    n = len(sorted_p)
    theoretical = np.arange(1, n + 1) / (n + 1)
    ax.plot(theoretical, sorted_p, ".", ms=2, color=COLORS["baseline1"], alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y = x (Uniform)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed Quantiles")
    ax.set_title("(b) Q-Q Plot (Super-uniformity Check)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_pvalue_diagnostics.pdf")
    plt.savefig(FIG_DIR / "fig1_pvalue_diagnostics.png")
    plt.close()
    print("Saved fig1_pvalue_diagnostics")


def fig2_evidence_trajectories(trajectories):
    """Evidence trajectories (log scale) for harmful and safe samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    harmful = [t for t in trajectories if t["is_harmful"]][:8]
    safe = [t for t in trajectories if not t["is_harmful"]][:8]

    for ax, samples, title, color in [
        (axes[0], harmful, "(a) Harmful Samples", COLORS["ours"]),
        (axes[1], safe, "(b) Safe Samples", COLORS["baseline1"]),
    ]:
        for t in samples:
            log_e = t["log_evidence"]
            ax.plot(range(len(log_e)), log_e, alpha=0.7, lw=1.2, color=color)
            if t["stopped"]:
                ax.plot(len(log_e) - 1, log_e[-1], "rv", ms=8)
        ax.axhline(y=np.log(1.0 / 0.05), color="red", ls="--", lw=1.5,
                   label=r"$\ln(1/\alpha)$ ($\alpha=0.05$)")
        ax.axhline(y=0, color="gray", ls=":", alpha=0.3)
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel(r"$\ln \tilde{E}_t$ (Log Evidence)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_evidence_trajectories.pdf")
    plt.savefig(FIG_DIR / "fig2_evidence_trajectories.png")
    plt.close()
    print("Saved fig2_evidence_trajectories")


def fig3_pareto_fpr_leakage(results):
    """Pareto frontier: FPR vs Token Leakage."""
    eprocess = results.get("eprocess_pareto", [])
    if not eprocess:
        print("No e-process Pareto data.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # e-process
    fprs = [p.get("prefix_fpr", p.get("fpr", 0)) for p in eprocess]
    leaks = [p.get("mean_leakage", 0) or 0 for p in eprocess]
    ax.plot(fprs, leaks, "o-", color=COLORS["ours"], lw=2, ms=8,
            label="Conformal-EADC (Ours)", zorder=3)

    # Baselines
    bl_colors = [COLORS["baseline1"], COLORS["baseline2"], COLORS["baseline3"]]
    for (bl_name, bl_data), c in zip(results.get("baselines", {}).items(), bl_colors):
        frontier = bl_data.get("frontier", [])
        if frontier:
            bl_fprs = [f["fpr"] for f in frontier]
            bl_leaks = [f["mean_leakage"] or 0 for f in frontier]
            ax.plot(bl_fprs, bl_leaks, "s--", color=c, lw=1.5, ms=5,
                    alpha=0.7, label=bl_name)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Mean Token Leakage")
    ax.set_title("Pareto Frontier: FPR vs. Token Leakage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_pareto_fpr_leakage.pdf")
    plt.savefig(FIG_DIR / "fig3_pareto_fpr_leakage.png")
    plt.close()
    print("Saved fig3_pareto_fpr_leakage")


def fig4_pareto_pir_leakage(results):
    """Pareto frontier: PIR vs Token Leakage."""
    eprocess = results.get("eprocess_pareto", [])
    if not eprocess:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    pirs = [p.get("mean_pir", 0) or 0 for p in eprocess]
    leaks = [p.get("mean_leakage", 0) or 0 for p in eprocess]
    ax.plot(pirs, leaks, "o-", color=COLORS["ours"], lw=2, ms=8,
            label="Conformal-EADC (Ours)", zorder=3)

    # NaiveSPRT has fixed PIR = 1.0 (evaluates every token)
    ax.axvline(x=1.0, color=COLORS["baseline3"], ls="--", lw=1.5,
               alpha=0.5, label="NaiveSPRT (PIR=1.0)")

    ax.set_xlabel("Probe Invocation Ratio (PIR)")
    ax.set_ylabel("Mean Token Leakage")
    ax.set_title("Pareto Frontier: PIR vs. Token Leakage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_pareto_pir_leakage.pdf")
    plt.savefig(FIG_DIR / "fig4_pareto_pir_leakage.png")
    plt.close()
    print("Saved fig4_pareto_pir_leakage")


def fig5_alpha_sweep(results):
    """α sweep: FPR control + power."""
    eprocess = results.get("eprocess_pareto", [])
    if not eprocess:
        return

    alphas = [p["alpha"] for p in eprocess]
    fprs = [p.get("prefix_fpr", p.get("fpr", 0)) for p in eprocess]
    powers = [p.get("power", 0) for p in eprocess]
    controlled = [p.get("controlled", False) for p in eprocess]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(alphas, fprs, "bo-", lw=2, ms=8, label="Empirical FPR", zorder=3)
    ax1.plot(alphas, alphas, "r--", lw=1.5, label="Target α (Bound)", zorder=2)
    for a, f, c in zip(alphas, fprs, controlled):
        if not c:
            ax1.plot(a, f, "rx", ms=15, mew=3, zorder=4)

    ax1.set_xlabel("Target FPR Level (α)")
    ax1.set_ylabel("Empirical FPR")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(alphas, powers, "gs--", lw=1.5, ms=6, alpha=0.7, label="Power")
    ax2.set_ylabel("Detection Power", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="lower right")

    plt.title("Anytime FPR Control & Detection Power vs. α")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_alpha_sweep.pdf")
    plt.savefig(FIG_DIR / "fig5_alpha_sweep.png")
    plt.close()
    print("Saved fig5_alpha_sweep")


def fig6_detection_delay(trajectories):
    """Detection delay distribution (harmful detected samples only)."""
    delays = []
    for t in trajectories:
        if t["is_harmful"] and t["stopped"] and t["harmful_onset"] is not None:
            if t["stop_prefix_len"] is not None:
                delay = max(0, t["stop_prefix_len"] - t["harmful_onset"])
                delays.append(delay)

    if not delays:
        print("No delays found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(delays, bins=30, color=COLORS["baseline1"], edgecolor="white", alpha=0.7)
    ax.axvline(np.mean(delays), color="red", ls="--", lw=1.5,
               label=f"Mean={np.mean(delays):.1f}")
    ax.axvline(np.median(delays), color="orange", ls="--", lw=1.5,
               label=f"Median={np.median(delays):.1f}")
    ax.set_xlabel("Detection Delay (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Delay Distribution")
    ax.legend()

    ax = axes[1]
    sorted_d = np.sort(delays)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, lw=2, color=COLORS["baseline1"])
    for pct in [50, 90]:
        val = np.percentile(delays, pct)
        ax.axhline(pct / 100, color="gray", ls=":", alpha=0.5)
        ax.axvline(val, color="gray", ls=":", alpha=0.5)
        ax.annotate(f"P{pct}={val:.0f}", xy=(val, pct / 100), fontsize=9)
    ax.set_xlabel("Detection Delay (tokens)")
    ax.set_ylabel("CDF")
    ax.set_title("(b) Empirical CDF")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_detection_delay.pdf")
    plt.savefig(FIG_DIR / "fig6_detection_delay.png")
    plt.close()
    print("Saved fig6_detection_delay")


def main():
    try:
        results, trajectories = load_results()
    except FileNotFoundError as e:
        print(f"Results not found: {e}\nRun train_probe.py, offline_profiler.py, run_streaming.py first.")
        return

    print("Generating figures...")
    fig1_pvalue_diagnostics(trajectories)
    fig2_evidence_trajectories(trajectories)
    fig3_pareto_fpr_leakage(results)
    fig4_pareto_pir_leakage(results)
    fig5_alpha_sweep(results)
    fig6_detection_delay(trajectories)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
