"""Causal inference helpers for the Memory resurfacing nudge experiment."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ── DiD ──────────────────────────────────────────────────────────────────────

def compute_did(df: pd.DataFrame, baseline_rate: float = 0.30, n_bootstrap: int = 1000, seed: int = 99):
    """
    Simulates a pre/post DiD design.
    Pre-period: baseline re-activation rate (before nudge experiment).
    Post-period: re-activation rate after Memory nudge.
    """
    cold = df[df["went_cold"] == 1].copy()
    if cold.empty:
        raise ValueError("No cold pairs available for DiD estimation.")

    # Simulate pre-period re-activation rate (before any nudge)
    # Treated group had same baseline as control (parallel trends assumption)
    rng = np.random.default_rng(seed)
    pre_rate_control = baseline_rate
    pre_rate_treatment = baseline_rate   # parallel trends

    # Post-period rates
    post_rate_control   = cold[cold["treatment"] == 0]["reactivated"].mean()
    post_rate_treatment = cold[cold["treatment"] == 1]["reactivated"].mean()

    did_estimate = (post_rate_treatment - pre_rate_treatment) - (post_rate_control - pre_rate_control)

    # Standard error via bootstrap.
    did_samples = []
    for _ in range(n_bootstrap):
        sample = cold.iloc[rng.integers(0, len(cold), size=len(cold))]
        post_t = sample[sample["treatment"] == 1]["reactivated"].mean()
        post_c = sample[sample["treatment"] == 0]["reactivated"].mean()
        did_samples.append((post_t - pre_rate_treatment) - (post_c - pre_rate_control))

    se = np.std(did_samples, ddof=1)
    ci_lower = did_estimate - 1.96 * se
    ci_upper = did_estimate + 1.96 * se
    t_stat = did_estimate / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    results = {
        "pre_rate_treatment":  pre_rate_treatment,
        "pre_rate_control":    pre_rate_control,
        "post_rate_treatment": post_rate_treatment,
        "post_rate_control":   post_rate_control,
        "did_estimate":        did_estimate,
        "se":                  se,
        "ci_lower":            ci_lower,
        "ci_upper":            ci_upper,
        "t_stat":              t_stat,
        "p_value":             p_value,
        "n_treatment":         (cold["treatment"] == 1).sum(),
        "n_control":           (cold["treatment"] == 0).sum(),
    }
    return results


def print_did_results(r: dict):
    print("\n" + "=" * 50)
    print("  DIFFERENCE-IN-DIFFERENCES RESULTS")
    print("=" * 50)
    print(f"  Pre-period re-activation (treatment): {r['pre_rate_treatment']:.1%}")
    print(f"  Pre-period re-activation (control):   {r['pre_rate_control']:.1%}")
    print(f"  Post-period re-activation (treatment):{r['post_rate_treatment']:.1%}")
    print(f"  Post-period re-activation (control):  {r['post_rate_control']:.1%}")
    print(f"  ─────────────────────────────────────")
    print(f"  DiD Estimate:  {r['did_estimate']:+.1%}")
    print(f"  95% CI:       [{r['ci_lower']:+.1%}, {r['ci_upper']:+.1%}]")
    print(f"  p-value:       {r['p_value']:.4f}")
    print(f"  n (treatment): {r['n_treatment']:,}")
    print(f"  n (control):   {r['n_control']:,}")
    print("=" * 50)


# ── Propensity Score Matching ─────────────────────────────────────────────────

def propensity_score_matching(df: pd.DataFrame, covariate_cols: list):
    cold = df[df["went_cold"] == 1].copy().reset_index(drop=True)
    if cold.empty:
        raise ValueError("No cold pairs available for propensity score matching.")

    scaler = StandardScaler()
    X = scaler.fit_transform(cold[covariate_cols].fillna(0))

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, cold["treatment"])
    cold["propensity"] = lr.predict_proba(X)[:, 1]

    treated = cold[cold["treatment"] == 1].copy()
    control = cold[cold["treatment"] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(control[["propensity"]])
    distances, indices = nn.kneighbors(treated[["propensity"]])
    matched_controls = control.iloc[indices.ravel()].reset_index(drop=True)

    matched_df = pd.DataFrame({
        "treated_reactivated": treated["reactivated"].reset_index(drop=True),
        "control_reactivated": matched_controls["reactivated"].reset_index(drop=True),
        "propensity_diff": distances.ravel(),
    })

    att = matched_df["treated_reactivated"].mean() - matched_df["control_reactivated"].mean()
    pair_diffs = matched_df["treated_reactivated"] - matched_df["control_reactivated"]
    se = pair_diffs.std(ddof=1) / np.sqrt(len(matched_df))

    print(f"\n=== PSM Robustness Check ===")
    print(f"  Matched pairs:     {len(matched_df):,}")
    print(f"  ATT estimate:      {att:+.1%}")
    print(f"  95% CI:           [{att - 1.96*se:+.1%}, {att + 1.96*se:+.1%}]")
    print(f"  Avg propensity Δ:  {matched_df['propensity_diff'].mean():.4f}")

    return att, se, matched_df


# ── Placebo Test ──────────────────────────────────────────────────────────────

def placebo_test(df: pd.DataFrame, n_simulations: int = 500, seed: int = 42):
    cold = df[df["went_cold"] == 1].copy()
    if cold.empty:
        raise ValueError("No cold pairs available for placebo testing.")
    placebo_estimates = []
    rng = np.random.default_rng(seed)

    for _ in range(n_simulations):
        shuffled_treatment = rng.permutation(cold["treatment"].values)
        t_rate = cold.loc[shuffled_treatment == 1, "reactivated"].mean() if (shuffled_treatment == 1).sum() > 0 else 0
        c_rate = cold.loc[shuffled_treatment == 0, "reactivated"].mean() if (shuffled_treatment == 0).sum() > 0 else 0
        placebo_estimates.append(t_rate - c_rate)

    print(f"\n=== Placebo Test ({n_simulations} simulations) ===")
    print(f"  Mean placebo estimate: {np.mean(placebo_estimates):+.4f}")
    print(f"  Std dev:               {np.std(placebo_estimates):.4f}")
    return placebo_estimates


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_did_results(r: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    periods = ["Pre-Period\n(Baseline)", "Post-Period\n(After Nudge)"]
    treatment_rates = [r["pre_rate_treatment"], r["post_rate_treatment"]]
    control_rates   = [r["pre_rate_control"],   r["post_rate_control"]]

    ax.plot(periods, treatment_rates, "o-", color="#FFFC00", linewidth=2.5, markersize=9, label="Treatment (Memory Nudge)")
    ax.plot(periods, control_rates,   "o--", color="#4ECDC4", linewidth=2.5, markersize=9, label="Control (No Nudge)")

    # Annotate DiD
    ax.annotate(
        f"DiD = {r['did_estimate']:+.1%}\np = {r['p_value']:.4f}",
        xy=(1, r["post_rate_treatment"]),
        xytext=(0.65, (r["post_rate_treatment"] + r["post_rate_control"]) / 2 + 0.04),
        color="white", fontsize=11,
        arrowprops=dict(arrowstyle="->", color="#FFFC00", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1A1A", edgecolor="#FFFC00")
    )

    ax.set_ylabel("Re-Activation Rate", color="white", fontsize=12)
    ax.set_title("A/B Test: Memory Resurfacing — DiD Results", color="white", fontsize=14, pad=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=11)
    ax.set_ylim(0.20, 0.65)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved DiD plot → {save_path}")
    return fig


def plot_placebo(placebo_estimates: list, true_estimate: float, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    ax.hist(placebo_estimates, bins=40, color="#4ECDC4", alpha=0.7, edgecolor="#0D0D0D")
    ax.axvline(true_estimate, color="#FFFC00", linewidth=2.5, linestyle="--", label=f"True estimate: {true_estimate:+.1%}")
    ax.axvline(0, color="#888", linewidth=1, linestyle="-")

    ax.set_xlabel("Placebo DiD Estimate", color="white", fontsize=11)
    ax.set_ylabel("Count", color="white", fontsize=11)
    ax.set_title("Placebo Test — Randomized Treatment Assignment", color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved placebo plot → {save_path}")
    return fig


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    from src.features import load_data, engineer_features, FEATURE_COLS

    df = load_data("data/sample_pairs.csv")
    df = engineer_features(df)

    os.makedirs("outputs", exist_ok=True)

    # DiD
    r = compute_did(df)
    print_did_results(r)
    plot_did_results(r, save_path="outputs/did_results.png")

    # PSM
    covariate_cols = ["streak_length", "avg_response_time_hrs", "pct_snaps_opened",
                      "shared_stories_viewed", "snap_map_checks", "days_since_friend_added"]
    propensity_score_matching(df, covariate_cols)

    # Placebo
    placebo_ests = placebo_test(df, n_simulations=500)
    plot_placebo(placebo_ests, true_estimate=r["did_estimate"], save_path="outputs/placebo_test.png")