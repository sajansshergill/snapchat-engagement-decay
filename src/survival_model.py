"""
survival_model.py
Kaplan-Meier + Cox Proportional Hazards model for friendship decay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")


def assign_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort"] = pd.cut(
        df["days_since_last_snap"],
        bins=[-1, 3, 10, 60],
        labels=["Active (0–3d)", "At-Risk (4–10d)", "Cold (10d+)"]
    )
    return df


def plot_km_curves(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    cohorts = df["cohort"].dropna().unique()
    colors = ["#FFFC00", "#FF6B6B", "#4ECDC4"]

    for cohort, color in zip(sorted(cohorts), colors):
        mask = df["cohort"] == cohort
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, "time_to_cold"],
            event_observed=df.loc[mask, "went_cold"],
            label=str(cohort)
        )
        kmf.plot_survival_function(
            ax=ax, ci_show=True, color=color, linewidth=2.5, alpha=0.9
        )

    ax.set_xlabel("Days", color="white", fontsize=12)
    ax.set_ylabel("Probability Friendship Stays Active", color="white", fontsize=12)
    ax.set_title("Friendship Survival Curves by Engagement Cohort", color="white", fontsize=14, pad=16)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved KM plot → {save_path}")
    return fig


def run_logrank_test(df: pd.DataFrame):
    active = df[df["cohort"] == "Active (0–3d)"]
    at_risk = df[df["cohort"] == "At-Risk (4–10d)"]

    result = logrank_test(
        active["time_to_cold"], at_risk["time_to_cold"],
        event_observed_A=active["went_cold"],
        event_observed_B=at_risk["went_cold"]
    )
    return result


def fit_cox_model(df: pd.DataFrame, feature_cols: list) -> CoxPHFitter:
    cox_df = df[feature_cols + ["time_to_cold", "went_cold"]].dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="time_to_cold", event_col="went_cold")
    return cph


def plot_cox_hazard_ratios(cph: CoxPHFitter, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary = summary.sort_values("exp(coef)", ascending=True)

    colors = ["#FF6B6B" if v > 1 else "#4ECDC4" for v in summary["exp(coef)"]]
    y_pos = range(len(summary))

    ax.barh(y_pos, summary["exp(coef)"] - 1, left=1, color=colors, alpha=0.85, height=0.6)
    ax.errorbar(
        summary["exp(coef)"], y_pos,
        xerr=[
            summary["exp(coef)"] - summary["exp(coef) lower 95%"],
            summary["exp(coef) upper 95%"] - summary["exp(coef)"]
        ],
        fmt="none", color="white", capsize=4, linewidth=1.5
    )
    ax.axvline(x=1, color="#FFFC00", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(summary.index, color="white", fontsize=10)
    ax.set_xlabel("Hazard Ratio (HR > 1 = higher decay risk)", color="white", fontsize=11)
    ax.set_title("Cox PH Model — Feature Hazard Ratios", color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved Cox HR plot → {save_path}")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.features import load_data, engineer_features, SURVIVAL_COLS

    df = load_data("data/sample_pairs.csv")
    df = engineer_features(df)
    df = assign_cohort(df)

    print("=== Kaplan-Meier Log-Rank Test ===")
    result = run_logrank_test(df)
    print(f"p-value: {result.p_value:.4f} | Test statistic: {result.test_statistic:.2f}")

    feature_cols = [c for c in SURVIVAL_COLS if c not in ["time_to_cold", "went_cold"]]
    cph = fit_cox_model(df, feature_cols)

    print("\n=== Cox PH Model Summary ===")
    cph.print_summary(decimals=3)

    import os
    os.makedirs("outputs", exist_ok=True)
    plot_km_curves(df, save_path="outputs/km_curves.png")
    plot_cox_hazard_ratios(cph, save_path="outputs/cox_hr.png")