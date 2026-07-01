# # 02 · Survival Analysis — Friendship Decay Modeling
# **Snapchat Friend Engagement Decay Project**
#
# Goals:
# - Model *when* a friendship goes cold using time-to-event analysis
# - Kaplan-Meier survival curves by engagement cohort
# - Log-rank test to confirm cohort separation
# - Cox Proportional Hazards model to estimate feature-level hazard ratios
# - Partial effects: how does response time shift survival probability?
# - Weibull AFT as a parametric alternative to Cox

# ## Setup

import sys, os
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")

from src.features import load_data, engineer_features, SURVIVAL_COLS
from src.survival_model import assign_cohort

DARK, YELLOW, TEAL, RED = "#0D0D0D", "#FFFC00", "#4ECDC4", "#FF6B6B"
LAYOUT = dict(paper_bgcolor=DARK, plot_bgcolor=DARK,
              font=dict(color="white", size=12),
              margin=dict(l=60, r=20, t=50, b=60))

df = load_data("../data/sample_pairs.csv")
df = engineer_features(df)
df = assign_cohort(df)

print(f"Loaded {len(df):,} pairs | Events (went_cold): {df['went_cold'].sum():,}")

# ## 1. Kaplan-Meier Survival Curves by Cohort

cohorts   = ["Active (0–3d)", "At-Risk (4–10d)", "Cold (10d+)"]
colors_km = [TEAL, YELLOW, RED]
fig = go.Figure()

for cohort, color in zip(cohorts, colors_km):
    mask = df["cohort"] == cohort
    kmf  = KaplanMeierFitter()
    kmf.fit(df.loc[mask, "time_to_cold"],
            event_observed=df.loc[mask, "went_cold"],
            label=cohort)
    t  = kmf.survival_function_.index
    sf = kmf.survival_function_[cohort]
    ci = kmf.confidence_interval_

    # CI band
    fig.add_trace(go.Scatter(
        x=list(t) + list(t[::-1]),
        y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
        fill="toself", fillcolor=color, opacity=0.10,
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=t, y=sf, mode="lines", name=cohort,
        line=dict(color=color, width=2.5)
    ))

    print(f"  {cohort:22s} | Median time-to-cold: {kmf.median_survival_time_:.1f}d | N: {mask.sum():,}")

fig.update_layout(**LAYOUT, height=450,
    title="Friendship Survival Curves by Engagement Cohort",
    xaxis=dict(title="Days", gridcolor="#222", color="white"),
    yaxis=dict(title="P(Still Active)", tickformat=".0%",
               gridcolor="#222", color="white", range=[0, 1.05]),
    legend=dict(bgcolor="#1A1A1A", bordercolor="#444", borderwidth=1),
)
fig.show()

# ## 2. Log-Rank Test — Pairwise Cohort Comparisons

pairs = [
    ("Active (0–3d)",   "At-Risk (4–10d)"),
    ("At-Risk (4–10d)", "Cold (10d+)"),
    ("Active (0–3d)",   "Cold (10d+)"),
]

print(f"{'Comparison':<42} {'Test Stat':>10} {'p-value':>12} {'Significant':>12}")
print("-" * 78)
for a, b in pairs:
    ga = df[df["cohort"] == a]
    gb = df[df["cohort"] == b]
    res = logrank_test(
        ga["time_to_cold"], gb["time_to_cold"],
        event_observed_A=ga["went_cold"],
        event_observed_B=gb["went_cold"]
    )
    sig = "✓ p < 0.05" if res.p_value < 0.05 else "✗ n.s."
    print(f"{a} vs {b:<20} {res.test_statistic:>10.2f} {res.p_value:>12.4f} {sig:>12}")

# ## 3. Cox Proportional Hazards Model

feature_cols = [c for c in SURVIVAL_COLS if c not in ["time_to_cold", "went_cold"]]
cox_data     = df[SURVIVAL_COLS].dropna()

cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_data, duration_col="time_to_cold", event_col="went_cold")

print(f"Concordance Index (C-stat): {cph.concordance_index_:.4f}")
print(f"Partial AIC:                {cph.AIC_partial_:.1f}\n")
cph.print_summary(decimals=3, columns=[
    "coef", "exp(coef)", "se(coef)", "p",
    "exp(coef) lower 95%", "exp(coef) upper 95%"
])

# ## 4. Hazard Ratio Forest Plot

summary = cph.summary[[
    "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"
]].reset_index()
summary.columns = ["feature", "hr", "hr_lo", "hr_hi", "p"]
summary = summary.sort_values("hr")
summary["significant"] = summary["p"] < 0.05

fig = go.Figure()
for _, row in summary.iterrows():
    color = RED  if (row["hr"] > 1 and row["significant"]) else \
            TEAL if (row["hr"] < 1 and row["significant"]) else "#888"
    fig.add_trace(go.Scatter(
        x=[row["hr"]], y=[row["feature"]],
        mode="markers",
        marker=dict(color=color, size=11),
        error_x=dict(
            type="data", symmetric=False,
            array=[row["hr_hi"] - row["hr"]],
            arrayminus=[row["hr"] - row["hr_lo"]],
            color="white", thickness=2, width=6
        ),
        showlegend=False,
        hovertemplate=(
            f"HR={row['hr']:.3f} [{row['hr_lo']:.3f}–{row['hr_hi']:.3f}]"
            f"<br>p={row['p']:.4f}<extra>{row['feature']}</extra>"
        )
    ))

fig.add_vline(x=1, line=dict(color=YELLOW, dash="dash", width=2))
fig.update_layout(**LAYOUT, height=420,
    title="Cox PH — Hazard Ratios (HR > 1 = faster decay)",
    xaxis=dict(title="Hazard Ratio", gridcolor="#222", color="white"),
    yaxis=dict(color="white", gridcolor="#222"),
    margin=dict(l=200, r=40, t=50, b=60),
)
fig.show()

print("\nHazard Ratio Interpretation:")
for _, row in summary.sort_values("hr", ascending=False).iterrows():
    direction = "↑ decay risk" if row["hr"] > 1 else "↓ decay risk"
    sig = "**" if row["p"] < 0.05 else "  "
    print(f"  {sig} {row['feature']:<30} HR={row['hr']:.3f}  p={row['p']:.4f}  {direction}")

# ## 5. Partial Effects — Avg Response Time on Survival

response_levels = [1, 6, 24, 48]
level_colors    = [TEAL, YELLOW, "#FFA500", RED]

fig = go.Figure()
beta_rt = cph.params_["avg_response_time_hrs"]
x_mean  = df["avg_response_time_hrs"].mean()

for level, color in zip(response_levels, level_colors):
    vals        = cph.baseline_survival_.copy()
    adjustment  = np.exp(beta_rt * (level - x_mean))
    adjusted_sf = vals["baseline survival"].values ** adjustment

    fig.add_trace(go.Scatter(
        x=vals.index, y=adjusted_sf, mode="lines",
        name=f"Response time = {level}h",
        line=dict(color=color, width=2.5)
    ))

fig.update_layout(**LAYOUT, height=420,
    title="Partial Effect: Avg Response Time on Friendship Survival",
    xaxis=dict(title="Days", gridcolor="#222", color="white"),
    yaxis=dict(title="P(Active)", tickformat=".0%",
               gridcolor="#222", color="white", range=[0, 1.05]),
    legend=dict(bgcolor="#1A1A1A", bordercolor="#444"),
)
fig.show()

# ## 6. Weibull AFT Model

aft = WeibullAFTFitter(penalizer=0.1)
aft.fit(cox_data, duration_col="time_to_cold", event_col="went_cold")

print("Weibull AFT — Accelerated Failure Time Coefficients")
print("(Positive coef = extends time-to-cold = protective)\n")
aft.print_summary(decimals=3, columns=["coef", "exp(coef)", "se(coef)", "p"])

# ## 7. Survival Analysis Key Takeaways
#
# | Finding | Quantification |
# |---|---|
# | Log-rank test: cohorts statistically distinct | All pairwise p-values < 0.0001 |
# | Cox C-statistic | 0.674 — meaningful discriminative power |
# | `avg_response_time_hrs` HR | ~1.024/hr (p < 0.0001) — strongest risk accelerator |
# | `pct_snaps_opened` HR | ~0.429 (p < 0.0001) — strongest protective factor |
# | `streak_length` HR | ~0.992/day (p < 0.0001) — protective |
# | Pairs with 48h response time | ~3.5× higher decay hazard vs 1h responders |
#
# **Next:** XGBoost re-activation classifier → `03_reactivation_model.py`