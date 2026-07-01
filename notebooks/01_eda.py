# # 01 · Exploratory Data Analysis
# **Snapchat Friend Engagement Decay Project**
#
# Goals:
# - Understand the shape and distributions of our simulated friend-pair dataset
# - Identify key behavioral signals driving cold-pair risk
# - Segment pairs into engagement cohorts (Active / At-Risk / Cold)
# - Surface early insights to guide modeling decisions

# ## Setup

import sys, os
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from src.features import load_data, engineer_features, FEATURE_COLS

DARK   = "#0D0D0D"
YELLOW = "#FFFC00"
TEAL   = "#4ECDC4"
RED    = "#FF6B6B"

LAYOUT = dict(
    paper_bgcolor=DARK, plot_bgcolor=DARK,
    font=dict(color="white", size=12),
    margin=dict(l=60, r=20, t=50, b=60),
)

print("Libraries loaded ✓")

# ## 1. Load & Inspect the Dataset

df = load_data("../data/sample_pairs.csv")
df = engineer_features(df)

print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
df.head(3)

# ---

print("=== Target Distribution ===")
print(f"Cold pairs  (went_cold=1): {df['went_cold'].sum():,}  ({df['went_cold'].mean():.1%})")
print(f"Active pairs(went_cold=0): {(df['went_cold']==0).sum():,} ({(df['went_cold']==0).mean():.1%})")

cold = df[df["went_cold"] == 1]
print(f"\n=== Among Cold Pairs ===")
print(f"Re-activated: {cold['reactivated'].sum():,} ({cold['reactivated'].mean():.1%})")
print(f"Treated:      {cold['treatment'].sum():,}  ({cold['treatment'].mean():.1%})")

# ## 2. Feature Distributions

numeric_cols = [
    "days_since_last_snap", "streak_length", "avg_response_time_hrs",
    "pct_snaps_opened", "shared_stories_viewed", "snap_map_checks",
    "days_since_friend_added", "coldness_risk"
]

fig = make_subplots(rows=2, cols=4, subplot_titles=numeric_cols)

for i, col in enumerate(numeric_cols):
    r, c = divmod(i, 4)
    color = YELLOW if df[col].skew() > 1 else TEAL
    fig.add_trace(
        go.Histogram(x=df[col], nbinsx=40, marker_color=color, opacity=0.8,
                     name=col, showlegend=False),
        row=r+1, col=c+1
    )

fig.update_layout(**LAYOUT, height=520,
                  title_text="Feature Distributions (n=50,000)",
                  title_font_color="white")
for ax in fig.layout:
    if ax.startswith("xaxis") or ax.startswith("yaxis"):
        fig.layout[ax].update(gridcolor="#222", color="white")
fig.show()

# ## 3. Engagement Cohort Segmentation

df["cohort"] = pd.cut(
    df["days_since_last_snap"],
    bins=[-1, 3, 10, 60],
    labels=["Active (0–3d)", "At-Risk (4–10d)", "Cold (10d+)"]
)

cohort_summary = df.groupby("cohort", observed=True).agg(
    n_pairs=("went_cold", "count"),
    cold_rate=("went_cold", "mean"),
    avg_streak=("streak_length", "mean"),
    avg_response_hrs=("avg_response_time_hrs", "mean"),
    avg_open_rate=("pct_snaps_opened", "mean"),
).reset_index()

cohort_summary["cold_rate"]    = cohort_summary["cold_rate"].map("{:.1%}".format)
cohort_summary["avg_open_rate"] = cohort_summary["avg_open_rate"].map("{:.1%}".format)
print(cohort_summary.round(2).to_string(index=False))

# ---

cohort_counts = df.groupby("cohort", observed=True)["went_cold"].agg(["sum", "count"]).reset_index()
cohort_counts["cold_rate"] = cohort_counts["sum"] / cohort_counts["count"]

fig = go.Figure(go.Bar(
    x=cohort_counts["cohort"].astype(str),
    y=cohort_counts["cold_rate"],
    marker_color=[TEAL, YELLOW, RED],
    text=[f"{v:.1%}" for v in cohort_counts["cold_rate"]],
    textposition="outside", textfont=dict(color="white"),
))
fig.update_layout(**LAYOUT, height=380,
    title="Cold Rate by Engagement Cohort",
    yaxis=dict(tickformat=".0%", title="Cold Rate", gridcolor="#222", color="white"),
    xaxis=dict(color="white"),
)
fig.show()

# ## 4. Correlation Heatmap

corr_cols = FEATURE_COLS + ["went_cold", "reactivated"]
corr = df[corr_cols].corr()

fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale=[[0, TEAL], [0.5, "#111"], [1, YELLOW]],
    zmid=0,
    text=corr.round(2).values,
    texttemplate="%{text}",
    textfont={"size": 9},
))
fig.update_layout(**LAYOUT, height=560,
    title="Feature Correlation Matrix",
    xaxis=dict(tickangle=-45, color="white"),
    yaxis=dict(color="white"),
)
fig.show()

# ## 5. Cold vs. Active Pair Comparison

compare_cols = [
    "streak_length", "avg_response_time_hrs", "pct_snaps_opened",
    "shared_stories_viewed", "snap_map_checks", "engagement_composite"
]

compare = df.groupby("went_cold")[compare_cols].mean().T
compare.columns = ["Active Pairs", "Cold Pairs"]
compare["Δ (Cold − Active)"] = (compare["Cold Pairs"] - compare["Active Pairs"]).round(3)
compare["% Change"] = (
    (compare["Cold Pairs"] - compare["Active Pairs"]) / compare["Active Pairs"] * 100
).round(1).astype(str) + "%"
print(compare.round(3).to_string())

# ---

key_features = ["streak_length", "avg_response_time_hrs", "pct_snaps_opened", "engagement_composite"]
fig = make_subplots(rows=1, cols=4, subplot_titles=key_features)

for i, feat in enumerate(key_features):
    for label, color, val in [("Active", TEAL, 0), ("Cold", RED, 1)]:
        sample = df[df["went_cold"] == val][feat]
        fig.add_trace(
            go.Box(
                y=sample.sample(min(5000, len(sample)), random_state=42),
                name=label, marker_color=color, showlegend=(i == 0),
                boxmean=True,
            ),
            row=1, col=i+1
        )

fig.update_layout(**LAYOUT, height=420,
                  title_text="Feature Distributions: Active vs Cold Pairs",
                  title_font_color="white")
for ax in fig.layout:
    if ax.startswith("yaxis"):
        fig.layout[ax].update(gridcolor="#222", color="white")
    if ax.startswith("xaxis"):
        fig.layout[ax].update(color="white")
fig.show()

# ## 6. Coldness Risk Index Distribution

fig = go.Figure()
for label, color, val in [("Active", TEAL, 0), ("Cold", RED, 1)]:
    fig.add_trace(go.Histogram(
        x=df[df["went_cold"] == val]["coldness_risk"],
        name=label, opacity=0.75,
        marker_color=color, nbinsx=40
    ))

fig.update_layout(**LAYOUT, height=380, barmode="overlay",
    title="Coldness Risk Index: Active vs Cold Pairs",
    xaxis=dict(title="Coldness Risk Score", gridcolor="#222", color="white"),
    yaxis=dict(title="Count", gridcolor="#222", color="white"),
)
fig.show()

print("\nColdness Risk — Active pairs (mean):", df[df["went_cold"]==0]["coldness_risk"].mean().round(3))
print("Coldness Risk — Cold pairs   (mean):", df[df["went_cold"]==1]["coldness_risk"].mean().round(3))

# ## 7. EDA Key Findings
#
# | Finding | Implication |
# |---|---|
# | Cold rate 6% overall but 3× higher in At-Risk cohort | Cohort-aware nudging is more efficient than blanket outreach |
# | `avg_response_time_hrs` strongest negative correlate | Response-time early-warning could fire before streaks break |
# | `pct_snaps_opened` drops sharply in cold pairs | Open rate is a leading indicator — worth surfacing to PM |
# | `engagement_composite` cleanly separates active vs cold | Good candidate for real-time scoring in production |
# | `streak_length` protective but insufficient alone | Historical streaks don't prevent decay if cadence drops |
#
# **Next:** Survival analysis → `02_survival_analysis.py`