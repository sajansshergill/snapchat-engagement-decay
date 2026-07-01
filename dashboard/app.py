"""Lightweight Streamlit dashboard for deployment."""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "sample_pairs.csv"

DARK = "#0D0D0D"
YELLOW = "#FFFC00"
TEAL = "#4ECDC4"
RED = "#FF6B6B"

BASE_COLUMNS = [
    "user_pair_id",
    "days_since_last_snap",
    "streak_length",
    "avg_response_time_hrs",
    "pct_snaps_opened",
    "shared_stories_viewed",
    "snap_map_checks",
    "friend_suggestion_rank",
    "days_since_friend_added",
    "notification_received_7d",
    "treatment",
    "time_to_cold",
    "went_cold",
    "reactivated",
]

DASHBOARD_COLS = [
    "user_pair_id",
    "days_since_last_snap",
    "streak_length",
    "avg_response_time_hrs",
    "pct_snaps_opened",
    "shared_stories_viewed",
    "snap_map_checks",
    "days_since_friend_added",
    "coldness_risk",
    "reactivation_score",
    "reactivated",
    "treatment",
]


st.set_page_config(
    page_title="Snapchat Engagement Decay",
    page_icon="📸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0D0D0D; color: white; }
    div[data-testid="stMetricValue"] { color: #FFFC00; }
    div[data-testid="stMetricLabel"] { color: #EEE; }
    </style>
    """,
    unsafe_allow_html=True,
)


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1 / (1 + np.exp(-x))


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError("data/sample_pairs.csv is missing. Regenerate and push the sample dataset.")

    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(BASE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric_cols = [col for col in BASE_COLUMNS if col != "user_pair_id"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["snap_velocity"] = df["streak_length"] / (df["days_since_friend_added"].clip(lower=0) + 1)
    df["response_speed_index"] = 1 / (df["avg_response_time_hrs"].clip(lower=0) + 1)
    df["engagement_composite"] = df["pct_snaps_opened"] * df["response_speed_index"]
    df["proximity_score"] = df["snap_map_checks"] + 0.5 * df["shared_stories_viewed"]
    df["coldness_risk"] = (
        df["days_since_last_snap"] * 0.30
        + df["avg_response_time_hrs"] * 0.10
        - df["streak_length"] * 0.05
        - df["pct_snaps_opened"] * 2.00
    ).clip(0, 10)

    score_logit = (
        -0.90
        + 0.035 * df["streak_length"]
        + 0.090 * df["shared_stories_viewed"]
        + 0.110 * df["snap_map_checks"]
        - 0.030 * df["avg_response_time_hrs"]
        - 0.002 * df["friend_suggestion_rank"]
        + 0.045 * df["notification_received_7d"]
    )
    df["reactivation_score"] = sigmoid(score_logit).clip(0, 1)
    df["cohort"] = pd.cut(
        df["days_since_last_snap"],
        bins=[-1, 3, 10, np.inf],
        labels=["Active (0-3d)", "At-Risk (4-10d)", "Cold (10d+)"],
    )
    return df


def compute_lift(df: pd.DataFrame) -> dict[str, float]:
    cold = df[df["went_cold"] == 1]
    treatment = cold[cold["treatment"] == 1]
    control = cold[cold["treatment"] == 0]
    treatment_rate = treatment["reactivated"].mean()
    control_rate = control["reactivated"].mean()
    lift = treatment_rate - control_rate
    se = sqrt(
        treatment_rate * (1 - treatment_rate) / max(len(treatment), 1)
        + control_rate * (1 - control_rate) / max(len(control), 1)
    )
    z_score = lift / se if se else 0.0
    return {
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "lift": lift,
        "ci_lower": lift - 1.96 * se,
        "ci_upper": lift + 1.96 * se,
        "p_value": erfc(abs(z_score) / sqrt(2)),
        "n_treatment": len(treatment),
        "n_control": len(control),
    }


def plot_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=DARK,
        plot_bgcolor=DARK,
        font=dict(color="white"),
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(bgcolor="#1A1A1A", bordercolor="#444", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="#222", color="white")
    fig.update_yaxes(gridcolor="#222", color="white")
    return fig


try:
    df = load_dashboard_data()
except Exception as exc:
    st.title("Snapchat Friend Engagement Decay")
    st.error(str(exc))
    st.stop()

st.title("Snapchat Friend Engagement Decay & Re-Activation")
st.caption("Synthetic portfolio project for decay risk, re-activation scoring, and experiment lift.")

with st.sidebar:
    st.header("Filters")
    cohorts = sorted(df["cohort"].dropna().astype(str).unique())
    selected_cohorts = st.multiselect("Engagement cohort", cohorts, default=cohorts)
    min_streak, max_streak = st.slider(
        "Streak length",
        min_value=int(df["streak_length"].min()),
        max_value=int(df["streak_length"].max()),
        value=(int(df["streak_length"].min()), int(df["streak_length"].quantile(0.95))),
    )
    show_rows = st.slider("Rows in table", 10, 200, 50, step=10)

filtered = df[
    df["cohort"].astype(str).isin(selected_cohorts)
    & df["streak_length"].between(min_streak, max_streak)
].copy()

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

cold_filtered = filtered[filtered["went_cold"] == 1]
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Friend Pairs", f"{len(filtered):,}")
kpi2.metric("Cold Rate", f"{filtered['went_cold'].mean():.1%}")
kpi3.metric("Cold Pairs", f"{len(cold_filtered):,}")
kpi4.metric("Re-Activation Rate", f"{cold_filtered['reactivated'].mean():.1%}" if len(cold_filtered) else "0.0%")

tab_overview, tab_model, tab_causal, tab_data = st.tabs(
    ["Decay Overview", "Re-Activation Scores", "Experiment Lift", "Pair Explorer"]
)

with tab_overview:
    left, right = st.columns(2)
    cohort_rates = (
        filtered.groupby("cohort", observed=True)
        .agg(n_pairs=("went_cold", "count"), cold_rate=("went_cold", "mean"))
        .reset_index()
    )
    fig = px.bar(
        cohort_rates,
        x="cohort",
        y="cold_rate",
        color="cohort",
        color_discrete_sequence=[TEAL, YELLOW, RED],
        text=cohort_rates["cold_rate"].map(lambda value: f"{value:.1%}"),
        title="Cold Rate by Engagement Cohort",
    )
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(plot_theme(fig), width="stretch")

    scatter_df = filtered.sample(min(len(filtered), 5000), random_state=42)
    fig = px.scatter(
        scatter_df,
        x="avg_response_time_hrs",
        y="pct_snaps_opened",
        color="went_cold",
        size="streak_length",
        color_continuous_scale=[TEAL, RED],
        title="Response Lag vs Open Rate",
        labels={"went_cold": "Went Cold"},
    )
    right.plotly_chart(plot_theme(fig), width="stretch")

    fig = px.histogram(
        filtered,
        x="coldness_risk",
        color="went_cold",
        nbins=40,
        barmode="overlay",
        color_discrete_sequence=[TEAL, RED],
        title="Coldness Risk Distribution",
    )
    st.plotly_chart(plot_theme(fig, height=360), width="stretch")

with tab_model:
    st.subheader("Cold Pair Re-Activation Scores")
    scored_filtered = cold_filtered.sort_values("reactivation_score", ascending=False)
    top_decile_cutoff = scored_filtered["reactivation_score"].quantile(0.90) if len(scored_filtered) else 0
    top_decile = scored_filtered[scored_filtered["reactivation_score"] >= top_decile_cutoff]

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Score", f"{scored_filtered['reactivation_score'].mean():.1%}" if len(scored_filtered) else "0.0%")
    m2.metric("Top-Decile Actual Rate", f"{top_decile['reactivated'].mean():.1%}" if len(top_decile) else "0.0%")
    m3.metric("Scored Cold Pairs", f"{len(scored_filtered):,}")

    fig = px.histogram(
        scored_filtered,
        x="reactivation_score",
        nbins=40,
        color="reactivated",
        color_discrete_sequence=[RED, YELLOW],
        title="Predicted Re-Activation Score Distribution",
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(plot_theme(fig, height=360), width="stretch")

    display_cols = [
        "user_pair_id",
        "reactivation_score",
        "streak_length",
        "avg_response_time_hrs",
        "pct_snaps_opened",
        "shared_stories_viewed",
        "snap_map_checks",
        "days_since_friend_added",
        "reactivated",
    ]
    st.dataframe(
        scored_filtered[display_cols].head(show_rows).style.format(
            {"reactivation_score": "{:.1%}", "pct_snaps_opened": "{:.1%}"}
        ),
        width="stretch",
    )

with tab_causal:
    st.subheader("Memory Resurfacing Experiment")
    lift = compute_lift(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Treatment Lift", f"{lift['lift']:+.1%}")
    c2.metric("95% CI", f"{lift['ci_lower']:+.1%} to {lift['ci_upper']:+.1%}")
    c3.metric("p-value", f"{lift['p_value']:.4f}")
    c4.metric("Experiment N", f"{lift['n_treatment'] + lift['n_control']:,}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Memory Nudge"],
        y=[lift["control_rate"], lift["treatment_rate"]],
        marker_color=[TEAL, YELLOW],
        text=[f"{lift['control_rate']:.1%}", f"{lift['treatment_rate']:.1%}"],
        textposition="outside",
    ))
    fig.update_yaxes(title="Re-Activation Rate", tickformat=".0%")
    fig.update_layout(title="Observed Re-Activation Rate by Experiment Arm")
    st.plotly_chart(plot_theme(fig), width="stretch")

with tab_data:
    st.subheader("Pair-Level Data")
    st.dataframe(filtered[DASHBOARD_COLS + ["cohort"]].head(show_rows), width="stretch")
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_friend_pairs.csv",
        mime="text/csv",
    )
