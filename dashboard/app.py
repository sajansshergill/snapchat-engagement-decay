"""Streamlit dashboard for the Snapchat engagement decay project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.causal import compute_did, propensity_score_matching
from src.classifier import evaluate_model, prepare_reactivation_data, train_xgb
from src.features import DASHBOARD_COLS, FEATURE_COLS, engineer_features, load_data
from src.survival_model import assign_cohort

DARK = "#0D0D0D"
YELLOW = "#FFFC00"
TEAL = "#4ECDC4"
RED = "#FF6B6B"

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


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = load_data(REPO_ROOT / "data" / "sample_pairs.csv")
    df = engineer_features(df)
    return assign_cohort(df)


@st.cache_resource(show_spinner=True)
def train_reactivation_model(df: pd.DataFrame):
    X, y, cold_df = prepare_reactivation_data(df, FEATURE_COLS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    model = train_xgb(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test)
    cold_df = cold_df.copy()
    cold_df["reactivation_score"] = model.predict_proba(cold_df[FEATURE_COLS])[:, 1]
    return model, metrics, cold_df


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
    df = get_data()
except Exception as exc:
    st.title("Snapchat Friend Engagement Decay")
    st.error(str(exc))
    st.code("python data/simulate_interactions.py", language="bash")
    st.stop()

st.title("Snapchat Friend Engagement Decay & Re-Activation")
st.caption("Synthetic portfolio project for decay prediction, re-activation scoring, and causal lift estimation.")

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
    show_rows = st.slider("Rows in score table", 10, 200, 50, step=10)

filtered = df[
    df["cohort"].astype(str).isin(selected_cohorts)
    & df["streak_length"].between(min_streak, max_streak)
].copy()

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Friend Pairs", f"{len(filtered):,}")
kpi2.metric("Cold Rate", f"{filtered['went_cold'].mean():.1%}")
cold_filtered = filtered[filtered["went_cold"] == 1]
kpi3.metric("Cold Pairs", f"{len(cold_filtered):,}")
kpi4.metric(
    "Re-Activation Rate",
    f"{cold_filtered['reactivated'].mean():.1%}" if len(cold_filtered) else "0.0%",
)

tab_overview, tab_model, tab_causal, tab_data = st.tabs(
    ["Decay Overview", "Re-Activation Model", "Causal Lift", "Pair Explorer"]
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
        text=cohort_rates["cold_rate"].map(lambda v: f"{v:.1%}"),
        title="Cold Rate by Engagement Cohort",
    )
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(plot_theme(fig), use_container_width=True)

    fig = px.scatter(
        filtered.sample(min(len(filtered), 5000), random_state=42),
        x="avg_response_time_hrs",
        y="pct_snaps_opened",
        color="went_cold",
        size="streak_length",
        color_continuous_scale=[TEAL, RED],
        title="Response Lag vs Open Rate",
        labels={"went_cold": "Went Cold"},
    )
    right.plotly_chart(plot_theme(fig), use_container_width=True)

    fig = px.histogram(
        filtered,
        x="coldness_risk",
        color="went_cold",
        nbins=40,
        barmode="overlay",
        color_discrete_sequence=[TEAL, RED],
        title="Coldness Risk Distribution",
    )
    st.plotly_chart(plot_theme(fig, height=360), use_container_width=True)

with tab_model:
    st.subheader("Cold Pair Re-Activation Scoring")
    try:
        model, metrics, scored_cold = train_reactivation_model(df)
        m1, m2, m3 = st.columns(3)
        m1.metric("AUC-ROC", f"{metrics['auc']:.3f}")
        m2.metric("Average Precision", f"{metrics['ap']:.3f}")
        m3.metric("Precision @ Top Decile", f"{metrics['precision_at_k']:.1%}")

        scored_filtered = scored_cold[
            scored_cold["user_pair_id"].isin(filtered["user_pair_id"])
        ].sort_values("reactivation_score", ascending=False)

        fig = px.histogram(
            scored_filtered,
            x="reactivation_score",
            nbins=40,
            color="reactivated",
            color_discrete_sequence=[RED, YELLOW],
            title="Predicted Re-Activation Score Distribution",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(plot_theme(fig, height=360), use_container_width=True)

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
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Model training failed: {exc}")

with tab_causal:
    st.subheader("Memory Resurfacing Experiment")
    try:
        did = compute_did(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DiD Lift", f"{did['did_estimate']:+.1%}")
        c2.metric("95% CI", f"{did['ci_lower']:+.1%} to {did['ci_upper']:+.1%}")
        c3.metric("p-value", f"{did['p_value']:.4f}")
        c4.metric("Experiment N", f"{did['n_treatment'] + did['n_control']:,}")

        periods = ["Pre", "Post"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods,
            y=[did["pre_rate_treatment"], did["post_rate_treatment"]],
            mode="lines+markers",
            name="Treatment",
            line=dict(color=YELLOW, width=3),
        ))
        fig.add_trace(go.Scatter(
            x=periods,
            y=[did["pre_rate_control"], did["post_rate_control"]],
            mode="lines+markers",
            name="Control",
            line=dict(color=TEAL, width=3, dash="dash"),
        ))
        fig.update_yaxes(title="Re-Activation Rate", tickformat=".0%")
        fig.update_layout(title="Difference-in-Differences: Pre/Post Rates")
        st.plotly_chart(plot_theme(fig), use_container_width=True)

        with st.expander("Run PSM robustness check"):
            covariates = [
                "streak_length",
                "avg_response_time_hrs",
                "pct_snaps_opened",
                "shared_stories_viewed",
                "snap_map_checks",
                "days_since_friend_added",
            ]
            att, se, matched = propensity_score_matching(df, covariates)
            st.write(
                f"Matched ATT: **{att:+.1%}** "
                f"(95% CI [{att - 1.96 * se:+.1%}, {att + 1.96 * se:+.1%}], "
                f"{len(matched):,} matched treated pairs)"
            )
    except Exception as exc:
        st.error(f"Causal analysis failed: {exc}")

with tab_data:
    st.subheader("Pair-Level Data")
    st.dataframe(
        filtered[DASHBOARD_COLS + ["cohort"]].head(show_rows),
        use_container_width=True,
    )
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_friend_pairs.csv",
        mime="text/csv",
    )
