"""Dependency-light Streamlit dashboard for Streamlit Community Cloud."""

from __future__ import annotations

import csv
from io import StringIO
from math import erfc, exp, sqrt
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "sample_pairs.csv"

DARK = "#0D0D0D"
YELLOW = "#FFFC00"
TEAL = "#4ECDC4"
RED = "#FF6B6B"

NUMERIC_COLUMNS = [
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

DASHBOARD_COLUMNS = [
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
    "cohort",
]


st.set_page_config(page_title="Snapchat Engagement Decay", page_icon="📸", layout="wide")
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


def sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def mean(rows: list[dict], column: str) -> float:
    return sum(float(row[column]) for row in rows) / len(rows) if rows else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cohort_for(days_since_last_snap: float) -> str:
    if days_since_last_snap <= 3:
        return "Active (0-3d)"
    if days_since_last_snap <= 10:
        return "At-Risk (4-10d)"
    return "Cold (10d+)"


def add_features(row: dict) -> dict:
    row["coldness_risk"] = max(
        0.0,
        min(
            10.0,
            row["days_since_last_snap"] * 0.30
            + row["avg_response_time_hrs"] * 0.10
            - row["streak_length"] * 0.05
            - row["pct_snaps_opened"] * 2.00,
        ),
    )
    score_logit = (
        -0.90
        + 0.035 * row["streak_length"]
        + 0.090 * row["shared_stories_viewed"]
        + 0.110 * row["snap_map_checks"]
        - 0.030 * row["avg_response_time_hrs"]
        - 0.002 * row["friend_suggestion_rank"]
        + 0.045 * row["notification_received_7d"]
    )
    row["reactivation_score"] = max(0.0, min(1.0, sigmoid(score_logit)))
    row["cohort"] = cohort_for(row["days_since_last_snap"])
    return row


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> list[dict]:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError("data/sample_pairs.csv is missing from the repository.")

    with DATA_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(set(["user_pair_id", *NUMERIC_COLUMNS]) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        rows = []
        for row in reader:
            parsed = {"user_pair_id": row["user_pair_id"]}
            for column in NUMERIC_COLUMNS:
                parsed[column] = float(row[column] or 0)
            rows.append(add_features(parsed))
    return rows


def compute_lift(rows: list[dict]) -> dict[str, float]:
    cold = [row for row in rows if row["went_cold"] == 1]
    treatment = [row for row in cold if row["treatment"] == 1]
    control = [row for row in cold if row["treatment"] == 0]
    treatment_rate = mean(treatment, "reactivated")
    control_rate = mean(control, "reactivated")
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


def themed(fig: go.Figure, height: int = 420) -> go.Figure:
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


def to_csv(rows: list[dict]) -> bytes:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=DASHBOARD_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in DASHBOARD_COLUMNS})
    return output.getvalue().encode("utf-8")


try:
    data = load_dashboard_data()
except Exception as exc:
    st.title("Snapchat Friend Engagement Decay")
    st.error(str(exc))
    st.stop()

st.title("Snapchat Friend Engagement Decay & Re-Activation")
st.caption("Synthetic portfolio project for decay risk, re-activation scoring, and experiment lift.")

with st.sidebar:
    st.header("Filters")
    cohorts = ["Active (0-3d)", "At-Risk (4-10d)", "Cold (10d+)"]
    selected_cohorts = st.multiselect("Engagement cohort", cohorts, default=cohorts)
    streak_values = [row["streak_length"] for row in data]
    min_streak, max_streak = st.slider(
        "Streak length",
        min_value=int(min(streak_values)),
        max_value=int(max(streak_values)),
        value=(int(min(streak_values)), int(quantile(streak_values, 0.95))),
    )
    show_rows = st.slider("Rows in table", 10, 200, 50, step=10)

filtered = [
    row
    for row in data
    if row["cohort"] in selected_cohorts and min_streak <= row["streak_length"] <= max_streak
]
if not filtered:
    st.warning("No rows match the selected filters.")
    st.stop()

cold_filtered = [row for row in filtered if row["went_cold"] == 1]
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Friend Pairs", f"{len(filtered):,}")
kpi2.metric("Cold Rate", f"{mean(filtered, 'went_cold'):.1%}")
kpi3.metric("Cold Pairs", f"{len(cold_filtered):,}")
kpi4.metric("Re-Activation Rate", f"{mean(cold_filtered, 'reactivated'):.1%}")

tab_overview, tab_model, tab_causal, tab_data = st.tabs(
    ["Decay Overview", "Re-Activation Scores", "Experiment Lift", "Pair Explorer"]
)

with tab_overview:
    left, right = st.columns(2)
    cohort_counts = []
    cohort_rates = []
    for cohort in cohorts:
        cohort_rows = [row for row in filtered if row["cohort"] == cohort]
        cohort_counts.append(len(cohort_rows))
        cohort_rates.append(mean(cohort_rows, "went_cold"))

    fig = go.Figure(go.Bar(
        x=cohorts,
        y=cohort_rates,
        marker_color=[TEAL, YELLOW, RED],
        text=[f"{value:.1%}" for value in cohort_rates],
        textposition="outside",
    ))
    fig.update_layout(title="Cold Rate by Engagement Cohort")
    fig.update_yaxes(title="Cold Rate", tickformat=".0%")
    left.plotly_chart(themed(fig), width="stretch")

    sample = filtered[:5000]
    fig = go.Figure(go.Scatter(
        x=[row["avg_response_time_hrs"] for row in sample],
        y=[row["pct_snaps_opened"] for row in sample],
        mode="markers",
        marker=dict(
            size=[max(4, min(18, row["streak_length"] / 12)) for row in sample],
            color=[row["went_cold"] for row in sample],
            colorscale=[[0, TEAL], [1, RED]],
            opacity=0.55,
        ),
    ))
    fig.update_layout(title="Response Lag vs Open Rate")
    fig.update_xaxes(title="Avg Response Time (hrs)")
    fig.update_yaxes(title="Pct Snaps Opened", tickformat=".0%")
    right.plotly_chart(themed(fig), width="stretch")

    fig = go.Figure()
    for label, color, value in [("Active", TEAL, 0), ("Cold", RED, 1)]:
        rows = [row for row in filtered if row["went_cold"] == value]
        fig.add_trace(go.Histogram(
            x=[row["coldness_risk"] for row in rows],
            name=label,
            marker_color=color,
            opacity=0.75,
            nbinsx=40,
        ))
    fig.update_layout(title="Coldness Risk Distribution", barmode="overlay")
    fig.update_xaxes(title="Coldness Risk")
    st.plotly_chart(themed(fig, height=360), width="stretch")

with tab_model:
    st.subheader("Cold Pair Re-Activation Scores")
    scored = sorted(cold_filtered, key=lambda row: row["reactivation_score"], reverse=True)
    cutoff = quantile([row["reactivation_score"] for row in scored], 0.90)
    top_decile = [row for row in scored if row["reactivation_score"] >= cutoff]

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Score", f"{mean(scored, 'reactivation_score'):.1%}")
    m2.metric("Top-Decile Actual Rate", f"{mean(top_decile, 'reactivated'):.1%}")
    m3.metric("Scored Cold Pairs", f"{len(scored):,}")

    fig = go.Figure()
    for label, color, value in [("Stayed Cold", RED, 0), ("Re-Activated", YELLOW, 1)]:
        rows = [row for row in scored if row["reactivated"] == value]
        fig.add_trace(go.Histogram(
            x=[row["reactivation_score"] for row in rows],
            name=label,
            marker_color=color,
            opacity=0.75,
            nbinsx=40,
        ))
    fig.update_layout(title="Predicted Re-Activation Score Distribution", barmode="overlay")
    fig.update_xaxes(title="Re-Activation Score", tickformat=".0%")
    st.plotly_chart(themed(fig, height=360), width="stretch")

    st.dataframe([{column: row[column] for column in DASHBOARD_COLUMNS} for row in scored[:show_rows]], width="stretch")

with tab_causal:
    st.subheader("Memory Resurfacing Experiment")
    lift = compute_lift(data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Treatment Lift", f"{lift['lift']:+.1%}")
    c2.metric("95% CI", f"{lift['ci_lower']:+.1%} to {lift['ci_upper']:+.1%}")
    c3.metric("p-value", f"{lift['p_value']:.4f}")
    c4.metric("Experiment N", f"{lift['n_treatment'] + lift['n_control']:,}")

    fig = go.Figure(go.Bar(
        x=["Control", "Memory Nudge"],
        y=[lift["control_rate"], lift["treatment_rate"]],
        marker_color=[TEAL, YELLOW],
        text=[f"{lift['control_rate']:.1%}", f"{lift['treatment_rate']:.1%}"],
        textposition="outside",
    ))
    fig.update_layout(title="Observed Re-Activation Rate by Experiment Arm")
    fig.update_yaxes(title="Re-Activation Rate", tickformat=".0%")
    st.plotly_chart(themed(fig), width="stretch")

with tab_data:
    st.subheader("Pair-Level Data")
    table_rows = [{column: row[column] for column in DASHBOARD_COLUMNS} for row in filtered[:show_rows]]
    st.dataframe(table_rows, width="stretch")
    st.download_button(
        "Download filtered CSV",
        to_csv(filtered),
        file_name="filtered_friend_pairs.csv",
        mime="text/csv",
    )
