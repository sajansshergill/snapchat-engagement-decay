"""Feature engineering utilities for engagement decay modeling."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


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


def _resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    repo_root = Path(__file__).resolve().parents[1]
    for alternative in (repo_root / candidate, repo_root / "data" / candidate.name):
        if alternative.exists():
            return alternative
    return candidate


def load_data(path: str | Path = "data/sample_pairs.csv") -> pd.DataFrame:
    """Load a CSV with consistent typing and a clear error for empty datasets."""
    csv_path = _resolve_data_path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"No usable dataset found at {csv_path}. Run `python data/simulate_interactions.py` first."
        )

    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(
            "SELECT * FROM read_csv_auto(?, header=True)",
            [str(csv_path)],
        ).df()
    finally:
        con.close()

    missing = sorted(set(BASE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic model features used across notebooks and the dashboard."""
    df = df.copy()
    numeric_cols = [c for c in BASE_COLUMNS if c != "user_pair_id"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interaction velocity: snaps per day of friendship
    df["snap_velocity"] = df["streak_length"] / (df["days_since_friend_added"].clip(lower=0) + 1)

    # Engagement composite: open rate × response speed index
    df["response_speed_index"] = 1 / (df["avg_response_time_hrs"].clip(lower=0) + 1)
    df["engagement_composite"] = df["pct_snaps_opened"] * df["response_speed_index"]

    # Recency-weighted streak (streaks on old friendships matter more)
    df["weighted_streak"] = df["streak_length"] * np.log1p(df["days_since_friend_added"].clip(lower=0))

    # Social proximity score
    df["proximity_score"] = df["snap_map_checks"] + df["shared_stories_viewed"] * 0.5

    # Coldness risk index (interpretable feature for dashboard)
    df["coldness_risk"] = (
        df["days_since_last_snap"] * 0.3
        + df["avg_response_time_hrs"] * 0.1
        - df["streak_length"] * 0.05
        - df["pct_snaps_opened"] * 2.0
    ).clip(0, 10).round(2)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def assign_engagement_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Segment friend pairs by snap recency."""
    df = df.copy()
    df["cohort"] = pd.cut(
        df["days_since_last_snap"],
        bins=[-1, 3, 10, np.inf],
        labels=["Active (0-3d)", "At-Risk (4-10d)", "Cold (10d+)"],
    )
    return df


FEATURE_COLS = [
    "days_since_last_snap",
    "streak_length",
    "avg_response_time_hrs",
    "pct_snaps_opened",
    "shared_stories_viewed",
    "snap_map_checks",
    "friend_suggestion_rank",
    "days_since_friend_added",
    "notification_received_7d",
    "snap_velocity",
    "engagement_composite",
    "weighted_streak",
    "proximity_score",
]

SURVIVAL_COLS = [
    "avg_response_time_hrs",
    "streak_length",
    "pct_snaps_opened",
    "shared_stories_viewed",
    "snap_map_checks",
    "days_since_friend_added",
    "notification_received_7d",
    "snap_velocity",
    "proximity_score",
    "time_to_cold",
    "went_cold",
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
    "reactivated",
    "treatment",
]