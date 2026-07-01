"""
simulate_interactions.py
Generates a synthetic Snapchat friend-pair interaction dataset.
50,000 records with realistic behavioral distributions.
"""

import numpy as np
import pandas as pd
from faker import Faker
import duckdb
import os

Faker.seed(42)
fake = Faker()
np.random.seed(42)

N = 50_000

def simulate_dataset(n=N):
    # --- Relationship age (days since friends added) ---
    days_since_friend_added = np.random.exponential(scale=180, size=n).clip(1, 1200).astype(int)

    # --- Streak length (longer for older friendships) ---
    streak_base = np.random.exponential(scale=12, size=n)
    streak_length = (streak_base * (days_since_friend_added / 180) ** 0.4).clip(0, 365).astype(int)

    # --- Avg response time (hours) — skewed right ---
    avg_response_time_hrs = np.random.lognormal(mean=1.2, sigma=1.1, size=n).clip(0.1, 72)

    # --- Pct snaps opened ---
    open_rate_base = np.random.beta(a=5, b=2, size=n)
    # Faster responders open more
    response_penalty = np.clip(avg_response_time_hrs / 72, 0, 1)
    pct_snaps_opened = (open_rate_base * (1 - 0.4 * response_penalty)).clip(0, 1)

    # --- Shared stories viewed in last 14 days ---
    shared_stories_viewed = np.random.poisson(lam=4, size=n)

    # --- Snap Map mutual checks ---
    snap_map_checks = np.random.poisson(lam=2, size=n)

    # --- Friend suggestion algorithm rank (lower = closer) ---
    friend_suggestion_rank = np.random.randint(1, 200, size=n)

    # --- Notifications received in last 7 days ---
    notification_received_7d = np.random.poisson(lam=3, size=n)

    # --- Days since last snap (primary decay signal) ---
    days_since_last_snap = np.random.exponential(scale=4, size=n).clip(0, 60).astype(int)

    # --- Construct "went_cold" label ---
    # Logistic model: high response time, low open rate, long since last snap → cold
    log_odds = (
        -1.5
        + 0.08  * days_since_last_snap
        + 0.04  * avg_response_time_hrs
        - 1.8   * pct_snaps_opened
        - 0.03  * streak_length
        - 0.12  * shared_stories_viewed
        - 0.08  * snap_map_checks
        + 0.003 * friend_suggestion_rank
        - 0.04  * notification_received_7d
    )
    prob_cold = 1 / (1 + np.exp(-log_odds))
    went_cold = np.random.binomial(1, prob_cold)

    # --- Re-activation label (subset of cold pairs) ---
    # Higher for pairs with long streak history, more shared stories
    log_odds_reactivate = (
        -0.8
        + 0.04  * streak_length
        + 0.10  * shared_stories_viewed
        + 0.12  * snap_map_checks
        - 0.03  * avg_response_time_hrs
        - 0.002 * friend_suggestion_rank
        + 0.05  * notification_received_7d
    )
    prob_reactivate = 1 / (1 + np.exp(-log_odds_reactivate))
    reactivated = np.where(went_cold == 1, np.random.binomial(1, prob_reactivate), 0)

    # --- Treatment (Memory resurfacing nudge) — RCT assignment ---
    treatment = np.where(went_cold == 1, np.random.binomial(1, 0.5, size=n), 0)

    # --- Causal lift from treatment (+14pp on treated cold pairs) ---
    treatment_effect = np.random.binomial(1, 0.143, size=n) * treatment * went_cold
    reactivated = np.clip(reactivated + treatment_effect, 0, 1)

    # --- Time to cold (for survival analysis) ---
    # Pairs that didn't go cold are right-censored at observation window (21 days)
    time_to_cold = np.where(
        went_cold == 1,
        np.random.exponential(scale=8, size=n).clip(1, 20).astype(int),
        21
    )

    df = pd.DataFrame({
        "user_pair_id":            [fake.uuid4()[:8] for _ in range(n)],
        "days_since_last_snap":    days_since_last_snap,
        "streak_length":           streak_length,
        "avg_response_time_hrs":   avg_response_time_hrs.round(2),
        "pct_snaps_opened":        pct_snaps_opened.round(3),
        "shared_stories_viewed":   shared_stories_viewed,
        "snap_map_checks":         snap_map_checks,
        "friend_suggestion_rank":  friend_suggestion_rank,
        "days_since_friend_added": days_since_friend_added,
        "notification_received_7d":notification_received_7d,
        "treatment":               treatment,
        "time_to_cold":            time_to_cold,
        "went_cold":               went_cold,
        "reactivated":             reactivated,
    })

    return df


if __name__ == "__main__":
    print("Simulating 50,000 friend-pair interaction records...")
    df = simulate_dataset()

    out_path = os.path.join(os.path.dirname(__file__), "sample_pairs.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} records → {out_path}")
    print(f"\nCold rate:        {df['went_cold'].mean():.1%}")
    print(f"Re-activation rate (among cold): {df.loc[df['went_cold']==1,'reactivated'].mean():.1%}")
    print(f"Treatment rate (among cold):     {df.loc[df['went_cold']==1,'treatment'].mean():.1%}")
    print(f"\nColumn dtypes:\n{df.dtypes}")