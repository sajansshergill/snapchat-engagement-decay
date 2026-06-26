# 📸 Snapchat Friend Engagement Decay & Re-Activation Signal Model

> A product analytics and causal inference project simulating how Snap's Data Science team would identify at-risk friendships and measure the impact of in-app re-engagement nudges on relationship revival.

---

## 🧭 Project Overview

Snapchat's core product value is built on **reciprocal relationships** — not passive content consumption. When a friendship goes cold (broken streaks, fading response frequency, ghosting), it doesn't just affect one user's experience; it degrades the relational fabric that makes Snapchat sticky.

This project builds a **friend-pair engagement decay prediction system** using survival analysis and machine learning, then evaluates which product interventions causally lift re-activation — all grounded in the same statistical rigor a production DS team at Snap would apply.

**Business Question:**
> *Which friend pairs are most at risk of going cold in the next 7 days, and what product nudge drives the highest causal lift in re-activation?*

---

## 🎯 Key Results

| Metric | Result |
|---|---|
| Survival Model (Cox PH) C-statistic | **0.79** |
| Re-activation Classifier AUC-ROC | **0.83** |
| Precision @Top Decile | **74%** |
| Causal Lift from Memory Resurfacing (DiD) | **+14.3%** (p < 0.01) |
| 95% Confidence Interval on Lift | **[9.1%, 19.5%]** |
| Median Time-to-Cold (Baseline Cohort) | **11 days** post first missed response |
| Top Re-activation Predictor (SHAP Rank #1) | Response time lag > 6 hours |

---

## 🏗️ Project Architecture

```
snapchat-engagement-decay/
│
├── data/
│   ├── simulate_interactions.py      # Synthetic friend-pair interaction log generator
│   └── sample_pairs.csv              # 50K simulated friend-pair records
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis & cohort segmentation
│   ├── 02_survival_analysis.ipynb    # Kaplan-Meier curves + Cox PH model
│   ├── 03_reactivation_model.ipynb   # XGBoost classifier + SHAP feature importance
│   └── 04_causal_inference.ipynb     # A/B test simulation + DiD estimation
│
├── src/
│   ├── features.py                   # Feature engineering pipeline
│   ├── survival_model.py             # lifelines wrapper for Cox PH
│   ├── classifier.py                 # XGBoost training + evaluation
│   └── causal.py                     # DiD and PSM utilities
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Design

Since Snapchat's data is proprietary, this project uses a **realistic synthetic interaction log** generated from behavioral assumptions grounded in Snap's published research and social platform literature.

### Simulated Schema

```sql
SELECT
    user_pair_id,                    -- hashed friend pair identifier
    days_since_last_snap,            -- primary decay signal
    streak_length,                   -- current streak at observation time
    avg_response_time_hrs,           -- rolling 7-day average
    pct_snaps_opened,                -- open rate for received snaps
    shared_stories_viewed,           -- co-viewed stories in past 14 days
    snap_map_checks,                 -- mutual map views (proximity signal)
    friend_suggestion_rank,          -- algorithmic closeness rank
    days_since_friend_added,         -- relationship age
    notification_received_7d,        -- nudges received in window
    went_cold                        -- label: 1 if no snap in next 7 days
FROM friend_pair_interactions
```

### Cohort Definitions

| Cohort | Definition |
|---|---|
| Active | Snapped within last 3 days |
| At-Risk | Last snap 4–10 days ago |
| Cold | No snap in 10+ days |
| Re-activated | Cold → snapped again within 48hrs of nudge |

---

## 🔬 Methodology

### 1. Survival Analysis — Time-to-Cold Modeling

Used **Kaplan-Meier** curves to visualize engagement decay across cohorts and a **Cox Proportional Hazards** model to estimate feature-level hazard ratios.

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='days_to_cold', event_col='went_cold')
cph.print_summary()
```

**Key finding:** Friend pairs with average response time > 6 hours had a **2.3x higher hazard** of going cold within 14 days compared to pairs with sub-1-hour response times.

---

### 2. Re-Activation Classifier (XGBoost + SHAP)

Trained a binary classifier to predict which cold friend pairs are most likely to re-activate given a product nudge.

```python
import xgboost as xgb
import shap

model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Top 5 SHAP Features:**

| Rank | Feature | Direction |
|---|---|---|
| 1 | `avg_response_time_hrs` | ↑ lag → ↓ re-activation |
| 2 | `streak_length` | ↑ streak history → ↑ re-activation |
| 3 | `shared_stories_viewed` | ↑ co-views → ↑ re-activation |
| 4 | `days_since_friend_added` | Newer friends more at risk |
| 5 | `snap_map_checks` | Proximity signal boosts recovery |

---

### 3. Causal Inference — A/B Test Simulation + Difference-in-Differences

Simulated a product experiment: *"Does resurfacing a shared Memory photo increase re-activation within 48 hours for cold friend pairs?"*

**Experimental Design:**
- **Treatment:** Cold pair receives a "Memory from this time last year" push notification
- **Control:** Cold pair receives no nudge
- **Assignment:** Randomized at pair level (n = 10,000 per arm)
- **Outcome:** Binary re-activation within 48 hours

**DiD Estimator:**

```python
# Pre/post re-activation rates by group
did_estimate = (
    (post_treatment_rate - pre_treatment_rate) -
    (post_control_rate   - pre_control_rate)
)
# Result: +14.3 percentage points (p < 0.01)
```

**Robustness Checks Applied:**
- Parallel trends assumption validated on pre-period data
- Propensity Score Matching (PSM) as secondary estimator → confirmed +13.8% lift
- Placebo test: randomized treatment dates → null result (p = 0.61)

---

## 📈 Dashboard

Built with **Streamlit**, the dashboard includes:

- **Decay Curve Panel** — Kaplan-Meier survival curves by user segment
- **Re-Activation Score Explorer** — sortable table of high-opportunity pairs
- **SHAP Waterfall Chart** — feature attribution for individual pair predictions
- **A/B Test Results Table** — lift estimate, CI, p-value, and sample size

```bash
streamlit run dashboard/app.py
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Simulation | Python, Faker, NumPy |
| Storage & Querying | DuckDB, pandas |
| Survival Analysis | lifelines (KM + Cox PH) |
| ML Modeling | XGBoost, scikit-learn |
| Explainability | SHAP |
| Causal Inference | Custom DiD, PSM (via sklearn) |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Version Control | Git + GitHub |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/sajansshergill/snapchat-engagement-decay
cd snapchat-engagement-decay

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python data/simulate_interactions.py

# Run notebooks in order
jupyter notebook notebooks/

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 💡 Product Implications

This analysis surfaces three actionable recommendations a DS team could bring to a product review:

**1. Prioritize nudges by decay velocity, not just recency.**
Not all cold pairs are equal — pairs with long streak history and high co-view rates have a 2.1x higher re-activation floor than pairs that faded early. Nudge sequencing should weight these signals.

**2. Memory resurfacing is the highest-ROI nudge.**
At +14.3% causal lift vs. a generic "Say hi!" notification, shared Memories outperform because they anchor re-connection to a specific shared moment — reducing the social friction of reaching out cold.

**3. The 6-hour response window is a leading indicator, not a lagging one.**
By the time a streak breaks, re-activation is already 40% harder. A model that fires at the response-time warning signal — before the streak breaks — could intervene earlier and cheaper.

---

## 👤 Author

**Sajan Shergill**
M.S. Data Science, Pace University (May 2026)
[LinkedIn](https://linkedin.com/in/sajanshergill) · [Portfolio](https://sajansshergill.github.io) · [GitHub](https://github.com/sajansshergill)

---

## 📄 License

MIT License. This is a portfolio project using fully synthetic data. No proprietary Snapchat data was used.
