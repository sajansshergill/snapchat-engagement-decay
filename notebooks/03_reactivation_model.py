# # 03 · Re-Activation Model — XGBoost + SHAP
# **Snapchat Friend Engagement Decay Project**
#
# Goals:
# - Train a binary classifier to score cold pairs by re-activation likelihood
# - Evaluate model performance (AUC-ROC, Precision-Recall, Precision @K)
# - Explain predictions with SHAP — global importance + individual waterfall
# - Build a re-activation scoring table for the top-opportunity pairs
# - Identify thresholds for production nudge targeting

# ## Setup

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

from src.features import load_data, engineer_features, FEATURE_COLS
from src.classifier import prepare_reactivation_data, train_xgb, evaluate_model

DARK, YELLOW, TEAL, RED = "#0D0D0D", "#FFFC00", "#4ECDC4", "#FF6B6B"
LAYOUT = dict(paper_bgcolor=DARK, plot_bgcolor=DARK,
              font=dict(color="white", size=12),
              margin=dict(l=60, r=20, t=50, b=60))

df = load_data("../data/sample_pairs.csv")
df = engineer_features(df)
print(f"Loaded {len(df):,} pairs | Cold: {df['went_cold'].sum():,}")

# ## 1. Prepare Training Data (Cold Pairs Only)

X, y, cold_df = prepare_reactivation_data(df, FEATURE_COLS)
print(f"Cold pairs for modeling: {len(X):,}")
print(f"Class balance — Reactivated: {y.mean():.1%}  |  Stayed cold: {(1-y).mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ## 2. Train XGBoost Classifier

print("Training XGBoost...")
model = train_xgb(X_train, y_train, X_val, y_val)
print(f"Best iteration: {model.best_iteration}")
print(f"Features used:  {model.n_features_in_}")

# ## 3. 5-Fold Cross-Validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_n_estimators = (model.best_iteration + 1) if model.best_iteration is not None else 200
cv_model = xgb.XGBClassifier(
    n_estimators=best_n_estimators, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42, eval_metric="auc", verbosity=0,
)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)

print(f"5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Per-fold:      {[f'{s:.4f}' for s in cv_scores]}")

# ## 4. Test Set Evaluation

metrics = evaluate_model(model, X_test, y_test)
probs   = metrics["probs"]

# ## 5. ROC & Precision-Recall Curves

fpr, tpr, _ = roc_curve(y_test, probs)
prec, rec, _ = precision_recall_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)
ap  = average_precision_score(y_test, probs)

fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC Curve", "Precision-Recall Curve"])

fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
    line=dict(color=YELLOW, width=2.5), name=f"AUC = {auc:.3f}"), row=1, col=1)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
    line=dict(color="#555", dash="dash"), showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
    line=dict(color=TEAL, width=2.5), name=f"AP = {ap:.3f}"), row=1, col=2)
fig.add_hline(y=y_test.mean(), line=dict(color="#555", dash="dash"),
    annotation_text=f"Baseline: {y_test.mean():.2f}", row=1, col=2)

fig.update_layout(**LAYOUT, height=420)
for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
    fig.layout[ax].update(gridcolor="#222", color="white")
fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
fig.update_xaxes(title_text="Recall", row=1, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
fig.update_yaxes(title_text="Precision", row=1, col=2)
fig.show()

# ## 6. Precision @K Curve

sorted_idx  = np.argsort(probs)[::-1]
sorted_true = y_test.values[sorted_idx]

k_values, p_at_k = [], []
for k in range(10, len(sorted_true), 10):
    k_values.append(k / len(sorted_true))
    p_at_k.append(sorted_true[:k].mean())

fig = go.Figure()
fig.add_trace(go.Scatter(x=k_values, y=p_at_k, mode="lines",
    line=dict(color=YELLOW, width=2.5), name="Precision @K"))
fig.add_hline(y=y_test.mean(), line=dict(color=RED, dash="dash"),
    annotation_text=f"Random baseline: {y_test.mean():.1%}", annotation_font_color=RED)
fig.add_vline(x=0.10, line=dict(color=TEAL, dash="dot"),
    annotation_text="Top 10%", annotation_font_color=TEAL)

fig.update_layout(**LAYOUT, height=380,
    title="Precision @K — Targeting Efficiency",
    xaxis=dict(title="Fraction of Cold Pairs Targeted",
               tickformat=".0%", gridcolor="#222", color="white"),
    yaxis=dict(title="Precision (Re-activation Rate)",
               tickformat=".0%", gridcolor="#222", color="white"),
)
fig.show()

top_10_prec = sorted_true[:int(0.10 * len(sorted_true))].mean()
print(f"Precision @Top 10%: {top_10_prec:.1%}  vs  random baseline {y_test.mean():.1%}")
print(f"Lift @Top 10%:      {top_10_prec / y_test.mean():.2f}×")

# ## 7. SHAP Global Feature Importance

try:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]
    importance_label = "Mean |SHAP|"
    explanation_name = "SHAP"
except Exception as exc:
    print(f"SHAP unavailable ({exc}); falling back to XGBoost feature importance.")
    shap_vals = np.tile(model.feature_importances_, (len(X_test), 1))
    importance_label = "XGBoost Importance"
    explanation_name = "Feature Importance"

mean_shap = np.abs(shap_vals).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X_test.columns,
    importance_label: mean_shap
}).sort_values(importance_label, ascending=False).reset_index(drop=True)
shap_df.index += 1

print(f"=== Top 10 Features by {importance_label} ===")
print(shap_df.head(10).to_string())

shap_sorted = shap_df.sort_values(importance_label)
fig = go.Figure(go.Bar(
    x=shap_sorted[importance_label],
    y=shap_sorted["Feature"],
    orientation="h",
    marker=dict(color=[
        YELLOW if v >= shap_df[importance_label].median() else TEAL
        for v in shap_sorted[importance_label]
    ]),
    text=[f"{v:.4f}" for v in shap_sorted[importance_label]],
    textposition="outside", textfont=dict(color="white"),
))
fig.update_layout(**LAYOUT, height=460,
    title=f"Global {explanation_name} — Re-Activation Model",
    xaxis=dict(title=importance_label, gridcolor="#222", color="white"),
    yaxis=dict(color="white"),
)
fig.show()

# ## 8. SHAP Beeswarm — Direction of Feature Effects

top6 = shap_df.head(6)["Feature"].tolist()
fig  = make_subplots(rows=2, cols=3, subplot_titles=top6)

for idx, feat in enumerate(top6):
    r, c      = divmod(idx, 3)
    feat_idx  = list(X_test.columns).index(feat)
    fig.add_trace(go.Scatter(
        x=X_test[feat].values,
        y=shap_vals[:, feat_idx],
        mode="markers",
        marker=dict(
            color=shap_vals[:, feat_idx],
            colorscale=[[0, TEAL], [0.5, "#111"], [1, YELLOW]],
            size=3, opacity=0.6, cmid=0
        ),
        showlegend=False,
        hovertemplate=f"{feat}: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<extra></extra>"
    ), row=r+1, col=c+1)
    fig.add_hline(y=0, line=dict(color="#555", width=1), row=r+1, col=c+1)

fig.update_layout(**LAYOUT, height=520,
    title_text=f"{explanation_name} Values vs Feature Values (Top 6 Features)",
    title_font_color="white")
for ax in fig.layout:
    if ax.startswith("xaxis") or ax.startswith("yaxis"):
        fig.layout[ax].update(gridcolor="#222", color="white")
fig.show()

# ## 9. Individual Pair — SHAP Waterfall

top_pair_idx = np.argmax(probs)
pair_shap    = shap_vals[top_pair_idx]
pair_feats   = X_test.iloc[top_pair_idx]

shap_order = np.argsort(np.abs(pair_shap))[::-1][:10]
feat_names = [X_test.columns[i] for i in shap_order]
feat_shaps = [pair_shap[i] for i in shap_order]
feat_vals  = [pair_feats.iloc[i] for i in shap_order]

fig = go.Figure(go.Waterfall(
    name="SHAP", orientation="h",
    measure=["relative"] * len(feat_shaps) + ["total"],
    x=feat_shaps + [sum(feat_shaps)],
    y=[f"{n}\n= {v:.2f}" for n, v in zip(feat_names, feat_vals)] + ["Final Prediction"],
    connector=dict(line=dict(color="#444")),
    increasing=dict(marker=dict(color=YELLOW)),
    decreasing=dict(marker=dict(color=RED)),
    totals=dict(marker=dict(color=TEAL)),
))
fig.update_layout(**LAYOUT, height=480,
    title=f"{explanation_name} Waterfall — Highest Scored Pair (P(reactivate) = {probs[top_pair_idx]:.1%})",
    xaxis=dict(title=f"{explanation_name} contribution", gridcolor="#222", color="white"),
    yaxis=dict(color="white"),
)
fig.show()

print(f"\nPair Profile:")
print(f"  Streak length:      {pair_feats['streak_length']:.0f} days")
print(f"  Avg response time:  {pair_feats['avg_response_time_hrs']:.1f} hours")
print(f"  Open rate:          {pair_feats['pct_snaps_opened']:.1%}")
print(f"  Predicted score:    {probs[top_pair_idx]:.1%}")

# ## 10. Production Re-Activation Scoring Table

cold_score_df = df[df["went_cold"] == 1].copy().reset_index(drop=True)
cold_score_df["reactivation_score"] = model.predict_proba(cold_score_df[FEATURE_COLS])[:, 1]
cold_score_df["score_decile"]       = pd.qcut(cold_score_df["reactivation_score"], q=10, labels=False) + 1

decile_table = cold_score_df.groupby("score_decile").agg(
    n_pairs=("reactivated", "count"),
    actual_reactivation_rate=("reactivated", "mean"),
    avg_score=("reactivation_score", "mean"),
    avg_streak=("streak_length", "mean"),
).reset_index().sort_values("score_decile", ascending=False)

decile_table["actual_reactivation_rate"] = decile_table["actual_reactivation_rate"].map("{:.1%}".format)
decile_table["avg_score"]  = decile_table["avg_score"].map("{:.1%}".format)
decile_table["avg_streak"] = decile_table["avg_streak"].map("{:.1f}".format)
decile_table.columns = ["Decile", "N Pairs", "Actual Re-Act Rate", "Avg Score", "Avg Streak"]
print(decile_table.to_string(index=False))

# ## 11. Model Summary
#
# | Metric | Value |
# |---|---|
# | 5-Fold CV AUC | **~0.61 ± 0.02** |
# | Test AUC-ROC | **0.61** |
# | Average Precision | **0.57** |
# | Precision @Top Decile | **~62%** vs 46% baseline |
# | Lift @Top 10% | **~1.35×** |
# | Top Feature | `engagement_composite` |
#
# **Next:** Causal inference → `04_causal_inference.py`