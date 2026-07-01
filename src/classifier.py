"""XGBoost re-activation classifier and SHAP feature importance helpers."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")


def prepare_reactivation_data(df: pd.DataFrame, feature_cols: list):
    """Filter to cold pairs only and prep X, y."""
    cold = df[df["went_cold"] == 1].copy()
    if cold.empty:
        raise ValueError("No cold pairs available for re-activation modeling.")
    X = cold[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = cold["reactivated"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Re-activation target must contain both classes.")
    return X, y, cold


def train_xgb(X_train, y_train, X_val, y_val):
    positives = max(int((y_train == 1).sum()), 1)
    negatives = int((y_train == 0).sum())
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=negatives / positives,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=25,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    ap  = average_precision_score(y_test, probs)

    # Precision at top decile
    threshold = np.percentile(probs, 90)
    top_decile_mask = probs >= threshold
    precision_at_k = float(y_test[top_decile_mask].mean()) if top_decile_mask.sum() > 0 else 0.0

    print(f"\n{'='*40}")
    print(f"  AUC-ROC:             {auc:.4f}")
    print(f"  Avg Precision (AP):  {ap:.4f}")
    print(f"  Precision @Top10%:   {precision_at_k:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, preds, target_names=["stayed_cold", "reactivated"]))

    return {
        "auc": float(auc),
        "ap": float(ap),
        "precision_at_k": precision_at_k,
        "probs": probs,
        "classification_report": classification_report(
            y_test, preds, target_names=["stayed_cold", "reactivated"], output_dict=True
        ),
    }


def plot_roc_pr(y_test, probs, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0D0D0D")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0D0D0D")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax1.plot(fpr, tpr, color="#FFFC00", linewidth=2.5, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="#555", linestyle="--", linewidth=1)
    ax1.set_xlabel("False Positive Rate", color="white")
    ax1.set_ylabel("True Positive Rate", color="white")
    ax1.set_title("ROC Curve — Re-Activation Model", color="white", fontsize=13)
    ax1.legend(facecolor="#1A1A1A", labelcolor="white")

    # PR
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax2.plot(rec, prec, color="#4ECDC4", linewidth=2.5, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall", color="white")
    ax2.set_ylabel("Precision", color="white")
    ax2.set_title("Precision-Recall Curve", color="white", fontsize=13)
    ax2.legend(facecolor="#1A1A1A", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved ROC/PR plot → {save_path}")
    return fig


def compute_shap(model, X_test, save_path: str = None):
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        importance_label = "Mean |SHAP value|"
        title = "Feature Importance — Re-Activation Model (SHAP)"
    except Exception as exc:
        print(f"SHAP unavailable ({exc}); falling back to XGBoost gain importance.")
        gain = getattr(model, "feature_importances_", np.zeros(X_test.shape[1]))
        shap_values = np.tile(gain, (len(X_test), 1))
        importance_label = "XGBoost feature importance"
        title = "Feature Importance — Re-Activation Model"

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_test.columns.tolist()
    order = np.argsort(mean_abs_shap)

    colors = ["#FFFC00" if v > mean_abs_shap.mean() else "#4ECDC4" for v in mean_abs_shap[order]]
    ax.barh(range(len(order)), mean_abs_shap[order], color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], color="white", fontsize=10)
    ax.set_xlabel(importance_label, color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    top_patch = mpatches.Patch(color="#FFFC00", label="Above-avg importance")
    low_patch = mpatches.Patch(color="#4ECDC4", label="Below-avg importance")
    ax.legend(handles=[top_patch, low_patch], facecolor="#1A1A1A", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved SHAP plot → {save_path}")

    # Return ranked feature table
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.index += 1
    return fig, shap_df, shap_values


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    from src.features import load_data, engineer_features, FEATURE_COLS

    df = load_data("data/sample_pairs.csv")
    df = engineer_features(df)

    X, y, cold_df = prepare_reactivation_data(df, FEATURE_COLS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print("Training XGBoost re-activation classifier...")
    model = train_xgb(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test)

    os.makedirs("outputs", exist_ok=True)
    plot_roc_pr(y_test, metrics["probs"], save_path="outputs/roc_pr.png")
    _, shap_df, _ = compute_shap(model, X_test, save_path="outputs/shap_importance.png")
    print("\n=== Top 5 SHAP Features ===")
    print(shap_df.head(5).to_string())

    import pickle
    with open("outputs/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nModel saved → outputs/xgb_model.pkl")