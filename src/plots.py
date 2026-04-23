import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_model_comparison(results_df):
    plt.figure(figsize=(12,6))
    sns.barplot(x="Model", y="F1", data=results_df)
    plt.xticks(rotation=45)
    plt.title("Model Comparison (F1 Score)")
    plt.tight_layout()
    plt.savefig("outputs/level3/model_comparison.png")
    plt.close()


def plot_metrics(results_df):
    plt.figure(figsize=(12,6))
    results_df.set_index("Model")[["Accuracy","Precision","Recall","F1"]].plot(kind="bar")
    plt.xticks(rotation=45)
    plt.title("Model Metrics Comparison")
    plt.tight_layout()
    plt.savefig("outputs/level3/all_metrics.png")
    plt.close()


def plot_roc_curve(y_test, probs):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/level3/roc_curve.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    import numpy as np

    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top 20 Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/level3/feature_importance.png")
    plt.close()