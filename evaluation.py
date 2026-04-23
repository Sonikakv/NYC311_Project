import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt

results = []

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # ROC-AUC (safe for models without predict_proba)
    try:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc)

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/level3/confusion_matrices/{name}.png")
    plt.close()

    results.append([name, acc, prec, rec, f1, auc])