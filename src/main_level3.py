from data_loader import load_data, select_columns
from preprocessing import preprocess
from models import get_classification_models
from evaluation import evaluate_model, results
from tuning import tune_model
from report import generate_report
from plots import plot_model_comparison, plot_metrics, plot_roc_curve, plot_feature_importance

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

import pandas as pd
import os

# ---------------- FOLDERS ----------------
os.makedirs("outputs/level3/confusion_matrices", exist_ok=True)
os.makedirs("outputs/level3/reports", exist_ok=True)

# ---------------- LOAD ----------------
df = load_data("data/nyc_311.csv")
df = select_columns(df)
df = preprocess(df)

# ---------------- USE 100K ----------------
df = df.sample(n=100000, random_state=42)

# ---------------- ENCODING ----------------
df = pd.get_dummies(df, columns=["complaint_type","borough","agency"], drop_first=True)

X = df.drop(["resolved_in_72hrs","resolution_time"], axis=1)
y = df["resolved_in_72hrs"]

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ---------------- BASELINE ----------------
print("\nBaseline:")
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print("Baseline Accuracy:", dummy.score(X_test, y_test))

# ---------------- SMOTE ----------------
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ---------------- MODELS ----------------
models = get_classification_models()

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# ---------------- TUNE RANDOM FOREST ----------------
print("\nTuning Random Forest...")

rf = RandomForestClassifier()

rf_params = {
    "n_estimators": [100,150],
    "max_depth": [10,20],
    "min_samples_split": [2,5]
}

best_rf = tune_model(rf, rf_params, X_train, y_train)

# ---------------- TUNE GRADIENT BOOSTING ----------------
print("\nTuning Gradient Boosting...")

gb = GradientBoostingClassifier()

gb_params = {
    "n_estimators": [50,100],
    "learning_rate": [0.05,0.1],
    "max_depth": [3,5]
}

best_gb = tune_model(gb, gb_params, X_train, y_train)

# ---------------- EVALUATE TUNED ----------------
print("\nEvaluating Tuned Models...")

evaluate_model("Tuned Random Forest", best_rf, X_train, X_test, y_train, y_test)
evaluate_model("Tuned Gradient Boosting", best_gb, X_train, X_test, y_train, y_test)

# ---------------- CROSS VALIDATION ----------------
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring="f1")
print("\nCV F1:", cv_scores.mean())

# ---------------- ROC ----------------
probs = best_rf.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, probs))

# ---------------- RESULTS ----------------
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1","AUC"])

print("\nFinal Results:\n", results_df)

# ---------------- PLOTS ----------------
plot_model_comparison(results_df)
plot_metrics(results_df)
plot_roc_curve(y_test, probs)
plot_feature_importance(best_rf, X.columns)

# ---------------- REPORT ----------------
generate_report(df, results_df)

print("\nFINAL LEVEL 3 COMPLETE")