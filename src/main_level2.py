from data_loader import load_data, select_columns
from preprocessing import preprocess
from models import get_classification_models, get_regression_models
from evaluation import evaluate_classification, evaluate_regression, results
from report import generate_decision_report

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# ---------------- FOLDERS ----------------
os.makedirs("outputs/level2/plots", exist_ok=True)
os.makedirs("outputs/level2/confusion_matrices", exist_ok=True)
os.makedirs("outputs/level2/advanced_plots", exist_ok=True)
os.makedirs("outputs/level2/feature_importance", exist_ok=True)

# ---------------- LOAD ----------------
df = load_data("data/nyc_311.csv")
df = select_columns(df)
df = preprocess(df)

df_raw = df.copy()
df = df.sample(n=50000, random_state=42)

# ---------------- TARGET ----------------
plt.figure(figsize=(6,4))
target = df["resolved_in_72hrs"].value_counts().rename({
    0:"Delayed (>72h)",
    1:"Resolved (≤72h)"
})

ax = target.plot(kind="bar", color=["#F44336","#4CAF50"])

for i, v in enumerate(target):
    ax.text(i, v, str(v), ha="center")

plt.title("Resolution Status Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/level2/plots/target.png")
plt.close()

# ---------------- BOROUGH ----------------
df_b = df[df["borough"] != "Unspecified"]

plt.figure(figsize=(7,4))
borough_vals = df_b.groupby("borough")["resolved_in_72hrs"].mean().sort_values()

ax = borough_vals.plot(
    kind="bar",
    color=["#F44336","#FF9800","#FFC107","#8BC34A","#4CAF50"]
)

for i, v in enumerate(borough_vals):
    ax.text(i, v, f"{v:.2f}", ha="center")

plt.xticks(rotation=30, ha="right")
plt.ylabel("Resolution Rate")
plt.title("Resolution Rate by Borough")
plt.tight_layout()
plt.savefig("outputs/level2/plots/borough.png")
plt.close()

# ---------------- HOUR ----------------
plt.figure(figsize=(8,4))
hour_data = df.groupby("hour")["resolved_in_72hrs"].mean()

plt.plot(hour_data, linewidth=2)
plt.axvspan(9,17,color="red",alpha=0.1,label="Peak Hours")

plt.xlabel("Hour of Day")
plt.ylabel("Resolution Rate")
plt.title("Resolution Rate Across Day")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/level2/plots/hour.png")
plt.close()

# ---------------- CORRELATION ----------------
plt.figure(figsize=(10,8))
df_corr = df.corr(numeric_only=True).round(2)

sns.heatmap(
    df_corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/level2/advanced_plots/correlation.png")
plt.close()

# ---------------- MONTH ----------------
trend = df.groupby("month")["resolved_in_72hrs"].mean()

plt.figure(figsize=(7,4))
ax = trend.plot(marker="o", linewidth=2)

for x, y in trend.items():
    ax.text(x, y + 0.005, f"{y:.2f}", ha="center")

plt.xticks([1,2,3], ["Jan","Feb","Mar"])
plt.xlabel("Month")
plt.ylabel("Resolution Rate")
plt.title("Monthly Resolution Improvement Trend")

plt.ylim(trend.min() - 0.02, trend.max() + 0.02)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/level2/advanced_plots/month.png")
plt.close()

# ---------------- PEAK ----------------
plt.figure(figsize=(6,4))
peak_data = df.groupby("is_peak_hour")["resolved_in_72hrs"].mean()
peak_data.index = ["Non-Peak","Peak"]

peak_data.plot(kind="bar", color=["#4CAF50","#F44336"])

plt.ylabel("Resolution Rate")
plt.title("Impact of Peak Hours on Resolution")

plt.tight_layout()
plt.savefig("outputs/level2/advanced_plots/peak.png")
plt.close()

# ---------------- ENCODING ----------------
df_encoded = pd.get_dummies(
    df,
    columns=["complaint_type","borough","agency","time_of_day"],
    drop_first=True
)

X = df_encoded.drop(["resolved_in_72hrs","resolution_time"], axis=1)
y = df_encoded["resolved_in_72hrs"]
X = X.fillna(0)

# ---------------- CLASSIFICATION ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = get_classification_models()

for name, model in models.items():
    evaluate_classification(name, model, X_train, X_test, y_train, y_test)

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
results_df.to_csv("outputs/level2/results.csv", index=False)

# ---------------- FEATURE IMPORTANCE ----------------
rf_model = models["Random Forest"]

imp_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False).head(15)

imp_df = imp_df.iloc[::-1]
imp_df["feature"] = imp_df["feature"].str.replace("_"," ")

plt.figure(figsize=(9,6))
imp_df.plot(kind="barh", x="feature", y="importance", legend=False, color="#2196F3")

plt.xlabel("Importance Score")
plt.title("Top Drivers of Resolution Outcome")

plt.tight_layout(pad=2)
plt.savefig("outputs/level2/feature_importance/importance.png")
plt.close()

# ---------------- REGRESSION ----------------
y_reg = df_encoded["resolution_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_models = get_regression_models()

for name, model in reg_models.items():
    evaluate_regression(name, model, X_train, X_test, y_train, y_test)

# ---------------- REPORT ----------------
generate_decision_report(df_raw, results_df)