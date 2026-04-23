from data_loader import load_data, select_columns
from preprocessing import preprocess
from eda import run_eda
from models import get_models
from evaluation import evaluate_model, results

from sklearn.model_selection import train_test_split
import pandas as pd

df = load_data("data/nyc_311.csv")

df = select_columns(df)

df = preprocess(df)

df = df.sample(n=50000, random_state=42)

run_eda(df)

df = pd.get_dummies(
    df,
    columns=["complaint_type", "borough", "agency", "time_of_day"],
    drop_first=True
)

X = df.drop("resolved_in_72hrs", axis=1)
y = df["resolved_in_72hrs"]


X = X.fillna(0)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = get_models()

for name, model in models.items():
    try:
        evaluate_model(name, model, X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"{name} failed:", e)

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

print("\nModel Comparison:\n")
print(results_df)

results_df.to_csv("outputs/results.csv", index=False)