import pandas as pd

def preprocess(df):
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce", format='mixed')
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce", format='mixed')

    df["resolution_time"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    df = df[df["resolution_time"] >= 0]

    df["resolved_in_72hrs"] = (df["resolution_time"] <= 72).astype(int)

    df["hour"] = df["created_date"].dt.hour
    df["day"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.month

    df["is_weekend"] = df["day"].isin([5,6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin(range(9,17)).astype(int)

    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[0,6,12,18,24],
        labels=["night","morning","afternoon","evening"]
    )

    df["complaint_freq"] = df["complaint_type"].map(df["complaint_type"].value_counts())
    df["borough_freq"] = df["borough"].map(df["borough"].value_counts())

    df = df.drop(columns=["created_date","closed_date"])
    df = df.dropna(subset=["resolution_time", "complaint_type", "borough"])

    return df