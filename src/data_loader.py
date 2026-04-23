import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def select_columns(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    col_map = {
        "created_date": ["created_date"],
        "closed_date": ["closed_date"],
        "complaint_type": ["complaint_type", "problem_(formerly_complaint_type)"],
        "borough": ["borough"],
        "agency": ["agency"],
        "latitude": ["latitude"],
        "longitude": ["longitude"]
    }

    selected = {}

    for key, options in col_map.items():
        for opt in options:
            if opt in df.columns:
                selected[key] = df[opt]
                break

    return pd.DataFrame(selected)