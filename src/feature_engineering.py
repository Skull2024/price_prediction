# src/feature_engineering.py

import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.weekday
        df["days_from_start"] = (df["date"] - df["date"].min()).dt.days

    df["room_density"] = df["room"] / (df["square"] + 1)
    df["is_studio"] = (df["room"] <= 1).astype(int)

    return df
