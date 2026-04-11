"""
utils/data_processing.py
Helper functions for data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[✔] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numerical values with median."""
    df = df.copy()
    missing = df.isnull().sum()
    if missing.any():
        print("[INFO] Missing values found:")
        print(missing[missing > 0])
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        print("[✔] Missing values filled with median.")
    else:
        print("[✔] No missing values found.")
    return df


def normalize_features(df: pd.DataFrame,
                        cols: list = None,
                        exclude: list = None) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMaxScaler to specified numeric columns.
    Returns (scaled_df, fitted_scaler).
    """
    df = df.copy()
    exclude = exclude or ['target']
    if cols is None:
        cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude]
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    print(f"[✔] Normalized {len(cols)} features: {cols}")
    return df, scaler


def encode_categoricals(df: pd.DataFrame,
                         cat_cols: list = None) -> pd.DataFrame:
    """One-Hot Encode categorical columns (drop_first to avoid multicollinearity)."""
    df = df.copy()
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"[✔] Encoded categorical columns: {cat_cols}")
    else:
        print("[INFO] No categorical columns to encode.")
    return df


def feature_selection(df: pd.DataFrame,
                       target: str = 'target',
                       top_n: int = 10) -> list:
    """Return top N features most correlated with the target."""
    corr = df.corr()[target].abs().drop(target).sort_values(ascending=False)
    top_features = corr.head(top_n).index.tolist()
    print(f"[✔] Top {top_n} features by correlation with '{target}':")
    print(corr.head(top_n))
    return top_features


def preprocess_pipeline(raw_path: str, cleaned_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    load → handle missing → normalize → encode → save.
    """
    df = load_data(raw_path)
    df = handle_missing_values(df)
    df, _ = normalize_features(df, exclude=['target'])
    df = encode_categoricals(df)
    df.to_csv(cleaned_path, index=False)
    print(f"[✔] Cleaned data saved to: {cleaned_path}")
    return df


if __name__ == "__main__":
    preprocess_pipeline(
        raw_path="../data/raw_data.csv",
        cleaned_path="../data/cleaned_data.csv"
    )
