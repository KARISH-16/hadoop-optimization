import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame):

    report = {}

    # ---------------- ORIGINAL SHAPE ----------------
    report["original_shape"] = df.shape

    # ---------------- REMOVE DUPLICATES ----------------
    before_dup = df.shape[0]
    df = df.drop_duplicates()
    report["duplicates_removed"] = before_dup - df.shape[0]

    # ---------------- REMOVE NULLS ----------------
    before_null = df.isnull().sum().sum()
    df = df.dropna()
    report["null_rows_removed"] = before_null

    # ---------------- CONVERT TO NUMERIC ----------------
    df = df.apply(pd.to_numeric, errors="coerce")

    # ---------------- HANDLE NaN ----------------
    nan_before = df.isna().sum().sum()
    df = df.fillna(df.median())
    report["nan_filled"] = nan_before

    # ---------------- OUTLIER HANDLING ----------------
    outlier_count = 0

    for col in df.columns:
        if col != "ExecutionTime":

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_count += outliers

            df[col] = np.clip(df[col], lower, upper)

    report["outliers_handled"] = int(outlier_count)

    # ---------------- FINAL SHAPE ----------------
    report["final_shape"] = df.shape

    return df, report