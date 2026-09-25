import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- LOAD DATA ----------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# ---------------- PREPROCESS ----------------
def preprocess(df):
    df = df.dropna()
    df = df.drop_duplicates()

    # convert everything to numeric (IMPORTANT FIX for your error)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    return df


# ---------------- SPLIT DATA ----------------
def split_data(df):
    X = df.drop("ExecutionTime", axis=1)
    y = df["ExecutionTime"]

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    y = pd.to_numeric(y, errors='coerce')
    y = y.fillna(0)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------- TRAIN GBRT ----------------
def train_gbrt(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model


# ---------------- TRAIN XGBOOST ----------------
def train_xgboost(X_train, y_train):
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    return model


# ---------------- EVALUATION ----------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ---------------- MAIN ----------------
if __name__ == "__main__":

    path = "hadoop_dataset.csv"   # change if needed

    df = load_data(path)
    df = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("\n🚀 Training GBRT...")
    gbrt = train_gbrt(X_train, y_train)
    gbrt_metrics = evaluate(gbrt, X_test, y_test)
    print("GBRT:", gbrt_metrics)

    print("\n🚀 Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate(xgb, X_test, y_test)
    print("XGBoost:", xgb_metrics)

    print("\n✅ Training Completed Successfully")