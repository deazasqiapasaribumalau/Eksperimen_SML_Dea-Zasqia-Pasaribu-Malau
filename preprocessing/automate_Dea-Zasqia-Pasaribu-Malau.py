"""
automate_Dea-Zasqia-Pasaribu-Malau.py
=======================================
Script otomatisasi preprocessing dataset Heart Disease.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(filepath):
    print(f"[1/5] Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Shape awal: {df.shape}")
    return df


def handle_missing_values(df):
    print("[2/5] Handling missing values...")
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing_cols = [c for c in num_cols if df[c].isnull().any()]
    if missing_cols:
        imputer = SimpleImputer(strategy="median")
        df[missing_cols] = imputer.fit_transform(df[missing_cols])
        print(f"      Diimputasi (median): {missing_cols}")
    print(f"      Missing values setelah imputasi: {df.isnull().sum().sum()}")
    return df


def handle_duplicates(df):
    print("[3/5] Handling duplicates...")
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"      Duplikat dihapus: {n_before - len(df)}")
    return df


def handle_outliers(df, columns):
    print("[4/5] Handling outliers (IQR capping)...")
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"      {col}: {n_out} outlier di-cap")
    return df


def scale_and_split(df, target_col="target", test_size=0.2, random_state=42):
    print("[5/5] Scaling dan split...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scale_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scale_cols = [c for c in scale_cols if c in X.columns]

    scaler = StandardScaler()
    X = X.copy()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(input_path, output_dir):
    print("=" * 55)
    print("  PIPELINE PREPROCESSING - HEART DISEASE DATASET")
    print("=" * 55)

    OUTLIER_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = handle_outliers(df, OUTLIER_COLS)
    X_train, X_test, y_train, y_test = scale_and_split(df)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("\n" + "=" * 55)
    print("PREPROCESSING SELESAI!")
    print(f"    Output: {output_dir}/")
    print(f"    X_train: {X_train.shape} | X_test: {X_test.shape}")
    print("=" * 55)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="heart_disease_raw.csv")
    parser.add_argument("--output", type=str, default="heart_disease_preprocessing")
    args = parser.parse_args()
    run_preprocessing_pipeline(args.input, args.output)
