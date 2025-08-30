#!/usr/bin/env python
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "silica_concentrate"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # ⚠️ Ne garder que les colonnes numériques (le dataset peut contenir une date/horodatage)
    num_df = df.select_dtypes(include="number")
    if TARGET not in num_df.columns:
        raise ValueError(f"Cible '{TARGET}' absente des colonnes numériques : {list(num_df.columns)}")

    X = num_df.drop(columns=[TARGET])
    y = num_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    os.makedirs(args.out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(args.out_dir, "X_train.csv"), index=True)
    X_test.to_csv(os.path.join(args.out_dir, "X_test.csv"), index=True)
    y_train.to_csv(os.path.join(args.out_dir, "y_train.csv"), index=True)
    y_test.to_csv(os.path.join(args.out_dir, "y_test.csv"), index=True)
    print("Split done.")

if __name__ == "__main__":
    main()
