#!/usr/bin/env python
import argparse, os, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # garde uniquement les colonnes numériques
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        raise ValueError("Aucune colonne numérique trouvée pour le scaling.")
    return num

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scaler_out", required=True)
    args = ap.parse_args()

    X_train = pd.read_csv(os.path.join(args.in_dir, "X_train.csv"), index_col=0)
    X_test  = pd.read_csv(os.path.join(args.in_dir, "X_test.csv"), index_col=0)

    X_train = ensure_numeric(X_train)
    X_test  = ensure_numeric(X_test)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)

    X_train_s.to_csv(os.path.join(args.out_dir, "X_train_scaled.csv"))
    X_test_s.to_csv(os.path.join(args.out_dir, "X_test_scaled.csv"))
    joblib.dump(scaler, args.scaler_out)
    print("Scaling done.")

if __name__ == "__main__":
    main()
