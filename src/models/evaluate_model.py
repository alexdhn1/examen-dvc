import argparse, os, json, joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_test", required=True)
    ap.add_argument("--y_test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_predictions", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    X_test = pd.read_csv(args.X_test, index_col=0)
    y_test = pd.read_csv(args.y_test, index_col=0).iloc[:,0]
    model = joblib.load(args.model)

    y_pred = model.predict(X_test)
    preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=X_test.index)
    os.makedirs(os.path.dirname(args.out_predictions), exist_ok=True)
    preds.to_csv(args.out_predictions)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Evaluation done:", metrics)

if __name__ == "__main__":
    main()
