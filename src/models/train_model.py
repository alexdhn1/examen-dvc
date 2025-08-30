import argparse, os, joblib
import pandas as pd
from sklearn.linear_model import ElasticNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_train", required=True)
    ap.add_argument("--y_train", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out_model", required=True)
    args = ap.parse_args()

    X_train = pd.read_csv(args.X_train, index_col=0)
    y_train = pd.read_csv(args.y_train, index_col=0).iloc[:,0]

    best_params = joblib.load(args.params)
    model = ElasticNet(max_iter=10000, random_state=42, **best_params)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(model, args.out_model)
    print("Model trained and saved.")

if __name__ == "__main__":
    main()