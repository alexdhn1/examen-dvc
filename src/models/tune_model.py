
import argparse, os, joblib
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_train", required=True)
    ap.add_argument("--y_train", required=True)
    ap.add_argument("--out_params", required=True)
    ap.add_argument("--out_cv", required=True)
    args = ap.parse_args()

    X_train = pd.read_csv(args.X_train, index_col=0)
    y_train = pd.read_csv(args.y_train, index_col=0).iloc[:,0]

    model = ElasticNet(max_iter=10000, random_state=42)
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0]
    }
    gs = GridSearchCV(
        model, param_grid,
        scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)

    os.makedirs(os.path.dirname(args.out_params), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_cv), exist_ok=True)

    joblib.dump(gs.best_params_, args.out_params)
    pd.DataFrame(gs.cv_results_).to_csv(args.out_cv, index=False)
    print(f"Best params: {gs.best_params_}")

if __name__ == "__main__":
    main()
