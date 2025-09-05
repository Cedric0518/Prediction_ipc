import os, json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from features import make_lag_features

MODELS_DIR = "models_phase2"

def train_one(series_df: pd.DataFrame, key: tuple[str,str], min_train_points=48):
    # key = (Province, ProductKey)
    df = series_df[(series_df["Province"]==key[0]) & (series_df["ProductKey"]==key[1])].copy()
    df = make_lag_features(df)

    # retire lignes sans lags
    feat_cols = [c for c in df.columns if c.startswith(("lag_","roll_mean_"))] + ["Year","Month"]
    df = df.dropna(subset=feat_cols + ["Indice_IPC_produit"])
    if len(df) < min_train_points:
        return None, None

    X = df[feat_cols]
    y = df["Indice_IPC_produit"]

    # split temporel simple
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, test_idx in tscv.split(X):
        m = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = m.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], p))
    mae_cv = float(np.mean(maes))

    # fit final
    model = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, {"mae_cv": mae_cv, "n_obs": int(len(df))}

def retrain_all(processed_csv="data/processed/series_food_tidy.csv", outdir=MODELS_DIR):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(processed_csv)
    df["Date"] = pd.PeriodIndex(df["Date"], freq="M").astype(str)

    keys = sorted(df[["Province","ProductKey"]].drop_duplicates().itertuples(index=False, name=None))
    metrics = []

    for key in keys:
        model, meta = train_one(df, key)
        if model is None:
            continue
        tag = f"{key[0]}__{key[1]}".replace(" ","_")
        joblib.dump(model, os.path.join(outdir, f"rf_{tag}.pkl"))
        metrics.append({"key": tag, **meta})

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("✅ Ré-entraînement terminé.")

if __name__ == "__main__":
    retrain_all()
