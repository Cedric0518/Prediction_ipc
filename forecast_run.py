import os, json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from features import make_lag_features
from viz import save_line_plot
from llm_recommendations import generate_advice
from pathlib import Path

MODELS_DIR = "models_phase2"
OUTPUTS_DIR = "outputs"

def _future_months(last_ym: str, horizon: int):
    p = pd.Period(last_ym, freq="M")
    return [(p + i).strftime("%Y-%m") for i in range(1, horizon+1)]

def forecast_series(history: pd.DataFrame, model, horizon=12):
    """
    history: colonnes [Date, Province, ProductKey, Indice_IPC_produit]
    Stratégie récursive: on prédit 1 mois, on ajoute au passé, etc.
    Retourne DataFrame future avec colonnes [Date, yhat].
    """
    hist = history.copy()
    hist["Date"] = pd.PeriodIndex(hist["Date"], freq="M").astype(str)
    last_ym = max(hist["Date"])
    fut_dates = _future_months(last_ym, horizon)

    preds = []
    cur = hist.copy()
    for ym in fut_dates:
        fx = make_lag_features(cur)
        feat_cols = [c for c in fx.columns if c.startswith(("lag_","roll_mean_"))] + ["Year","Month"]
        row = fx[fx["Date"] == pd.Period(ym, freq="M").to_timestamp()]
        if row.empty:
            # on fabrique une ligne future à partir de la dernière
            last = fx.sort_values("Date").iloc[-1:].copy()
            last["Date"] = pd.Period(ym, freq="M").to_timestamp()
            row = pd.concat([fx.iloc[0:0], last], ignore_index=True)
        yhat = float(model.predict(row[feat_cols])[0])
        preds.append({"Date": ym, "yhat": yhat})
        # ajoute la prédiction dans l'historique pour la suite
        cur = pd.concat([cur, pd.DataFrame([{
            "Date": ym, "Province": hist["Province"].iloc[0],
            "ProductKey": hist["ProductKey"].iloc[0],
            "Indice_IPC_produit": yhat, "Unit":"index"
        }])], ignore_index=True)
    return pd.DataFrame(preds)

def run_all(processed_csv="data/processed/series_food_tidy.csv", horizon=12, outdir=OUTPUTS_DIR):
    os.makedirs(outdir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y-%m-%d")
    out_csv_dir = os.path.join(outdir, run_tag, "predictions")
    out_png_dir = os.path.join(outdir, run_tag, "graphs")
    out_txt_dir = os.path.join(outdir, run_tag, "advice")
    for d in (out_csv_dir, out_png_dir, out_txt_dir):
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(processed_csv)
    keys = sorted(df[["Province","ProductKey"]].drop_duplicates().itertuples(index=False, name=None))

    all_summaries = []
    for prov, prod in keys:
        tag = f"{prov}__{prod}".replace(" ","_")
        mdl_path = os.path.join(MODELS_DIR, f"rf_{tag}.pkl")
        if not os.path.exists(mdl_path):
            continue
        model = joblib.load(mdl_path)
        hist = df[(df["Province"]==prov) & (df["ProductKey"]==prod)][["Date","Province","ProductKey","Indice_IPC_produit"]].copy()

        fut = forecast_series(hist, model, horizon=horizon)
        fut.to_csv(os.path.join(out_csv_dir, f"forecast_{tag}.csv"), index=False)

        # Graphique (historique + forecast)
        save_line_plot(hist, fut, title=f"{prod} – {prov}", path=os.path.join(out_png_dir, f"{tag}.png"))

        # Conseils via agent IA (fallback automatique si pas d'OPENAI_API_KEY)
        advice_text = generate_advice(province=prov, product=prod, df_fc=fut)
        with open(os.path.join(out_txt_dir, f"advice_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(advice_text)

        all_summaries.append({"key": tag, "last_hist": hist["Date"].max(), "first_forecast": fut["Date"].min(), "last_forecast": fut["Date"].max()})

    with open(os.path.join(outdir, run_tag, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"✅ Prévisions terminées. Résultats dans {os.path.join(outdir, run_tag)}")

if __name__ == "__main__":
    # horizon par défaut: 12
    run_all()
