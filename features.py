import pandas as pd
import numpy as np

def make_lag_features(df: pd.DataFrame, group_cols=("Province","ProductKey","Unit"), target="Indice_IPC_produit"):
    # suppose df = [Date, Province, ProductKey, Indice_IPC_produit]
    dfx = df.copy()
    dfx["Date"] = pd.PeriodIndex(dfx["Date"], freq="M").to_timestamp()

    # Ajoute une "Unit" par défaut (CAD per unit ou index). Ici, on met 'index'
    if "Unit" not in dfx.columns:
        dfx["Unit"] = "index"

    dfx = dfx.sort_values(["Province","ProductKey","Date"]).reset_index(drop=True)

    for L in [1,3,6,12]:
        dfx[f"lag_{L}"] = dfx.groupby(list(group_cols))[target].shift(L)
    for w in [3,6,12]:
        dfx[f"roll_mean_{w}"] = dfx.groupby(list(group_cols))[target].shift(1).rolling(w).mean()

    # Encodages simples
    dfx["Year"] = dfx["Date"].dt.year
    dfx["Month"] = dfx["Date"].dt.month

    return dfx
