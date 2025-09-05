import io
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime
import yaml

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

def load_cfg(path="config_phase2.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _wds_full_table_csv_link(pid: str, lang: str = "en") -> str:
    """
    WDS: getFullTableDownloadCSV -> retourne une URL zip du CSV complet.
    Doc: https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/<PID>/<lang>
    """
    url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/{lang}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    obj = r.json()
    if obj.get("status") != "SUCCESS":
        raise RuntimeError(f"WDS FAILED for PID {pid}: {obj}")
    return obj["object"]  # lien zip

def _download_zip_to_df(zip_url: str) -> pd.DataFrame:
    r = requests.get(zip_url, timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # prend le premier CSV du zip
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    return pd.read_csv(f)
    raise RuntimeError("No CSV found in ZIP.")

def download_table(pid: str, lang: str = "en") -> pd.DataFrame:
    link = _wds_full_table_csv_link(pid, lang)
    return _download_zip_to_df(link)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Harmonise noms fréquents des colonnes T1 : REF_DATE, GEO, Products, UOM, VALUE...
    out = df.rename(columns={c: c.strip().replace(" ", "_") for c in df.columns})
    return out

def filter_cpi_products(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    # Selon la structure de la table par produit (ex. 18-10-0004-13) ou via retail prices
    # On filtre par texte (casefold) dans les colonnes produit/libellé
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["product", "group", "class", "vector", "series", "label"])]
    if not candidates:
        return df  # fallback: retourne tout (puis filtrer plus tard)
    col = candidates[0]
    mask = df[col].astype(str).str.casefold().apply(lambda s: any(k in s for k in [k.lower() for k in keywords]))
    return df[mask]

def filter_provinces(df: pd.DataFrame, provinces: list[str]) -> pd.DataFrame:
    # Filtre par colonne GEO-like
    geo_col = None
    for c in df.columns:
        if c.upper() in ("GEO", "GEOGRAPHY"):
            geo_col = c
            break
    if geo_col is None:
        # essaye colonnes avec 'geo'
        candidates = [c for c in df.columns if "geo" in c.lower()]
        if candidates:
            geo_col = candidates[0]
    if geo_col is None:
        return df
    return df[df[geo_col].isin(provinces)]

def restrict_since(df: pd.DataFrame, start_ym: str) -> pd.DataFrame:
    # REF_DATE typiquement "YYYY-MM"
    if "REF_DATE" in df.columns:
        df = df.copy()
        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
        return df[df["REF_DATE"] >= pd.to_datetime(start_ym)]
    # parfois 'Reference_period' etc.
    for c in df.columns:
        if "REF_DATE" in c.upper() or "REFPER" in c.upper() or "DATE" in c.upper():
            dfx = df.copy()
            dfx[c] = pd.to_datetime(dfx[c], errors="coerce")
            return dfx[dfx[c] >= pd.to_datetime(start_ym)]
    return df

def ingest_all():
    cfg = load_cfg()
    raw_dir = cfg["paths"]["raw"]
    proc_dir = cfg["paths"]["processed"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # 1) CPI (table générale) – utile pour IPC_global / contrôles
    df_cpi = download_table(cfg["tables"]["cpi_all"], cfg["lang"])
    df_cpi = normalize_cols(df_cpi)
    df_cpi = filter_provinces(df_cpi, cfg["provinces"])
    df_cpi = restrict_since(df_cpi, cfg["start_date"])
    df_cpi.to_csv(os.path.join(raw_dir, f"cpi_{cfg['tables']['cpi_all']}.csv"), index=False)

    # 2) Prix de détail – aliments
    df_r = download_table(cfg["tables"]["retail_food"], cfg["lang"])
    df_r = normalize_cols(df_r)
    df_r = filter_provinces(df_r, cfg["provinces"])
    # Filtre par denrées cibles (milk/eggs/meat via keywords)
    parts = []
    for key, meta in cfg["products"].items():
        parts.append(filter_cpi_products(df_r, meta["keywords"]).assign(ProductKey=key))
    df_food = pd.concat(parts, ignore_index=True) if parts else df_r
    df_food = restrict_since(df_food, cfg["start_date"])
    df_food.to_csv(os.path.join(raw_dir, f"retail_food_{cfg['tables']['retail_food']}.csv"), index=False)

    # 3) Construire un jeu "tidy" minimal (REF_DATE, GEO, ProductKey, VALUE)
    value_col = "VALUE" if "VALUE" in df_food.columns else [c for c in df_food.columns if c.upper()=="VALUE"][0]
    geo_col = "GEO" if "GEO" in df_food.columns else [c for c in df_food.columns if "geo" in c.lower()][0]
    date_col = "REF_DATE"
    out = df_food[[date_col, geo_col, "ProductKey", value_col]].rename(
        columns={date_col:"Date", geo_col:"Province", value_col:"Indice_IPC_produit"}
    )
    # Harmonise Province labels (optionnel)
    out["Date"] = pd.to_datetime(out["Date"]).dt.to_period("M").astype(str)
    out.to_csv(os.path.join(proc_dir, "series_food_tidy.csv"), index=False)

    print("✅ Ingestion terminée.")
    return out

if __name__ == "__main__":
    ingest_all()
