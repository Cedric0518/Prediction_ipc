# ingest_statcan.py
# Télécharge les tables StatCan (CSV zip) via lien direct "base8" + fallback WDS,
# filtre provinces/produits, borne la période depuis start_date,
# et génère un dataset "tidy" : data/processed/series_food_tidy.csv
#
# Dépendances : requests, pandas, PyYAML
#   pip install requests pandas PyYAML

import io
import os
import zipfile
import argparse
from typing import List, Optional, Tuple

import requests
import pandas as pd
import yaml

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

# ======================= CONFIG =======================

def load_cfg(path: str = "config_phase2.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ======================= PID / URL =======================

def _pid_formats(pid: str) -> Tuple[str, str, str]:
    """
    Retourne (pid_numeric_10, pid_hyphen, pid_base8)
      - Accepte '1810000401' (numérique 10),
               '18-10-0004-01' (avec tirets),
               ou '18100004' (base8).
      - pid_base8 = 8 premiers chiffres => utilisé par l'URL CSV direct.
    """
    p = pid.strip()
    if "-" in p:
        # hyphen -> digits
        digits = p.replace("-", "")
        if len(digits) < 10:
            raise ValueError(f"PID hyphen inattendu: {pid}")
        pid_numeric_10 = digits[:10]
        pid_base8 = digits[:8]
        return pid_numeric_10, p, pid_base8
    else:
        if len(p) == 10:
            pid_numeric_10 = p
            pid_hyphen = f"{p[0:2]}-{p[2:4]}-{p[4:8]}-{p[8:10]}"
            pid_base8 = p[:8]
            return pid_numeric_10, pid_hyphen, pid_base8
        elif len(p) == 8:
            pid_base8 = p
            pid_numeric_10 = p + "01"  # par défaut, utile pour WDS fallback
            pid_hyphen = f"{p[0:2]}-{p[2:4]}-{p[4:8]}-01"
            return pid_numeric_10, pid_hyphen, pid_base8
        else:
            raise ValueError(f"PID numérique inattendu: {pid}")

def _csv_full_table_url(pid_base8: str, lang: str = "en") -> str:
    """
    URL direct (ZIP) "CSV Download entire table" :
      https://www150.statcan.gc.ca/n1/tbl/csv/<base8>-eng.zip
    """
    suffix = "eng" if lang.lower().startswith("en") else "fra"
    return f"https://www150.statcan.gc.ca/n1/tbl/csv/{pid_base8}-{suffix}.zip"

def _wds_full_table_csv_link(pid_numeric_10: str, lang: str = "en") -> str:
    """
    Fallback WDS -> renvoie un lien ZIP pour la table complète.
    """
    url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid_numeric_10}/{lang}"
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    obj = r.json()
    if obj.get("status") != "SUCCESS" or "object" not in obj:
        raise RuntimeError(f"WDS failed for PID {pid_numeric_10}: {obj}")
    return obj["object"]

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
    """
    1) Essaie CSV direct avec pid_base8 (…/n1/tbl/csv/<base8>-eng.zip)
    2) Fallback WDS (getFullTableDownloadCSV) avec pid_numeric_10
    """
    pid_numeric_10, pid_hyphen, pid_base8 = _pid_formats(pid)
    # 1) Direct CSV (table entière)
    try:
        url = _csv_full_table_url(pid_base8, lang)
        print(f"Trying CSV direct (base8): {url}")
        return _download_zip_to_df(url)
    except Exception as e1:
        print(f"CSV direct failed for {pid_base8}: {e1}")
        # 2) Fallback WDS
        try:
            link = _wds_full_table_csv_link(pid_numeric_10, lang)
            print(f"Trying WDS link: {link}")
            return _download_zip_to_df(link)
        except Exception as e2:
            raise RuntimeError(
                f"Both CSV-direct and WDS failed for PID {pid}. "
                f"CSV err: {e1} | WDS err: {e2}"
            )

# ======================= Préparation / Filtres =======================

def _casefold_list(xs: List[str]) -> List[str]:
    return [x.casefold() for x in xs]

def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Trouve une colonne par noms possibles (insensible à la casse, tolère substring).
    """
    cols = list(df.columns)
    folded = {c.casefold(): c for c in cols}
    # match exact (case-insensitive)
    for cand in candidates:
        if cand.casefold() in folded:
            return folded[cand.casefold()]
    # match 'contient'
    for c in cols:
        low = c.casefold()
        if any(cand.casefold() in low for cand in candidates):
            return c
    return None

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip() for c in df.columns})

def restrict_since(df: pd.DataFrame, start_ym: str) -> pd.DataFrame:
    date_col = get_col(df, ["REF_DATE", "Reference period", "Date", "REFDATE"])
    if date_col is None:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[out[date_col] >= pd.to_datetime(start_ym)]

def filter_provinces(df: pd.DataFrame, provinces: List[str]) -> pd.DataFrame:
    geo_col = get_col(df, ["GEO", "Geography", "GEOGRAPHY"])
    if geo_col is None:
        return df
    wanted = set(_casefold_list([str(x) for x in provinces]))
    mask = df[geo_col].astype(str).str.casefold().isin(wanted)
    return df[mask]

def filter_by_keywords(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """
    Filtre sur une colonne produit probable (Products/Series/Label/Food/Category/Description).
    """
    candidates = ["Products", "Product", "Series", "Label", "Food", "Category", "Description"]
    col = get_col(df, candidates)
    if col is None:
        return df
    keys = _casefold_list(keywords)
    mask = df[col].astype(str).str.casefold().apply(lambda s: any(k in s for k in keys))
    return df[mask].copy()

# ======================= Ingestion principale =======================

def ingest_all(cfg_path: str = "config_phase2.yaml") -> pd.DataFrame:
    cfg = load_cfg(cfg_path)

    raw_dir = cfg["paths"]["raw"]
    proc_dir = cfg["paths"]["processed"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    start_date = cfg.get("start_date", "2000-01")
    lang = cfg.get("lang", "en")

    # 1) CPI (table générale)
    print("⬇️  Téléchargement CPI (table générale)...")
    df_cpi = download_table(cfg["tables"]["cpi_all"], lang)
    df_cpi = normalize_cols(df_cpi)
    df_cpi = filter_provinces(df_cpi, cfg["provinces"])
    df_cpi = restrict_since(df_cpi, start_date)
    cpi_out = os.path.join(raw_dir, f"cpi_{cfg['tables']['cpi_all']}.csv")
    df_cpi.to_csv(cpi_out, index=False)
    print(f"✅ CPI sauvegardé: {cpi_out} ({len(df_cpi):,} lignes)")

    # 2) Prix de détail – aliments
    print("⬇️  Téléchargement Prix de détail (aliments)...")
    df_r = download_table(cfg["tables"]["retail_food"], lang)
    df_r = normalize_cols(df_r)
    df_r = filter_provinces(df_r, cfg["provinces"])
    df_r = restrict_since(df_r, start_date)
    retail_out = os.path.join(raw_dir, f"retail_food_{cfg['tables']['retail_food']}.csv")
    df_r.to_csv(retail_out, index=False)
    print(f"✅ Retail food sauvegardé: {retail_out} ({len(df_r):,} lignes)")

    # 3) Dataset "tidy": Date, Province, ProductKey, Indice_IPC_produit
    print("🧹 Construction du dataset 'tidy' (Date, Province, ProductKey, Indice_IPC_produit)...")
    date_col = get_col(df_r, ["REF_DATE", "Reference period", "Date"])
    geo_col  = get_col(df_r, ["GEO", "Geography", "GEOGRAPHY"])
    val_col  = get_col(df_r, ["VALUE", "Value"])
    uom_col  = get_col(df_r, ["UOM", "Unit", "Units"])
    prod_col = get_col(df_r, ["Products", "Product", "Series", "Label", "Food", "Category"])

    if any(c is None for c in [date_col, geo_col, val_col]):
        missing = [("date_col", date_col), ("geo_col", geo_col), ("val_col", val_col)]
        raise RuntimeError(f"Colonnes essentielles manquantes dans retail_food : {missing}")

    parts = []
    for key, meta in cfg["products"].items():
        dff = filter_by_keywords(df_r, meta.get("keywords", []))
        if dff.empty:
            print(f"⚠️  Aucun match pour {key} avec keywords={meta.get('keywords')}")
            continue
        tmp = pd.DataFrame({
            "Date": pd.to_datetime(dff[date_col], errors="coerce").dt.to_period("M").astype(str),
            "Province": dff[geo_col].astype(str),
            "ProductKey": key,
            "Indice_IPC_produit": pd.to_numeric(dff[val_col], errors="coerce")
        })
        if prod_col:
            tmp["ProductLabel"] = dff[prod_col].astype(str)
        if uom_col:
            tmp["Unit"] = dff[uom_col].astype(str)
        else:
            tmp["Unit"] = "index"
        parts.append(tmp)

    if not parts:
        raise RuntimeError("Aucun segment produit (milk/eggs/meat) détecté après filtrage.")

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["Indice_IPC_produit"]).sort_values(
        ["Province", "ProductKey", "Date"]
    ).reset_index(drop=True)

    processed_path = os.path.join(proc_dir, "series_food_tidy.csv")
    out.to_csv(processed_path, index=False)
    print(f"✅ Tidy dataset sauvegardé: {processed_path} ({len(out):,} lignes)")
    return out

# ======================= CLI =======================

def main():
    ap = argparse.ArgumentParser(description="Ingestion StatCan (CPI + Retail Food) -> tidy CSV")
    ap.add_argument("--config", default="config_phase2.yaml", help="Chemin du fichier YAML de configuration")
    args = ap.parse_args()
    ingest_all(args.config)

if __name__ == "__main__":
    main()
