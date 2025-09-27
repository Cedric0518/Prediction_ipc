import os
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

OUTPUTS_ROOT = Path(".")
PROCESSED_SERIES = Path("data/processed/series_food_tidy.csv")  # optionnel (historique)

# ---------- Helpers ----------
def list_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)

def parse_tag(tag: str) -> Tuple[str, str]:
    # tag = 'Province__ProductKey' avec espaces -> '_' ; on retransforme pour l'affichage.
    if "__" not in tag:
        return (tag.replace("_", " "), "")
    prov, prod = tag.split("__", 1)
    return (prov.replace("_", " "), prod.replace("_", " "))

def tag_from(province: str, product: str) -> str:
    return f"{province}__{product}".replace(" ", "_")

def load_forecast_csv(run_dir: Path, tag: str) -> pd.DataFrame:
    f = run_dir / "predictions" / f"forecast_{tag}.csv"
    df = pd.read_csv(f)
    df["Date"] = pd.PeriodIndex(df["Date"], freq="M").to_timestamp()
    return df

def load_advice(run_dir: Path, tag: str) -> Optional[str]:
    f = run_dir / "advice" / f"advice_{tag}.txt"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return None

def load_history(province: str, product_key: str) -> Optional[pd.DataFrame]:
    if not PROCESSED_SERIES.exists():
        return None
    df = pd.read_csv(PROCESSED_SERIES)
    m = (df["Province"].astype(str) == province) & (df["ProductKey"].astype(str) == product_key)
    hist = df.loc[m, ["Date", "Indice_IPC_produit"]].copy()
    if hist.empty:
        return None
    hist["Date"] = pd.PeriodIndex(hist["Date"], freq="M").to_timestamp()
    return hist.sort_values("Date")

# ---------- UI ----------
st.set_page_config(page_title="Phase 2 — Prévisions IPC (Canada)", layout="wide")
st.title("📈 Phase 2 — Prévisions IPC automatisées")

try:
    runs = list_runs(OUTPUTS_ROOT)
    if not runs:
        st.info(
            "Aucun run détecté dans `outputs/`.\n\n"
            "Lance le pipeline :\n"
            "`python ingest_statcan.py` → `python retrain.py` → `python forecast_run.py`\n\n"
            "Ou télécharge l’artefact GitHub Actions et place-le dans `outputs/`."
        )
    else:
        run_dir = st.selectbox("🗂️ Sélection du run", runs, format_func=lambda p: p.name)

        pred_dir = run_dir / "predictions"
        csvs = sorted(pred_dir.glob("forecast_*.csv"))
        if not csvs:
            st.error("Aucun fichier de prévision trouvé dans ce run.")
        else:
            tags = [c.stem.replace("forecast_", "") for c in csvs]
            pairs = [parse_tag(t) for t in tags]  # [(Province, ProductKey)]
            provinces = sorted(set(p for p, _ in pairs))
            products = sorted(set(q for _, q in pairs))

            colA, colB = st.columns(2)
            with colA:
                province = st.selectbox("📍 Province", provinces)
            with colB:
                product = st.selectbox("🍽️ Produit", products)

            tag = tag_from(province, product)
            if tag not in tags:
                st.warning("Pas de prévision pour cette combinaison (province/produit) dans ce run.")
            else:
                df_fc = load_forecast_csv(run_dir, tag)
                advice = load_advice(run_dir, tag)
                df_hist = load_history(province, product)  # peut être None

                # Graphique
                st.subheader(f"Graphique — {product} ({province})")
                fig, ax = plt.subplots()
                if df_hist is not None and not df_hist.empty:
                    ax.plot(df_hist["Date"], df_hist["Indice_IPC_produit"], marker="o", label="Historique")
                ax.plot(df_fc["Date"], df_fc["yhat"], marker="x", label="Prévision 12 mois")
                ax.set_xlabel("Date")
                ax.set_ylabel("Indice IPC (proxy)")
                ax.set_title(f"{product} — {province}")
                ax.legend()
                st.pyplot(fig)

                # Tableaux & téléchargements
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Prévisions (aperçu)**")
                    st.dataframe(df_fc.head(12))
                with col2:
                    png_path = run_dir / "graphs" / f"{tag}.png"
                    if png_path.exists():
                        st.markdown("**Image générée**")
                        st.image(str(png_path))
                    else:
                        st.info("Aucune image PNG trouvée pour ce couple.")

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "💾 Télécharger le CSV prévision",
                        df_fc.to_csv(index=False).encode("utf-8"),
                        file_name=f"forecast_{tag}.csv",
                        mime="text/csv",
                    )
                with dl_col2:
                    if (run_dir / "graphs" / f"{tag}.png").exists():
                        st.download_button(
                            "🖼️ Télécharger le PNG",
                            (run_dir / "graphs" / f"{tag}.png").read_bytes(),
                            file_name=f"{tag}.png",
                            mime="image/png",
                        )

                # Conseils
                st.subheader("🧠 Recommandations")
                if advice:
                    st.text(advice)
                else:
                    st.info("Pas de fichier de conseils pour ce couple (province/produit).")

except Exception as e:
    st.error("Une erreur est survenue dans l’application.")
    st.exception(e)
