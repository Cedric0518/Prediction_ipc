import os
import pandas as pd
import matplotlib.pyplot as plt
from llm_recommendations import generate_advice
from pathlib import Path
import json
import re
from statistics import mean
import streamlit as st

# ---------- Réglages ----------
BASE_DIR = Path(__file__).resolve().parent
def _outputs_root() -> Path:
    # 1) variable d’environnement (prioritaire)
    v = os.getenv("OUTPUTS_ROOT")
    if v:
        return Path(v).resolve()
    # 2) secret Streamlit (utile surtout en Cloud)
    try:
        if "OUTPUTS_ROOT" in st.secrets:
            return Path(st.secrets["OUTPUTS_ROOT"]).resolve()
    except Exception:
        pass
    # 3) défaut local
    return (BASE_DIR / "outputs").resolve()

OUTPUTS_ROOT = _outputs_root()
PROCESSED_SERIES = BASE_DIR / "data/processed/series_food_tidy.csv"
st.caption(f"OUTPUTS_ROOT détecté : {OUTPUTS_ROOT}")


PAGE_TITLE = "🍁 Prévisions d'Indice de Prix à la Consommation (IPC) alimentaires – Canada "
MODEL_NAME = "Random Forest"
MODEL_NOTES = "lags(1,3,6,12) + moyennes(3,6,12), TSCV, 12 mois récursifs"

# ---------- Styles (CSS) ----------
st.set_page_config(page_title="Phase 2 — Prévisions IPC", layout="wide")
st.markdown(
    """
    <style>
      /* compact layout */
      .block-container {padding-top: 1rem; padding-bottom: 2rem; }

      /* ticker bar */
      .ticker-wrap {
        position: sticky; top: 0; z-index: 999;
        background: linear-gradient(90deg, #0ea5e9, #22c55e);
        padding: .35rem 0; margin-bottom: .5rem; border-radius: .5rem;
        overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.08);
      }
      .ticker {
        display: inline-block; white-space: nowrap;
        animation: scroll 22s linear infinite;
        color: white; font-weight: 600; letter-spacing: .2px;
      }
      .ticker span{ margin: 0 .75rem; opacity: .95; }
      @keyframes scroll {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
      }

      /* cards */
      .card {
        border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,.04); background: white;
      }
      .muted { color: #6b7280; font-size: .9rem; }
      .pill {
        display:inline-block; padding:.2rem .6rem; border-radius:999px;
        font-size:.8rem; background:#eef2ff; color:#4f46e5; font-weight:600;
        border:1px solid #e5e7eb; margin-right:.4rem;
      }
      .adv-box {
        border-left: 4px solid #16a34a; padding:.5rem .9rem; background:#f8fafc;
        border-radius: 8px; margin-top:.4rem;
      }
      /* hide default footer */
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def list_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)

def parse_tag(tag: str) -> tuple[str, str]:
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

def load_history(province: str, product_key: str) -> pd.DataFrame | None:
    if not PROCESSED_SERIES.exists():
        return None
    df = pd.read_csv(PROCESSED_SERIES)

    m = (df["Province"].astype(str) == province) & (df["ProductKey"].astype(str) == product_key)
    hist = df.loc[m, ["Date", "Indice_IPC_produit"]].copy()
    if hist.empty:
        return None

    # -> une valeur par mois : moyenne (tu peux mettre 'median' si tu préfères)
    hist["Date"] = pd.PeriodIndex(hist["Date"], freq="M").to_timestamp()
    hist = (hist
            .groupby("Date", as_index=False)
            .agg(Indice_IPC_produit=("Indice_IPC_produit", "mean"))
            .sort_values("Date"))

    # Option: ajoute une moyenne glissante pour lisser la courbe
    hist["MA6"] = hist["Indice_IPC_produit"].rolling(6, min_periods=1).mean()
    return hist

def load_advice_text(run_dir: Path, tag: str) -> str | None:
    f = run_dir / "advice" / f"advice_{tag}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else None

def advice_sections(text: str) -> dict[str, str]:
    """
    Découpe le .txt en sections via titres '## Consommateurs', '## Agriculteurs / Investisseurs', '## Gouvernements'.
    Retourne un dict {section: contenu_markdown}. Si échec → {'Conseils': texte complet}.
    """
    if not text:
        return {}

    import re
    parts = {}
    # NOTE: ici on matche la vraie fin de section avec \n## (et non \\n##)
    pattern = r"(?s)##\s*(Consommateurs|Agriculteurs\s*/\s*Investisseurs|Gouvernements)\s*(.*?)(?=\n##|\Z)"
    for m in re.finditer(pattern, text):
        title = m.group(1).replace("  ", " ")
        body = m.group(2).strip()
        parts[title] = body

    # si on n'a rien découpé, renvoyer tout
    return parts if parts else {"Conseils": text}


def _infer_horizon_months(run_dir: Path) -> int | None:
    """Déduit l’horizon (nb de mois) en lisant un CSV de forecast."""
    pred_dir = run_dir / "predictions"
    for p in sorted(pred_dir.glob("forecast_*.csv")):
        try:
            df = pd.read_csv(p, nrows=10000)
            return int(len(df))
        except Exception:
            continue
    return None

def load_summary(run_dir: Path) -> dict:
    """
    Lit outputs/<run>/summary.json et renvoie toujours un dict.
    - Si le JSON est déjà un objet -> on le renvoie tel quel.
    - Si c'est une liste d'objets -> on calcule des moyennes (mae, rmse, r2) et un horizon.
    - Sinon -> {}.
    """
    f = run_dir / "summary.json"
    if not f.exists():
        # On tente d'inférer juste l'horizon pour l’UI
        return {"horizon_months": _infer_horizon_months(run_dir)}

    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {"horizon_months": _infer_horizon_months(run_dir)}

    # Cas 1 : déjà un dict
    if isinstance(data, dict):
        # On complète gentiment s'il manque l'horizon
        if "horizon_months" not in data or data["horizon_months"] in (None, 0):
            data["horizon_months"] = _infer_horizon_months(run_dir)
        return data

    # Cas 2 : liste -> agréger
    if isinstance(data, list):
        rows = [d for d in data if isinstance(d, dict)]
        def avg(keys):
            vals = []
            for d in rows:
                for k in keys:
                    v = d.get(k)
                    if isinstance(v, (int, float)):
                        vals.append(float(v)); break
            return (round(mean(vals), 4) if vals else None)

        agg = {
            "avg_mae":  avg(["mae", "MAE"]),
            "avg_rmse": avg(["rmse", "RMSE"]),
            "avg_r2":   avg(["r2", "R2", "r2_score"]),
        }
        hz = None
        # chercher horizon dans la liste
        for d in rows:
            for k in ("horizon_months", "horizon"):
                if isinstance(d.get(k), (int, float)) and d.get(k) > 0:
                    hz = int(d[k]); break
            if hz: break
        if not hz:
            hz = _infer_horizon_months(run_dir)
        agg["horizon_months"] = hz
        # tu peux aussi propager une date lisible si présente
        for d in rows:
            for k in ("last_observation", "last_date"):
                if isinstance(d.get(k), str):
                    agg["last_observation"] = d[k]
                    break
            if "last_observation" in agg: break
        return agg

    # Cas 3 : autre type
    return {"horizon_months": _infer_horizon_months(run_dir)}

def _get_openai_key() -> tuple[str | None, str | None]:
    """
    Retourne (clé, source) où source ∈ {"ENV","SECRETS"}.
    Si trouvée dans secrets, on la recopie dans os.environ pour l'SDK.
    """
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k, "ENV"
    try:
        if "OPENAI_API_KEY" in st.secrets:
            k = str(st.secrets["OPENAI_API_KEY"])
            os.environ["OPENAI_API_KEY"] = k
            return k, "SECRETS"
    except Exception:
        pass
    return None, None

# ---------- UI : Header + Ticker ----------
st.title(PAGE_TITLE)

runs = list_runs(OUTPUTS_ROOT)
if not runs:
    st.info(
        "Aucun run détecté dans `outputs/`.\n\n"
        "Lance le pipeline : `python ingest_statcan.py` → `python retrain.py` → `python forecast_run.py`\n\n"
        "Ou dézippe l’artefact GitHub Actions dans `outputs/`."
    )
    st.stop()

# Sélecteurs (sidebar)
with st.sidebar:
    st.header("⚙️ Filtres")
    run_dir = st.selectbox("🗂️ Run (YYYY-MM-DD)", runs, format_func=lambda p: p.name)
    pred_dir = run_dir / "predictions"
    tags = sorted([c.stem.replace("forecast_", "") for c in pred_dir.glob("forecast_*.csv")])
    pairs = [parse_tag(t) for t in tags]
    provinces = sorted(set(p for p, _ in pairs))
    products = sorted(set(q for _, q in pairs))
    province = st.selectbox("📍 Province", provinces)
    product = st.selectbox("🍽️ Produit", products)

# KPI + ticker
summary = load_summary(run_dir)
avg_mae = summary.get("avg_mae")
avg_rmse = summary.get("avg_rmse")
avg_r2 = summary.get("avg_r2")
horizon = summary.get("horizon_months", 12)
last_obs = summary.get("last_observation", "")

k1, k2, k3, k4 = st.columns(4)
k1.metric("MAE moyen", f"{avg_mae:.2f}" if avg_mae is not None else "—")
k2.metric("RMSE moyen", f"{avg_rmse:.2f}" if avg_rmse is not None else "—")
k3.metric("R² moyen", f"{avg_r2:.3f}" if avg_r2 is not None else "—")
k4.metric("Horizon", f"{horizon} mois")

ticker_msgs = [
    f"Modèle: {MODEL_NAME}",
    f"Horizon: {horizon} mois",
    f"Dernier dataset: {last_obs or run_dir.name}",
    f"Notes: {MODEL_NOTES}",
]
if avg_mae is not None: ticker_msgs.insert(1, f"MAE≈{avg_mae:.2f}")
if avg_r2 is not None: ticker_msgs.insert(2, f"R²≈{avg_r2:.3f}")

st.markdown(
    f"""<div class='ticker-wrap'><div class='ticker'>{"".join(f"<span>• {m}</span>" for m in ticker_msgs)}</div></div>""",
    unsafe_allow_html=True,
)

# ---------- Données & Graph ----------
tag = tag_from(province, product)
if tag not in tags:
    st.warning("Pas de prévision pour cette combinaison (province/produit) dans ce run.")
    st.stop()

df_fc = load_forecast_csv(run_dir, tag)
df_hist = load_history(province, product)

st.subheader(f"📈 {product} — {province}")
with st.container():
    fig, ax = plt.subplots()

    if df_hist is not None and not df_hist.empty:
        # nuage de points (historique) + moyenne glissante (plus lisible)
        ax.scatter(
            df_hist["Date"], df_hist["Indice_IPC_produit"],
            s=10, alpha=0.25, label="Historique (points)"
        )
        if "MA6" in df_hist.columns:
            ax.plot(
                df_hist["Date"], df_hist["MA6"],
                linewidth=2, label="Historique (moyenne 6 mois)"
            )

    ax.plot(df_fc["Date"], df_fc["yhat"], marker="x", label=f"Prévision {len(df_fc)} mois")
    ax.set_xlabel("Date"); ax.set_ylabel("Indice IPC (proxy)")
    ax.legend()
    st.pyplot(fig)

# ---------- Téléchargements ----------
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "💾 Télécharger le CSV prévision",
        df_fc.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{tag}.csv", mime="text/csv",
    )
with c2:
    png = (run_dir / "graphs" / f"{tag}.png")
    if png.exists():
        st.download_button(
            "🖼️ Télécharger le PNG",
            png.read_bytes(), file_name=f"{tag}.png", mime="image/png",
        )

# ---------- Conseils : boutons par catégorie ----------
st.subheader("🧠 Recommandations")


adv_text = load_advice_text(run_dir, tag)
sections = advice_sections(adv_text or "")

# Ordre préféré si présent, sinon on garde l'ordre trouvé
preferred = ["Consommateurs", "Agriculteurs / Investisseurs", "Gouvernements"]
available = [lbl for lbl in preferred if lbl in sections] or (list(sections.keys()) or ["Conseils"])

# Sélecteur robuste (horizontal)
choice = st.radio("Voir les conseils pour :", options=available, index=0, horizontal=True)

def render_box(md: str):
    st.markdown(f"<div class='adv-box'>{md}</div>", unsafe_allow_html=True)

# Affichage
render_box(sections.get(choice, sections.get(available[0], "Aucun contenu.")))

# --- Ligne d'action ---
st.divider()
colA, colB = st.columns([1, 2])

with colA:
    # Indicateur de disponibilité de la clé (ENV ou secrets)
    key, src = _get_openai_key()
    if key:
        st.caption(f"🔐 IA: **ON** (source: {src})")
    else:
        st.caption("🔐 IA: **OFF** — définis `OPENAI_API_KEY` pour activer la génération.")

with colB:
    regen = st.button("🤖 (Re)générer les conseils avec l’IA", type="primary", help="Génère un nouveau texte pour ce couple (province, produit).")

if regen:
    if not key:
        st.warning("Aucune clé trouvée. En local : `export OPENAI_API_KEY=\"sk-...\"` puis relance Streamlit. Sur Cloud : ajoute le secret dans Settings → Secrets.")
    else:
        try:
            # df_fc est déjà chargé plus haut (prévisions 12 mois pour ce couple)
            new_text = generate_advice(province=province, product=product, df_fc=df_fc)

            advice_dir = run_dir / "advice"
            advice_dir.mkdir(parents=True, exist_ok=True)
            (advice_dir / f"advice_{tag}.txt").write_text(new_text, encoding="utf-8")

            st.success("Conseils régénérés via l’IA ✅")
            st.rerun()
        except Exception as e:
            st.error("Échec de la régénération via l’IA.")
            st.exception(e)

st.caption(f"Modèle: {MODEL_NAME} • {MODEL_NOTES}")
