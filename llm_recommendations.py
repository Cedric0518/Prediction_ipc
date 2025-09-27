import os
import pandas as pd


import os
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd

# --- Configuration via variables d'environnement ---
# Modèle OpenAI (surcharge possible)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class TrendSummary:
    direction: str         # "hausse" | "baisse" | "stable"
    pct12m: float          # variation % entre premier et dernier point
    vol: float             # volatilité moyenne (% absolue des returns)
    start: Optional[float] # première valeur yhat
    end: Optional[float]   # dernière valeur yhat
    n: int                 # nombre de points utilisés


def _compute_trend(df_fc: pd.DataFrame):
    """Retourne dict: {'direction','pct12m','vol'} à partir de df_fc['yhat']."""
    if "yhat" not in df_fc.columns or df_fc["yhat"].dropna().empty:
        return {"direction": "stable", "pct12m": 0.0, "vol": 0.0}
    y = pd.to_numeric(df_fc["yhat"], errors="coerce").dropna().reset_index(drop=True)
    if len(y) < 2:
        v0 = float(y.iloc[0])
        return {"direction": "stable", "pct12m": 0.0, "vol": 0.0}
    start, end = float(y.iloc[0]), float(y.iloc[-1])
    pct = 0.0 if start == 0 else (end - start) / start * 100.0
    vol = float(y.pct_change().abs().mean() * 100.0)
    if pct > 1.0:
        direction = "hausse"
    elif pct < -1.0:
        direction = "baisse"
    else:
        direction = "stable"
    return {"direction": direction, "pct12m": float(pct), "vol": vol}

def _fallback(province: str, product: str, t: dict) -> str:
    dir_txt = {"hausse": "en **hausse**", "baisse": "en **baisse**", "stable": "**stable**"}.get(t["direction"], "stable")
    header = (
        f"# Recommandations — {product} ({province})\n\n"
        f"Tendance prévue sur ~12 mois : {dir_txt} (~{t['pct12m']:.1f}%).\n\n"
    )
    cons = [
        "Ajuster le budget alimentaire mensuel.",
        "Comparer enseignes/marques (MDD), surveiller les promotions.",
        "Anticiper les achats non périssables en période favorable.",
        "Prévoir des substitutions proches si le produit devient onéreux."
    ]
    agri = [
        "Adapter progressivement prix et volumes à la tendance.",
        "Sécuriser les intrants (contrats/couvertures simples) si volatilité élevée.",
        "Diversifier les canaux de vente, surveiller la rotation des stocks."
    ]
    gov = [
        "Suivre l’inflation alimentaire et cibler les aides si besoin.",
        "Renforcer transparence et concurrence dans la chaîne.",
        "Communiquer sur des substitutions abordables et nutritives."
    ]
    def bullets(lines): return "\n".join(f"- {x}" for x in lines)
    return (
        header +
        "## Consommateurs\n" + bullets(cons) + "\n\n" +
        "## Agriculteurs / Investisseurs\n" + bullets(agri) + "\n\n" +
        "## Gouvernements\n" + bullets(gov) + "\n"
    )

def _call_openai(system_prompt: str, user_prompt: str) -> str:
    # Import local pour ne pas imposer openai si fallback
    from openai import OpenAI
    client = OpenAI()  # lit OPENAI_API_KEY
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text

def generate_advice(province: str, product: str, df_fc: pd.DataFrame) -> str:
    """
    >>> texte = generate_advice("Quebec", "eggs", df_fc)
    Retourne un Markdown avec 3 sections : ## Consommateurs, ## Agriculteurs / Investisseurs, ## Gouvernements.
    Utilise OpenAI si OPENAI_API_KEY est défini, sinon fallback déterministe.
    """
    trend = _compute_trend(df_fc)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback(province, product, trend)

    system = (
        "Tu es un conseiller économique canadien. Rédige en français, clair et actionnable. "
        "Structure EXACTEMENT en trois sections Markdown : "
        "## Consommateurs, ## Agriculteurs / Investisseurs, ## Gouvernements. "
        "3–6 puces par section, phrases courtes, pas de jargon. "
        "N'invente aucun chiffre : utilise la direction, l'amplitude (%) et la volatilité données."
    )
    user = (
        f"Province: {province}\n"
        f"Produit: {product}\n"
        f"Tendance 12 mois: {trend['direction']} (~{trend['pct12m']:.1f}%)\n"
        f"Volatilité mensuelle estimée: ~{trend['vol']:.1f}%\n"
        "Consignes: Consommateurs (budget/substitutions/promo), "
        "Agriculteurs/Investisseurs (prix/volume/contrats/stock), "
        "Gouvernements (aides ciblées/concurrence/communication).\n"
    )
    try:
        text = _call_openai(system, user).strip()
        # filet de sécurité : s'il manque les sections attendues, fallback
        if "## Consommateurs" not in text or "## Gouvernements" not in text:
            raise ValueError("Réponse LLM incomplète")
        return text
    except Exception:
        return _fallback(province, product, trend)

# Test manuel rapide
if __name__ == "__main__":
    import numpy as np
    idx = pd.period_range("2025-01", periods=12, freq="M")
    df_demo = pd.DataFrame({"Date": idx.astype(str), "yhat": np.linspace(100, 110, 12)})
    print(generate_advice("Quebec", "eggs", df_demo))