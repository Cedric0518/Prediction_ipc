import numpy as np
import pandas as pd

def _compute_trend(hist: pd.DataFrame, fut: pd.DataFrame, target="Indice_IPC_produit"):
    # trend = moy(forecast M1-3) - moy(observé derniers 3 mois)
    h = hist.copy()
    h = h.sort_values("Date").tail(3)
    base = h[target].mean()
    f = fut.copy().head(3)  # premiers 3 mois
    delta = f["yhat"].mean() - base
    pct = 100.0 * delta / base if base else 0.0
    return pct

def recommend_for_series(hist: pd.DataFrame, fut: pd.DataFrame, product: str, province: str) -> str:
    pct = _compute_trend(hist, fut)
    if pct >= 2.0:
        direction = "hausse"
    elif pct <= -2.0:
        direction = "baisse"
    else:
        direction = "stable"

    base = f"Série: {product} – {province}\nTendance 3 mois: {pct:+.1f}% → {direction.upper()}\n\n"

    # Consommateurs
    if direction == "hausse":
        c = "- **Consommateurs**: Anticipez le budget, privilégiez formats familiaux et promotions; envisagez des substituts proches.\n"
    elif direction == "baisse":
        c = "- **Consommateurs**: Opportunité d’achat; surveillez les promotions et faites des stocks raisonnables (produits non périssables).\n"
    else:
        c = "- **Consommateurs**: Prix relativement stables; comparez par enseigne, suivez la tendance mensuelle.\n"

    # Agriculteurs / Investisseurs
    if direction == "hausse":
        a = "- **Agriculteurs/Investisseurs**: Ajustez progressivement prix/volumes; sécurisez intrants; envisagez couverture simple (contrats à terme/coups de frein sur coûts).\n"
    elif direction == "baisse":
        a = "- **Agriculteurs/Investisseurs**: Optimisez stocks; différenciez produits à plus forte valeur; prudence sur expansions; protégez marges.\n"
    else:
        a = "- **Agriculteurs/Investisseurs**: Maintenez le cap; surveillez coûts des intrants et délais logistiques.\n"

    # Gouvernements
    if direction == "hausse":
        g = "- **Gouvernements**: Renforcez la veille inflation; envisagez aides ciblées temporaires; surveillez marges et chaînes d’approvisionnement.\n"
    elif direction == "baisse":
        g = "- **Gouvernements**: Consolidez les mesures de concurrence; ajustez les aides; suivez l’effet sur paniers des ménages vulnérables.\n"
    else:
        g = "- **Gouvernements**: Veille ordinaire; transparence sur coûts/ marges; préparer plans de contingence.\n"

    note = "\n*Remarque: messages informatifs, pas de conseil financier personnalisé.*\n"
    return base + c + a + g + note
