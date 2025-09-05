import matplotlib.pyplot as plt
import pandas as pd

def save_line_plot(hist: pd.DataFrame, forecast: pd.DataFrame, title: str, path: str):
    plt.figure()
    hh = hist.copy()
    hh["Date"] = pd.PeriodIndex(hh["Date"], freq="M").to_timestamp()
    ff = forecast.copy()
    ff["Date"] = pd.PeriodIndex(ff["Date"], freq="M").to_timestamp()

    plt.plot(hh["Date"], hh["Indice_IPC_produit"], label="Historique", marker="o")
    plt.plot(ff["Date"], ff["yhat"], label="Prévision", marker="x")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Indice IPC (proxy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=144)
    plt.close()
