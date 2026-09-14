"""
Ergebnis der Patch-Längen-Variation im Einsatzfall (deployment).

Oben Zeitkontext-Fehler, unten Sprungfaktor. p8 ist das Mittel über fünf Trainingsläufe
(Seeds 5/7/9/12/20, Spanne darunter), alle anderen Längen zeigen den Lauf mit Seed 9.

Die Konsolenausgabe vergleicht jede Länge per exaktem Rangtest mit den p8-Läufen und stellt die
Streuung der p8-Läufe der Spannweite über alle Patch-Längen gegenüber.

Datenquelle: results/cached_eval_summary.csv.

    python -m abbildungen.plot_patch_len_result
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

from abbildungen.plot_style import COLORS, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure

IN_CSV = Path("results/cached_eval_summary.csv")
OUT = Path("abbildungen/outputs/patch_len_result")
VARIANT = "deployment"     # "deployment" = Einsatzfall | "oracle" = echter Vortageswert
REF = "p8"                 # Referenzkonfiguration

POINT_COLOR = COLORS["red"]

# Reihenfolge auf der x-Achse. seed=None mittelt über alle Läufe, sonst Filter auf Seed 9.
PATCHES = {
    "p1": {"label": "1", "seed": 9, "n_seeds": 1},
    "p2": {"label": "2", "seed": 9, "n_seeds": 1},
    "p4": {"label": "4", "seed": 9, "n_seeds": 1},
    "p6": {"label": "6", "seed": 9, "n_seeds": 1},
    "p8": {"label": "8\n(Main)", "seed": None, "n_seeds": 5},
    "p12": {"label": "12", "seed": 9, "n_seeds": 1},
    "p24": {"label": "24", "seed": 9, "n_seeds": 1},
}

# (Spalte, Faktor, Achsenbeschriftung, Etikett des Mittelwerts)
METRICS = [
    ("error_context_mean", 100, "Zeitkontextfehler [%]", "{:.1f} %"),
    ("jump_factor_median", 1, "Sprungfaktor [-]", "{:.2f}"),
]


def load() -> dict[str, pd.DataFrame]:
    """Ein DataFrame je Patch-Länge (nur arch == base)."""
    df = pd.read_csv(IN_CSV)
    df = df[(df["variant"] == VARIANT) & (df["arch"] == "base") & (df["patch"] != "noenc")]
    runs = {}
    for patch, p in PATCHES.items():
        sel = df[df["patch"] == patch]
        if p["seed"] is not None:
            sel = sel[sel["seed"] == p["seed"]]
        assert len(sel) == p["n_seeds"], f"{patch}: {len(sel)} statt {p['n_seeds']} Lauf/Laeufe in {IN_CSV}"
        runs[patch] = sel
    return runs


def draw(ax, runs: dict[str, pd.DataFrame], column: str, factor: float, ylabel: str, fmt: str,
        show_labels: bool) -> None:
    keys = list(PATCHES.keys())
    for i, key in enumerate(keys):
        v = runs[key][column] * factor
        mean = v.mean()
        ax.plot(i, mean, "o", ms=7.5, color=POINT_COLOR, zorder=3)
        ax.annotate(fmt.format(mean).replace(".", ","), (i, mean), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
        if len(v) > 1:
            # Spanne der p8-Läufe
            min_str, max_str = fmt.format(v.min()), fmt.format(v.max())
            if min_str.endswith(" %"):
                min_str = min_str[:-2]
            span = f"({min_str}–{max_str})".replace(".", ",")
            ax.annotate(span, (i, mean), xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color=COLORS["black"])
    ax.set_xticks(range(len(keys)))
    if show_labels:
        ax.set_xticklabels([PATCHES[k]["label"] for k in keys])
        for key, tick_label in zip(keys, ax.get_xticklabels()):
            if key == REF:
                tick_label.set_fontweight("bold")
        ax.set_xlabel("Patch-Länge [h]")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))  # Dezimalkomma
    ax.set_ylabel(ylabel)


def exact_rank_p(a: np.ndarray, b: np.ndarray) -> float:
    """Zweiseitiger exakter Permutationstest auf die Rangsumme von a im gepoolten Datensatz."""
    pooled = np.concatenate([a, b])
    ranks = stats.rankdata(pooled)
    obs = ranks[:len(a)].sum()
    sums = np.array([ranks[list(c)].sum() for c in itertools.combinations(range(len(pooled)), len(a))])
    return float(np.mean(np.abs(sums - sums.mean()) >= abs(obs - sums.mean()) - 1e-9))


def main() -> None:
    runs = load()
    apply_thesis_style()
    fig, axes = plt.subplots(len(METRICS), 1, figsize=fig_size(1.0, 0.8))
    for i, (ax, metric) in enumerate(zip(axes, METRICS)):
        draw(ax, runs, *metric, show_labels=(i == len(METRICS) - 1))

    # Referenzlinie Sprungfaktor 1 (reale Tagesübergänge)
    jump_ax = axes[[m[0] for m in METRICS].index("jump_factor_median")]
    jump_ax.axhline(1.0, color=REAL_COLOR, lw=0.8, ls=(0, (5, 4)), zorder=1)
    jump_ax.text(-0.45, 1.0, "real", ha="left", va="bottom", fontsize=10, color=REAL_COLOR)

    fig.tight_layout(h_pad=3.0)
    despine(fig)
    save_figure(fig, OUT)
    print(f"-> {OUT}.pdf / .png   (VARIANT = {VARIANT})")
    report(runs)


def report(runs: dict[str, pd.DataFrame]) -> None:
    for column, factor, ylabel, _ in METRICS:
        print(f"\n[{VARIANT}] {ylabel}")
        ref = runs[REF][column].to_numpy() * factor
        for patch in PATCHES:
            v = runs[patch][column].to_numpy() * factor
            mark = "  <- Referenz" if patch == REF else ""
            p = "" if patch == REF else f"  Rangtest gegen {REF} p={exact_rank_p(v, ref):.3f}"
            span = f"  Spanne {v.min():6.3f}..{v.max():6.3f}" if len(v) > 1 else ""
            print(f"  {patch:<4} n={len(v)}  Mittel {v.mean():6.3f}{span}{p}{mark}")

    means = pd.Series({patch: (runs[patch]["error_context_mean"] * 100).mean() for patch in PATCHES})
    ref_vals = runs[REF]["error_context_mean"].to_numpy() * 100
    print(f"\n  Streuung der {len(ref_vals)} {REF}-Laeufe:        {ref_vals.max() - ref_vals.min():.2f} pp "
          f"({ref_vals.min():.2f}..{ref_vals.max():.2f})")
    print(f"  Spannweite aller Patch-Mittel:      {means.max() - means.min():.2f} pp "
          f"({means.min():.2f}..{means.max():.2f})")
    print("  -> die Streuung einer Konfiguration ueber ihre Laeufe uebersteigt die Spannweite "
          "ueber alle Patch-Laengen.")


if __name__ == "__main__":
    main()
