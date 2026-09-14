"""
Hauptvergleich: baseline (Modell der vorangegangenen Arbeit) gegen main (mit Encoder), je fünf
Trainingsläufe, im Einsatzfall (deployment).

Für baseline wird das je Sensor aus cluster_proportions.csv gezogene Cluster verwendet
(--cluster-source sampled), da das Cluster eines neuen Gebäudes nicht bekannt ist.

Links Zeitkontext-Fehler (Mittel über die Testgebäude), rechts Sprungfaktor (Median über die
Testgebäude). Punkt = Mittel über die fünf Trainingsläufe, darunter die Spanne der Läufe.

Datenquelle: results/cached_eval_clustersampled_summary.csv (baseline) und
results/cached_eval_summary.csv (main).

    python -m abbildungen.plot_main_vs_baseline
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from abbildungen.plot_style import COLORS, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure

IN_CSV = Path("results/cached_eval_summary.csv")
IN_CSV_SAMPLED = Path("results/cached_eval_clustersampled_summary.csv")
OUT = Path("abbildungen/outputs/main_vs_baseline")
VARIANT = "deployment"
# Reihenfolge auf der x-Achse; Läufe werden über (arch, patch) ausgewählt, csv = Datenquelle.
MODELS = {
    "baseline": {"label": "Baseline", "color": COLORS["magenta"], "arch": "baseline", "patch": "noenc", "csv": IN_CSV_SAMPLED},
    "main": {"label": "Main", "color": COLORS["red"], "arch": "base", "patch": "p8", "csv": IN_CSV},
}
# (Spalte, Faktor, Achsenbeschriftung, Etikett des Mittelwerts)
METRICS = [
    ("error_context_mean", 100, "Zeitkontextfehler [%]", "{:.1f} %"),
    ("jump_factor_median", 1, "Sprungfaktor [-]", "{:.2f}"),
]


def load() -> dict[str, pd.DataFrame]:
    runs = {}
    for key, m in MODELS.items():
        df = pd.read_csv(m["csv"])
        df = df[df["variant"] == VARIANT]
        runs[key] = df[(df["arch"] == m["arch"]) & (df["patch"] == m["patch"])]
        assert len(runs[key]) > 1, f"{key}: nur {len(runs[key])} Lauf/Laeufe in {m['csv']}"
    return runs


def draw(ax, runs: dict[str, pd.DataFrame], column: str, factor: float, ylabel: str, fmt: str) -> None:
    for i, (key, m) in enumerate(MODELS.items()):
        v = runs[key][column] * factor
        mean = v.mean()
        ax.plot(i, mean, "o", ms=7.5, color=m["color"], zorder=3)
        ax.annotate(fmt.format(mean).replace(".", ","), (i, mean), xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=10)
        # Spanne über die Seeds
        min_str, max_str = fmt.format(v.min()), fmt.format(v.max())
        if min_str.endswith(" %"):
            min_str = min_str[:-2]
        span = f"({min_str}–{max_str})".replace(".", ",")
        ax.annotate(span, (i, mean), xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=COLORS["black"])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m["label"] for m in MODELS.values()])
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))  # Dezimalkomma
    ax.set_ylabel(ylabel)


def main() -> None:
    runs = load()
    apply_thesis_style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=fig_size(1.0, 0.5))
    for ax, metric in zip(axes, METRICS):
        draw(ax, runs, *metric)

    # Referenzlinie Sprungfaktor 1 (reale Tagesübergänge)
    jump_ax = axes[[m[0] for m in METRICS].index("jump_factor_median")]
    jump_ax.axhline(1.0, color=REAL_COLOR, lw=0.8, ls=(0, (5, 4)), zorder=1)
    jump_ax.text(-0.45, 1.0, "real", ha="left", va="bottom", fontsize=10, color=REAL_COLOR)

    fig.tight_layout(w_pad=2.5)
    despine(fig)
    save_figure(fig, OUT)
    print(f"-> {OUT}.pdf / .png   (VARIANT = {VARIANT})")
    report(runs)


def report(runs: dict[str, pd.DataFrame]) -> None:
    for column, factor, ylabel, _ in METRICS:
        print(f"\n[{VARIANT}] {ylabel}")
        for key, m in MODELS.items():
            v = runs[key][column] * factor
            print(f"  {m['label']:<12} n={len(v)}  Mittel {v.mean():6.3f}  Spanne {v.min():6.3f}..{v.max():6.3f}"
                  f"  ({v.max() - v.min():.3f})")


if __name__ == "__main__":
    main()
