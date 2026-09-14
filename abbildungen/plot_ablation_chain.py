"""
Kovariaten-Ablation im Einsatzfall (deployment). Bei Modellen mit Cluster-Kovariate wird das je
Sensor aus cluster_proportions.csv gezogene Cluster verwendet (--cluster-source sampled), da das
Cluster eines neuen Gebäudes nicht bekannt ist.

Oben Zeitkontext-Fehler (Mittel über die Testgebäude), unten Sprungfaktor (Median über die
Testgebäude). Punkt = Mittel über die Trainingsläufe; bei main und baseline (5 Seeds) steht die
Spanne der Läufe darunter. Rot = mit Encoder, Magenta = ohne Encoder.

Datenquelle: results/cached_eval_summary.csv und results/cached_eval_clustersampled_summary.csv.

    python -m abbildungen.plot_ablation_chain
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from abbildungen.plot_style import COLORS, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure

IN_CSV = Path("results/cached_eval_summary.csv")
IN_CSV_SAMPLED = Path("results/cached_eval_clustersampled_summary.csv")
OUT = Path("abbildungen/outputs/ablation_chain")
VARIANT = "deployment"

NOENC_COLOR = COLORS["magenta"]
ENC_COLOR = COLORS["red"]

# Reihenfolge auf der x-Achse. Läufe werden über (arch, patch) ausgewählt; csv = Datenquelle,
# seed = None mittelt über alle Seeds, n_seeds = erwartete Anzahl Läufe.
MODELS = {
    "baseline": {"label": "Baseline", "color": NOENC_COLOR, "arch": "baseline", "patch": "noenc",
                "csv": IN_CSV_SAMPLED, "seed": None, "n_seeds": 5, "bold": True},
    "baseline_h23": {"label": "+h23", "color": NOENC_COLOR, "arch": "baseline_h23", "patch": "noenc",
                "csv": IN_CSV_SAMPLED, "seed": 9, "n_seeds": 1, "bold": False},
    "baseline_category": {"label": "+Kategorie", "color": NOENC_COLOR, "arch": "baseline_category", "patch": "noenc",
                "csv": IN_CSV, "seed": 9, "n_seeds": 1, "bold": False},
    "main_noenc": {"label": "+h23+Kategorie", "color": NOENC_COLOR, "arch": "base", "patch": "noenc",
                "csv": IN_CSV, "seed": 9, "n_seeds": 1, "bold": False},
    "main": {"label": "Main", "color": ENC_COLOR, "arch": "base", "patch": "p8",
                "csv": IN_CSV, "seed": None, "n_seeds": 5, "bold": True},
    "main_noh23": {"label": "-h23", "color": ENC_COLOR, "arch": "noh23", "patch": "p8",
                "csv": IN_CSV, "seed": 9, "n_seeds": 1, "bold": False},
    "main_nocat": {"label": "-Kategorie", "color": ENC_COLOR, "arch": "nocat", "patch": "p8",
                "csv": IN_CSV, "seed": 9, "n_seeds": 1, "bold": False},
    "main_cluster": {"label": "+Cluster", "color": ENC_COLOR, "arch": "cluster", "patch": "p8",
                "csv": IN_CSV_SAMPLED, "seed": 9, "n_seeds": 1, "bold": False},
}
GROUP_GAP_AFTER = "main_noenc"   # Trennlinie nach diesem Modell

# (Spalte, Faktor, Achsenbeschriftung, Etikett des Mittelwerts)
METRICS = [
    ("error_context_mean", 100, "Zeitkontextfehler [%]", "{:.1f} %"),
    ("jump_factor_median", 1, "Sprungfaktor [-]", "{:.2f}"),
]


def load() -> dict[str, pd.DataFrame]:
    runs = {}
    for key, m in MODELS.items():
        df = pd.read_csv(m["csv"])
        sel = df[(df["variant"] == VARIANT) & (df["arch"] == m["arch"]) & (df["patch"] == m["patch"])]
        if m["seed"] is not None:
            sel = sel[sel["seed"] == m["seed"]]
        assert len(sel) == m["n_seeds"], f"{key}: {len(sel)} statt {m['n_seeds']} Lauf/Laeufe in {m['csv']}"
        runs[key] = sel
    return runs


def draw(ax, runs: dict[str, pd.DataFrame], column: str, factor: float, ylabel: str, fmt: str,
        show_labels: bool) -> None:
    keys = list(MODELS.keys())
    for i, key in enumerate(keys):
        m = MODELS[key]
        v = runs[key][column] * factor
        mean = v.mean()
        ax.plot(i, mean, "o", ms=7.5, color=m["color"], zorder=3)
        ax.annotate(fmt.format(mean).replace(".", ","), (i, mean), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
        if len(v) > 1:
            # Spanne über die Seeds
            min_str, max_str = fmt.format(v.min()), fmt.format(v.max())
            if min_str.endswith(" %"):
                min_str = min_str[:-2]
            span = f"({min_str}–{max_str})".replace(".", ",")
            ax.annotate(span, (i, mean), xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color=COLORS["black"])
    gap_idx = keys.index(GROUP_GAP_AFTER)
    ax.axvline(gap_idx + 0.5, color=COLORS["grid"], lw=0.9, zorder=1)
    ax.set_xticks(range(len(keys)))
    if show_labels:
        ax.set_xticklabels([MODELS[k]["label"] for k in keys], fontsize=8, rotation=30, ha="right", rotation_mode="anchor")
        for key, tick_label in zip(keys, ax.get_xticklabels()):
            if MODELS[key]["bold"]:
                tick_label.set_fontweight("bold")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))  # Dezimalkomma
    ax.set_ylabel(ylabel)


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
        for key in MODELS:
            v = runs[key][column] * factor
            print(f"  {key:<20} n={len(v)}  Mittel {v.mean():6.3f}  Spanne {v.min():6.3f}..{v.max():6.3f}"
                  f"  ({v.max() - v.min():.3f})")


if __name__ == "__main__":
    main()
