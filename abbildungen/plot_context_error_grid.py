"""
Zeitkontext-Fehler aufgeschlüsselt nach den 54 Zeitkontexten: je Kontext ein Boxplot der Fehler
aller Testgebäude. Anordnung: drei Blöcke nach Tagesart, darin Spalten nach Jahreszeit und Zeilen
nach Tageszeit; gemeinsame y-Achse.

Ein Diagramm je Modell. Datenquelle: results/cached_eval_per_context.csv.

    python -m abbildungen.plot_context_error_grid [--version main] [--variant deployment]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from abbildungen.plot_style import COLORS, GENERATED_COLOR, apply_thesis_style, fig_size, label, save_figure
from src.evaluation.evaluation_metrics import DAYTYPE_NAMES, SEASON_NAMES, TIME_OF_DAY_NAMES

OUT_DIR = Path("abbildungen/outputs")
PER_CONTEXT_CSV = Path("results/cached_eval_per_context.csv")
N_SEASON, N_DAYTYPE, N_TOD = len(SEASON_NAMES), len(DAYTYPE_NAMES), len(TIME_OF_DAY_NAMES)
GROUP_GAP = 0.3
CAP_MARGIN = 1.35   # y-Achse bis zum höchsten Whisker-Ende * CAP_MARGIN


def load(version: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(PER_CONTEXT_CSV)
    sub = df[(df["version_suffix"] == version) & (df["variant"] == variant) & (df["n"] > 0)]
    if sub.empty:
        raise SystemExit(f"Keine Zeilen fuer version_suffix={version!r}, variant={variant!r} in {PER_CONTEXT_CSV}. "
                         f"Vorhanden: {sorted(df['version_suffix'].unique())}")
    return sub


def context_values(sub: pd.DataFrame, season: str, daytype: str, tod: str) -> np.ndarray:
    m = sub[(sub["season"] == season) & (sub["daytype"] == daytype) & (sub["tod"] == tod)]
    return m["error"].to_numpy() * 100


def draw_panel(ax, values: np.ndarray) -> float:
    """Zeichnet einen Boxplot und gibt das obere Whisker-Ende zurück."""
    bp = ax.boxplot([values], positions=[0], widths=0.55, patch_artist=True, showfliers=True,
                    flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": COLORS["gray"],
                                "markeredgecolor": "none", "alpha": 0.6},
                    medianprops={"color": COLORS["black"], "lw": 1.3},
                    whiskerprops={"color": COLORS["black"], "lw": 0.8},
                    capprops={"color": COLORS["black"], "lw": 0.8},
                    boxprops={"lw": 0.8})
    bp["boxes"][0].set_facecolor(GENERATED_COLOR)
    bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][0].set_edgecolor(COLORS["black"])
    ax.scatter([0], [values.mean()], marker="D", s=10, color=COLORS["black"], zorder=5)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([0])
    ax.set_xticklabels([f"n={len(values)}"], fontsize=6.5)
    return bp["caps"][1].get_ydata()[0]


def build_figure(sub: pd.DataFrame, version: str, variant: str):
    fig = plt.figure(figsize=fig_size(1.0, 1.2))
    ratios = []
    for d in range(N_DAYTYPE):
        ratios += [1] * N_SEASON + ([GROUP_GAP] if d < N_DAYTYPE - 1 else [])
    gs = fig.add_gridspec(N_TOD, len(ratios), width_ratios=ratios, hspace=0.18, wspace=0.12,
                          left=0.11, right=0.99, top=0.88, bottom=0.12)
    first = None
    axes = np.empty((N_TOD, N_DAYTYPE, N_SEASON), dtype=object)
    upper_caps = []
    for d in range(N_DAYTYPE):
        for s in range(N_SEASON):
            col = d * (N_SEASON + 1) + s
            for t in range(N_TOD):
                ax = fig.add_subplot(gs[t, col], sharey=first)
                first = first or ax
                axes[t, d, s] = ax
                values = context_values(sub, SEASON_NAMES[s], DAYTYPE_NAMES[d], TIME_OF_DAY_NAMES[t])
                upper_caps.append(draw_panel(ax, values))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="y", labelsize=8, length=2, pad=1.5)
                ax.tick_params(axis="x", length=0, pad=1.5)
                if col > 0:
                    ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.set_ylabel(label(TIME_OF_DAY_NAMES[t]), fontsize=10, labelpad=4)
                if t == 0:
                    ax.set_title(label(SEASON_NAMES[s]), fontsize=8.5, pad=3)
    y_top = max(upper_caps) * CAP_MARGIN
    first.set_ylim(0, y_top)
    first.yaxis.set_major_locator(MultipleLocator(20))
    first.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))

    # Maximum der Werte oberhalb der Achsengrenze als Text
    for d in range(N_DAYTYPE):
        for s in range(N_SEASON):
            for t in range(N_TOD):
                values = context_values(sub, SEASON_NAMES[s], DAYTYPE_NAMES[d], TIME_OF_DAY_NAMES[t])
                clipped = values[values > y_top]
                if len(clipped):
                    axes[t, d, s].text(0.95, 0.95, f"max {clipped.max():.0f}", transform=axes[t, d, s].transAxes,
                                       ha="right", va="top", fontsize=6, color=COLORS["gray"])

    for d in range(N_DAYTYPE):
        pos = axes[0, d, N_SEASON // 2].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.035, label(DAYTYPE_NAMES[d]), ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    fig.supylabel("Zeitkontext-Fehler unter den Testgebäuden [%]", x=0.005, fontsize=11)
    fig.text(0.55, 0.085, f"{version} / {variant} - Boxplot je Kontext (n = Testgebäude, "
             "Raute = Mittelwert)", ha="center", va="top", fontsize=10)
    return fig


def report(sub: pd.DataFrame, version: str, variant: str) -> None:
    rows = []
    for d in range(N_DAYTYPE):
        for s in range(N_SEASON):
            for t in range(N_TOD):
                values = context_values(sub, SEASON_NAMES[s], DAYTYPE_NAMES[d], TIME_OF_DAY_NAMES[t])
                rows.append({"daytype": label(DAYTYPE_NAMES[d]), "season": label(SEASON_NAMES[s]),
                            "tod": label(TIME_OF_DAY_NAMES[t]), "mean": values.mean(), "median": np.median(values)})
    df = pd.DataFrame(rows)
    table = df.pivot_table(index="tod", columns=["daytype", "season"], values="mean")
    print(f"\n[{version} / {variant}] Mittelwert je Zeitkontext [%]:")
    print(table.round(1).to_string())
    best = df.loc[df["mean"].idxmin()]
    worst = df.loc[df["mean"].idxmax()]
    print(f"\nBester Kontext:       {best['daytype']} / {best['season']} / {best['tod']}  "
          f"Mittel {best['mean']:.1f} %  Median {best['median']:.1f} %")
    print(f"Schlechtester Kontext: {worst['daytype']} / {worst['season']} / {worst['tod']}  "
          f"Mittel {worst['mean']:.1f} %  Median {worst['median']:.1f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="main", help="version_suffix des Checkpoints")
    ap.add_argument("--variant", default="deployment", choices=["deployment", "oracle"])
    args = ap.parse_args()

    sub = load(args.version, args.variant)

    apply_thesis_style()
    fig = build_figure(sub, args.version, args.variant)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"context_error_grid_{args.version}_{args.variant}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")
    report(sub, args.version, args.variant)


if __name__ == "__main__":
    main()
