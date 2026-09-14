"""
Bester und schlechtester Zeitkontext von main und baseline über alle Testgebäude.

Die Kontexte werden über evaluation_metrics.context_error_over_sensors bestimmt (Fehler je
Sensor, dann über die Testgebäude gemittelt; Seed 9). Dargestellt ist je Kontext die Verteilung
der Fehler der einzelnen Testgebäude als Boxplot, mit Median (Linie) und Mittelwert (Raute).

Variante: deployment. Datenquelle: results/cached_eval_per_context.csv.

    python -m abbildungen.plot_context_error_extremes
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from abbildungen.plot_style import COLORS, apply_thesis_style, despine, fig_size, label, save_figure
from src.evaluation.evaluation_metrics import context_error_over_sensors

OUT_DIR = Path("abbildungen/outputs")
PER_CONTEXT_CSV = Path("results/cached_eval_per_context.csv")
VARIANT = "deployment"
MODELS = ["main", "baseline"]           # version_suffix (Seed 9)
MODEL_LABELS = {"main": "main", "baseline": "baseline"}
MODEL_COLORS = {"main": COLORS["red"], "baseline": COLORS["gray"]}
KINDS = [("beste", "idxmin"), ("schlechteste", "idxmax")]
GROUP_GAP = 1.0


def rank_contexts(per_context: pd.DataFrame, version_suffix: str) -> pd.DataFrame:
    """Fehler der 54 Zeitkontexte eines Checkpoints, gemittelt über die Testgebäude."""
    sub = per_context[(per_context["version_suffix"] == version_suffix) & (per_context["variant"] == VARIANT)]
    return context_error_over_sensors(sub)


def context_values(per_context: pd.DataFrame, version_suffix: str, season: str, daytype: str,
                   tod: str) -> np.ndarray:
    """Fehler je Testgebäude in diesem Kontext, in %."""
    sub = per_context[(per_context["version_suffix"] == version_suffix) & (per_context["variant"] == VARIANT)
                      & (per_context["season"] == season) & (per_context["daytype"] == daytype)
                      & (per_context["tod"] == tod) & (per_context["n"] > 0)]
    return sub["error"].to_numpy() * 100


def context_label(season: str, daytype: str, tod: str) -> str:
    return f"{label(season)}, {label(daytype)}, {label(tod)}"


def main() -> None:
    per_context = pd.read_csv(PER_CONTEXT_CSV)

    groups = []   # (kind, [(version_suffix, season, daytype, tod, values, error_mean_pct), ...])
    for kind, picker in KINDS:
        entries = []
        for version_suffix in MODELS:
            agg = rank_contexts(per_context, version_suffix)
            idx = agg["error_mean"].idxmin() if picker == "idxmin" else agg["error_mean"].idxmax()
            row = agg.loc[idx]
            season, daytype, tod = str(row["season"]), str(row["daytype"]), str(row["tod"])
            values = context_values(per_context, version_suffix, season, daytype, tod)
            entries.append((version_suffix, season, daytype, tod, values, row["error_mean"] * 100))
        groups.append((kind, entries))

    positions, box_data, colors, xticklabels, group_centers, group_headers = [], [], [], [], [], []
    pos = 0.0
    for kind, entries in groups:
        start = pos
        for version_suffix, season, daytype, tod, values, mean_pct in entries:
            positions.append(pos)
            box_data.append(values)
            colors.append(MODEL_COLORS[version_suffix])
            xticklabels.append(MODEL_LABELS[version_suffix])
            pos += 1.0
        group_centers.append((start + pos - 1.0) / 2)
        same_ctx = len({(e[1], e[2], e[3]) for e in entries}) == 1
        if same_ctx:
            group_headers.append(f"{kind}r Kontext\n{context_label(*entries[0][1:4])}")
        else:
            group_headers.append(f"{kind}r Kontext\n(je Modell verschieden, siehe Report)")
        pos += GROUP_GAP

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=fig_size(0.85, 0.85))
    bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True, showfliers=True,
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": COLORS["gray"],
                                "markeredgecolor": "none", "alpha": 0.6},
                    medianprops={"color": COLORS["black"], "lw": 1.6},
                    whiskerprops={"color": COLORS["black"], "lw": 1.0},
                    capprops={"color": COLORS["black"], "lw": 1.0},
                    boxprops={"lw": 1.0})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(COLORS["black"])
    means = [mean_pct for _, entries in groups for *_, mean_pct in entries]
    ax.scatter(positions, means, marker="D", s=26, color=COLORS["black"], zorder=5)

    # y-Achse oberhalb der Whisker begrenzen; Ausreißer außerhalb werden als Text angegeben.
    upper_caps = [cap.get_ydata()[0] for cap in bp["caps"][1::2]]
    y_top = max(upper_caps) * 1.35
    for pos, values in zip(positions, box_data):
        clipped = values[values > y_top]
        if len(clipped):
            ax.text(pos, y_top * 0.98, f"+{len(clipped)}: bis {clipped.max():.0f} %", ha="center", va="top",
                    fontsize=7.5, color=COLORS["gray"], rotation=90 if len(clipped) == 1 else 0)

    ax.set_xticks(positions, xticklabels, fontsize=10)
    for center, header in zip(group_centers, group_headers):
        ax.text(center, 1.0, header, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", linespacing=1.4)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))
    ax.set_ylabel("Zeitkontext-Fehler unter den Testgebäuden [%]")
    despine(fig)

    handles = [Line2D([], [], marker="s", ls="", markersize=10, markerfacecolor=MODEL_COLORS["main"],
                      markeredgecolor=COLORS["black"], alpha=0.55, label="main"),
               Line2D([], [], marker="s", ls="", markersize=10, markerfacecolor=MODEL_COLORS["baseline"],
                      markeredgecolor=COLORS["black"], alpha=0.55, label="baseline"),
               Line2D([], [], color=COLORS["black"], lw=1.6, label="Median"),
               Line2D([], [], marker="D", ls="", markersize=6, color=COLORS["black"],
                      label="Mittelwert (= Rangkriterium)")]
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"context_error_extremes_{VARIANT}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")

    print(f"\n[{VARIANT}] bester/schlechtester Zeitkontext je Modell, Fehlerverteilung ueber die Testgebaeude:")
    for kind, entries in groups:
        for version_suffix, season, daytype, tod, values, mean_pct in entries:
            print(f"  {version_suffix:8s} {kind:13s} {context_label(season, daytype, tod):<32} "
                  f"n={len(values):3d} Gebaeude  Mittel {mean_pct:5.1f} %  Median {np.median(values):5.1f} %  "
                  f"IQR [{np.percentile(values, 25):.1f}, {np.percentile(values, 75):.1f}]  "
                  f"Max {values.max():.1f} %")


if __name__ == "__main__":
    main()
