"""
Dauerlinien je Zeitkontext für ein Beispielgebäude (dasselbe Sensor-Jahr wie
plot_duration_curve_example.py): 54 Felder, je Feld Messung, erzeugtes Profil und die Fläche
dazwischen.

Anordnung: drei Blöcke nach Tagesart, darin Spalten nach Jahreszeit und Zeilen nach Tageszeit.
Gemeinsame y-Achse (Last / Jahresmittellast); die x-Achse zeigt je Feld die Stundenzahl des
Kontexts.

Datenquelle: data/samples/<version>/.

    python -m abbildungen.plot_duration_curve_all_timecontexts [--version main] [--variant deployment]
                                                              [--quantile 0.5 | --sensor <id> --year <JJJJ>]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from abbildungen.plot_duration_curve_example import AREA_ALPHA, QUANTILE, category_of, num, pick, sensor_years
from abbildungen.plot_style import COLORS, GENERATED_COLOR, REAL_COLOR, apply_thesis_style, fig_size, label, save_figure
from src.evaluation.compare_cached_samples import holiday_flags, load_cache
from src.evaluation.evaluation_metrics import (DAYTYPE_NAMES, SEASON_NAMES, TIME_OF_DAY_NAMES, assign_context,
                                               duration_curve_error, duration_curve_error_per_sensor)

OUT_DIR = Path("abbildungen/outputs")
N_SEASON, N_DAYTYPE, N_TOD = len(SEASON_NAMES), len(DAYTYPE_NAMES), len(TIME_OF_DAY_NAMES)
LW = 1.0
GROUP_GAP = 0.3                         # Abstand zwischen den Tagesart-Blöcken, relativ zur Feldbreite


def context_index(season: int, daytype: int, tod: int) -> int:
    """Reihenfolge von evaluation_metrics.assign_context."""
    return (season * N_DAYTYPE + daytype) * N_TOD + tod


def context_groups(cache: dict, sensor_id: str, year: int) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Reale und erzeugte Stundenwerte je Zeitkontext für ein Sensor-Jahr sowie die zugehörigen Zeilen."""
    m = (cache["sensor_id"] == sensor_id) & (pd.DatetimeIndex(cache["day"]).year == year)
    rows = {k: cache[k][m] for k in ("sensor_id", "day", "y_real", "y_sample")}
    rows["is_holiday"] = holiday_flags(rows["sensor_id"], rows["day"])
    real_ctx, gen_ctx = assign_context(rows["y_real"], rows["y_sample"], rows["day"], rows["is_holiday"])
    return real_ctx, gen_ctx, rows


def draw_panel(ax, real: np.ndarray, gen: np.ndarray) -> None:
    ax.axhline(1.0, color=COLORS["gray"], lw=0.7, ls=(0, (2, 3)), zorder=0)
    if len(real) == 0:
        ax.text(0.5, 0.5, "–", transform=ax.transAxes, ha="center", va="center", color=COLORS["gray"])
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        return
    r = np.sort(real)[::-1]
    g = np.sort(gen)[::-1]
    hours = np.arange(1, len(r) + 1)
    ax.fill_between(hours, r, g, color=GENERATED_COLOR, alpha=AREA_ALPHA, lw=0, zorder=1)
    ax.plot(hours, r, color=REAL_COLOR, lw=LW, zorder=3)
    ax.plot(hours, g, color=GENERATED_COLOR, lw=LW, zorder=2)
    # Stundenzahl des Kontexts als Tick am rechten Rand
    ax.set_xlim(0, len(r))
    ax.set_xticks([len(r)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, "{:,.0f}")))


def build_figure(real_ctx: list[np.ndarray], gen_ctx: list[np.ndarray]):
    fig = plt.figure(figsize=fig_size(1.0, 1.2))
    ratios = []
    for d in range(N_DAYTYPE):
        ratios += [1] * N_SEASON + ([GROUP_GAP] if d < N_DAYTYPE - 1 else [])
    gs = fig.add_gridspec(N_TOD, len(ratios), width_ratios=ratios, hspace=0.18, wspace=0.12,
                          left=0.11, right=0.99, top=0.88, bottom=0.12)
    top = max(max(c.max() for c in real_ctx if len(c)), max(c.max() for c in gen_ctx if len(c))) * 1.05
    first = None
    axes = np.empty((N_TOD, N_DAYTYPE, N_SEASON), dtype=object)
    for d in range(N_DAYTYPE):
        for s in range(N_SEASON):
            col = d * (N_SEASON + 1) + s
            for t in range(N_TOD):
                ax = fig.add_subplot(gs[t, col], sharey=first)
                first = first or ax
                axes[t, d, s] = ax
                c = context_index(s, d, t)
                draw_panel(ax, real_ctx[c], gen_ctx[c])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="y", labelsize=8, length=2, pad=1.5)
                ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1.5)
                if col > 0:
                    ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.set_ylabel(label(TIME_OF_DAY_NAMES[t]), fontsize=10, labelpad=4)
                if t == 0:
                    ax.set_title(label(SEASON_NAMES[s]), fontsize=8.5, pad=3)
    first.set_ylim(0, top)
    first.yaxis.set_major_locator(MultipleLocator(2))
    first.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))

    # Tagesart über dem jeweiligen Block
    for d in range(N_DAYTYPE):
        pos = axes[0, d, N_SEASON // 2].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.035, label(DAYTYPE_NAMES[d]), ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    fig.supylabel("Last / Jahresmittellast [-]", x=0.015, fontsize=11)
    fig.text(0.55, 0.085, "Stunden des Zeitkontexts, absteigend sortiert [h] (je Feld eigene Stundenzahl)",
             ha="center", va="top", fontsize=10)
    handles = [Line2D([], [], color=REAL_COLOR, lw=1.6, label="Messung"),
               Line2D([], [], color=GENERATED_COLOR, lw=1.6, label="Erzeugtes Profil"),
               Patch(facecolor=GENERATED_COLOR, alpha=AREA_ALPHA, lw=0, label="Fläche zwischen den Dauerlinien"),
               Line2D([], [], color=COLORS["gray"], lw=0.8, ls=(0, (2, 3)), label="Jahresmittellast")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.55, 0.0))
    return fig


def report(chosen: pd.Series, real_ctx, gen_ctx, rows: dict, category: str, args) -> None:
    errors = duration_curve_error(real_ctx, gen_ctx)
    counts = np.array([len(c) for c in real_ctx])
    filled = counts > 0
    weighted = np.sum(errors[filled] * counts[filled]) / counts[filled].sum()
    glob = duration_curve_error([rows["y_real"].ravel()], [rows["y_sample"].ravel()])[0]
    # Vergleich mit dem Wert aus evaluation_metrics
    ref, _ = duration_curve_error_per_sensor(rows["y_real"], rows["y_sample"], rows["sensor_id"], rows["day"],
                                             rows["is_holiday"])
    print(f"\n[{args.version} / {args.variant}] Sensor {chosen['sensor_id']}, Jahr {chosen['year']}, "
          f"Kategorie {category!r}, {counts.sum()} h in {int(filled.sum())} von {len(counts)} Kontexten")
    print(f"  Zeitkontext-Fehler {weighted * 100:.2f} % (Metrik: {ref['error_context'].iloc[0] * 100:.2f} %)  "
          f"Jahresdauerlinie {glob * 100:.2f} %  -> Struktur {(weighted - glob) * 100:.2f} %")
    table = pd.DataFrame(
        {(label(DAYTYPE_NAMES[d]), label(SEASON_NAMES[s])):
         [errors[context_index(s, d, t)] * 100 for t in range(N_TOD)]
         for d in range(N_DAYTYPE) for s in range(N_SEASON)},
        index=[label(n) for n in TIME_OF_DAY_NAMES])
    hours = table.copy()
    for d in range(N_DAYTYPE):
        for s in range(N_SEASON):
            hours[(label(DAYTYPE_NAMES[d]), label(SEASON_NAMES[s]))] = \
                [counts[context_index(s, d, t)] for t in range(N_TOD)]
    print("\nFehler je Zeitkontext [% der Jahresmittellast]:")
    print(table.round(1).to_string())
    print("\nStunden je Zeitkontext:")
    print(hours.astype(int).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="main", help="version_suffix des Checkpoints")
    ap.add_argument("--variant", default="deployment", choices=["deployment", "oracle"])
    ap.add_argument("--quantile", type=float, default=QUANTILE,
                    help="Quantil des globalen Fehlers ueber die vollstaendigen Sensor-Jahre (Default Median, "
                         "wie die Jahresabbildung - damit zeigen beide dasselbe Gebaeude)")
    ap.add_argument("--sensor", help="bestimmter Sensor statt Quantil-Auswahl (mit --year)")
    ap.add_argument("--year", type=int)
    args = ap.parse_args()
    if (args.sensor is None) != (args.year is None):
        ap.error("--sensor und --year nur gemeinsam")

    cache = load_cache(args.version, args.variant)
    if cache is None:
        raise SystemExit(f"Kein {args.variant}-Cache fuer {args.version} unter data/samples/.")
    years = sensor_years(cache)
    if args.sensor is None:
        chosen = pick(years, args.quantile)
    else:
        sel = years[(years["sensor_id"] == args.sensor) & (years["year"] == args.year)]
        if sel.empty:
            raise SystemExit(f"Sensor {args.sensor} / {args.year} nicht im Test-Cache.")
        chosen = sel.iloc[0]
    real_ctx, gen_ctx, rows = context_groups(cache, chosen["sensor_id"], int(chosen["year"]))

    apply_thesis_style()
    fig = build_figure(real_ctx, gen_ctx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"duration_curve_all_timecontexts_{args.version}_{args.variant}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")
    report(chosen, real_ctx, gen_ctx, rows, category_of(chosen["sensor_id"]), args)


if __name__ == "__main__":
    main()
