"""
Dauerlinien in ausgewählten Zeitkontexten (EXAMPLE_CONTEXTS) für dasselbe Beispielgebäude wie
plot_duration_curve_example.py: je Feld Messung, erzeugtes Profil und die Fläche dazwischen.

Die x-Achse zeigt je Feld die Stunden des Kontexts, die y-Achse ist gemeinsam
(Last / Jahresmittellast). Die Konsolenausgabe teilt den Fehler je Kontext in zu hoch und zu
niedrig erzeugte Last auf.

Datenquelle: data/samples/<version>/.

    python -m abbildungen.plot_duration_curve_timecontext_example [--version main] [--variant deployment]
                                                                 [--quantile 0.5 | --sensor <id> --year <JJJJ>]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from abbildungen.plot_duration_curve_all_timecontexts import context_groups, context_index
from abbildungen.plot_duration_curve_example import AREA_ALPHA, QUANTILE, num, pick, sensor_years
from abbildungen.plot_style import (COLORS, GENERATED_COLOR, REAL_COLOR, apply_thesis_style, despine, fig_size, label,
                                    save_figure)
from src.evaluation.compare_cached_samples import load_cache
from src.evaluation.evaluation_metrics import (DAYTYPE_NAMES, SEASON_NAMES, TIME_OF_DAY_BOUNDS, TIME_OF_DAY_NAMES,
                                               duration_curve_error)

OUT_DIR = Path("abbildungen/outputs")
# (Jahreszeit, Tagesart, Tageszeit), von links nach rechts
EXAMPLE_CONTEXTS = [
    ("Winter", "Werktag", "Mittag"),
    ("Uebergang", "Samstag", "Nacht"),
    ("Sommer", "Sonntag", "Abend"),
]


def context_id(season: str, daytype: str, tod: str) -> int:
    return context_index(SEASON_NAMES.index(season), DAYTYPE_NAMES.index(daytype), TIME_OF_DAY_NAMES.index(tod))


def title(season: str, daytype: str, tod: str) -> str:
    start, end = TIME_OF_DAY_BOUNDS[TIME_OF_DAY_NAMES.index(tod)]
    return f"{label(daytype)}, {label(season)}\n{label(tod)} ({start}–{end} Uhr)"


def draw_panel(ax, real: np.ndarray, gen: np.ndarray, heading: str) -> None:
    r = np.sort(real)[::-1]
    g = np.sort(gen)[::-1]
    hours = np.arange(1, len(r) + 1)
    ax.fill_between(hours, r, g, color=GENERATED_COLOR, alpha=AREA_ALPHA, lw=0, zorder=1)
    ax.plot(hours, r, color=REAL_COLOR, lw=1.4, zorder=3)
    ax.plot(hours, g, color=GENERATED_COLOR, lw=1.4, zorder=2)
    ax.axhline(1.0, color=COLORS["gray"], lw=0.8, ls=(0, (2, 3)), zorder=0)
    ax.set_xlim(0, len(r))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: num(x, "{:,.0f}")))
    ax.set_xlabel("Stunden [h]")
    ax.set_title(heading, fontsize=10, pad=6)


def build_figure(real_ctx, gen_ctx):
    ids = [context_id(*c) for c in EXAMPLE_CONTEXTS]
    fig, axes = plt.subplots(1, len(ids), figsize=fig_size(1.0, 0.52), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, c, spec in zip(axes, ids, EXAMPLE_CONTEXTS):
        draw_panel(ax, real_ctx[c], gen_ctx[c], title(*spec))
    top = max(max(real_ctx[c].max(), gen_ctx[c].max()) for c in ids) * 1.05
    axes[0].set_ylim(0, top)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))
    axes[0].set_ylabel("Last / Jahresmittellast [-]")
    despine(fig)

    handles = [Line2D([], [], color=REAL_COLOR, lw=1.6, label="Messung"),
               Line2D([], [], color=GENERATED_COLOR, lw=1.6, label="Erzeugtes Profil"),
               Patch(facecolor=GENERATED_COLOR, alpha=AREA_ALPHA, lw=0, label="Fläche zwischen den Dauerlinien"),
               Line2D([], [], color=COLORS["gray"], lw=0.8, ls=(0, (2, 3)), label="Jahresmittellast")]
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    return fig


def report(chosen, real_ctx, gen_ctx, rows, args) -> None:
    counts = np.array([len(c) for c in real_ctx])
    errors = duration_curve_error(real_ctx, gen_ctx)
    weighted = np.sum(errors[counts > 0] * counts[counts > 0]) / counts[counts > 0].sum()
    glob = duration_curve_error([rows["y_real"].ravel()], [rows["y_sample"].ravel()])[0]
    print(f"\n[{args.version} / {args.variant}] Sensor {chosen['sensor_id']}, Jahr {chosen['year']}: "
          f"Zeitkontext-Fehler {weighted * 100:.2f} %, Jahresdauerlinie {glob * 100:.2f} %")
    print("Gezeigte Kontexte [% der Jahresmittellast; zu hoch + zu niedrig = Fehler]:")
    for spec in EXAMPLE_CONTEXTS:
        c = context_id(*spec)
        r, g = np.sort(real_ctx[c]), np.sort(gen_ctx[c])
        over, under = np.clip(g - r, 0, None).mean(), np.clip(r - g, 0, None).mean()
        print(f"  {' / '.join(label(s) for s in spec):<34} {counts[c]:4d} h  Fehler {errors[c] * 100:5.1f}  "
              f"= zu hoch {over * 100:5.1f} + zu niedrig {under * 100:5.1f}   "
              f"Mittel Messung {r.mean():.2f} erzeugt {g.mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="main", help="version_suffix des Checkpoints")
    ap.add_argument("--variant", default="deployment", choices=["deployment", "oracle"])
    ap.add_argument("--quantile", type=float, default=QUANTILE,
                    help="Quantil des globalen Fehlers ueber die vollstaendigen Sensor-Jahre (Default Median, "
                         "wie die Jahresabbildung - damit zeigen alle drei Abbildungen dasselbe Gebaeude)")
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
    empty = [spec for spec in EXAMPLE_CONTEXTS if len(real_ctx[context_id(*spec)]) == 0]
    if empty:
        raise SystemExit(f"Kontexte ohne Stunden in diesem Sensor-Jahr: {empty} - EXAMPLE_CONTEXTS anpassen.")

    apply_thesis_style()
    fig = build_figure(real_ctx, gen_ctx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"duration_curve_timecontext_example_{args.version}_{args.variant}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")
    report(chosen, real_ctx, gen_ctx, rows, args)


if __name__ == "__main__":
    main()
