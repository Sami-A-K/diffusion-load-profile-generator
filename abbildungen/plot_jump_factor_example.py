"""
Sprungfaktor an einem Beispielgebäude (evaluation_metrics.day_boundary_error_per_sensor).

Lastsprung = Stunde 0 eines Tages minus Stunde 23 des Vortags, für alle Übergänge zwischen
aufeinanderfolgenden Tagen. Dargestellt sind die Verteilungen der Sprünge (mit Vorzeichen) für
Messung und erzeugtes Profil sowie jeweils der mittlere Betrag als Linienpaar bei ±Ø|Sprung|.
Der Sprungfaktor ist das Verhältnis der mittleren Beträge (erzeugt / real).

Einheit: Last / Jahresmittellast. Die x-Achse endet beim Quantil X_UPPER_Q der Beträge; die
Mittelwerte enthalten alle Sprünge.

Ausgewählt wird das Gebäude, dessen Sprungfaktor dem Quantil QUANTILE über die Testgebäude am
nächsten liegt (Default: Median); alternativ mit --sensor.

Datenquelle: data/samples/<version>/.

    python -m abbildungen.plot_jump_factor_example [--version main] [--variant deployment]
                                                   [--quantile 0.5 | --sensor <id>]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from abbildungen.plot_duration_curve_example import category_of
from abbildungen.plot_style import GENERATED_COLOR, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure
from src.evaluation.compare_cached_samples import load_cache
from src.evaluation.evaluation_metrics import day_boundary_error_per_sensor

OUT_DIR = Path("abbildungen/outputs")
QUANTILE = 0.5
MIN_TRANSITIONS = 100                  # Mindestanzahl Tagesübergänge je Gebäude
X_UPPER_Q = 0.95                       # x-Achse bis ± diesem Quantil der Beträge
HALF_BINS = 20                         # Klassen je Seite, zusätzlich eine Klasse um 0
LINE_TOP = 0.78                        # Höhe der Ø-|Sprung|-Linien als Achsenanteil


def comma(fmt: str):
    return FuncFormatter(lambda v, _: fmt.format(v).replace(".", ",").replace("-", "−"))


def boundary_jumps(cache: dict, sensor_id: str) -> pd.DataFrame:
    """Realer und erzeugter Lastsprung (Stunde 0 minus Stunde 23 des Vortags) je Tagesübergang
    zwischen aufeinanderfolgenden Tagen eines Gebäudes."""
    idx = np.where(cache["sensor_id"] == sensor_id)[0]
    idx = idx[np.argsort(cache["day"][idx])]
    yr, yg = cache["y_real"][idx].astype(np.float64), cache["y_sample"][idx].astype(np.float64)
    consecutive = np.diff(cache["day"][idx]) == np.timedelta64(1, "D")
    return pd.DataFrame({"real": yr[1:, 0] - yr[:-1, 23], "gen": yg[1:, 0] - yg[:-1, 23]})[consecutive]


def pick_sensor(per_sensor: pd.DataFrame, quantile: float) -> pd.Series:
    ok = per_sensor[(per_sensor["n_transitions"] >= MIN_TRANSITIONS) & np.isfinite(per_sensor["jump_factor"])]
    target = ok["jump_factor"].quantile(quantile)
    return ok.loc[(ok["jump_factor"] - target).abs().idxmin()]


def draw(ax, jumps: pd.DataFrame) -> tuple[float, float]:
    real, gen = jumps["real"].to_numpy(), jumps["gen"].to_numpy()
    upper = np.quantile(np.abs(np.r_[real, gen]), X_UPPER_Q)
    width = upper / (HALF_BINS + 0.5)
    bins = (np.arange(-HALF_BINS - 1, HALF_BINS + 1) + 0.5) * width     # eine Klasse um 0 zentriert

    ax.hist(gen, bins, histtype="step", color=GENERATED_COLOR, lw=1.6, zorder=2)
    ax.hist(real, bins, histtype="step", color=REAL_COLOR, lw=1.6, zorder=3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)

    # Mittlerer Betrag als Linienpaar ±m je Verteilung
    m_real, m_gen = np.abs(real).mean(), np.abs(gen).mean()
    for m, color, label, y in ((m_real, REAL_COLOR, "real", 0.97), (m_gen, GENERATED_COLOR, "erzeugt", 0.88)):
        for s in (-1, 1):
            ax.axvline(s * m, ymax=LINE_TOP, color=color, lw=1.2, ls=(0, (5, 3)), zorder=4)
        ax.annotate(f"Ø |Sprung| {label} {m:.2f}".replace(".", ","), (m, y), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=10, color=REAL_COLOR)
    ax.text(0.99, 0.70, f"Sprungfaktor\n= {m_gen:.2f} / {m_real:.2f}\n= {m_gen / m_real:.2f}".replace(".", ","),
            transform=ax.transAxes, ha="right", va="top", fontsize=11)

    ax.set_xlim(bins[0], bins[-1])
    ax.xaxis.set_major_formatter(comma("{:g}"))
    ax.set_xlabel("Lastsprung / Jahresmittellast [-]")
    ax.set_ylabel("Anzahl Tagesübergänge")
    ax.grid(False)
    handles = [Line2D([], [], color=REAL_COLOR, lw=1.6, label="Messung"),
               Line2D([], [], color=GENERATED_COLOR, lw=1.6, label="Erzeugtes Profil")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)   # Legende über der Achse
    return m_real, m_gen


def report(chosen: pd.Series, per_sensor: pd.DataFrame, jumps: pd.DataFrame, m_real: float, m_gen: float,
           category, args) -> None:
    ok = per_sensor[(per_sensor["n_transitions"] >= MIN_TRANSITIONS) & np.isfinite(per_sensor["jump_factor"])]
    upper = np.quantile(np.abs(np.r_[jumps["real"], jumps["gen"]]), X_UPPER_Q)
    print(f"\n[{args.version} / {args.variant}] Sensor {chosen['sensor_id']}, Kategorie {category!r}, "
          f"{len(jumps)} Tagesuebergaenge")
    print(f"  Sprungfaktor {m_gen / m_real:.3f} (= {m_gen:.3f} / {m_real:.3f}); Median ueber {len(ok)} Gebaeude "
          f"mit >= {MIN_TRANSITIONS} Uebergaengen {ok['jump_factor'].median():.3f}, Rang "
          f"{(ok['jump_factor'] < chosen['jump_factor']).mean():.0%}")
    print(f"  Mittel mit Vorzeichen real {jumps['real'].mean():+.3f}  erzeugt {jumps['gen'].mean():+.3f}   "
          f"Anteil positiv real {(jumps['real'] > 0).mean():.0%}  erzeugt {(jumps['gen'] > 0).mean():.0%}")
    print(f"  x-Achse bis +-{upper:.3f}; ausserhalb: real {(jumps['real'].abs() > upper).sum()}, "
          f"erzeugt {(jumps['gen'].abs() > upper).sum()} (in den Mittelwerten enthalten)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="main", help="version_suffix des Checkpoints")
    ap.add_argument("--variant", default="deployment", choices=["deployment", "oracle"])
    ap.add_argument("--quantile", type=float, default=QUANTILE,
                    help="Quantil des Sprungfaktors ueber die Test-Gebaeude (Default Median)")
    ap.add_argument("--sensor", help="bestimmtes Gebaeude statt Quantil-Auswahl")
    args = ap.parse_args()

    cache = load_cache(args.version, args.variant)
    if cache is None:
        raise SystemExit(f"Kein {args.variant}-Cache fuer {args.version} unter data/samples/.")
    per_sensor = day_boundary_error_per_sensor(cache["y_real"], cache["y_sample"], cache["sensor_id"], cache["day"])
    if args.sensor is None:
        chosen = pick_sensor(per_sensor, args.quantile)
    else:
        sel = per_sensor[per_sensor["sensor_id"] == args.sensor]
        if sel.empty:
            raise SystemExit(f"Sensor {args.sensor} nicht im Test-Cache.")
        chosen = sel.iloc[0]
    jumps = boundary_jumps(cache, chosen["sensor_id"])

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=fig_size(1.0, 0.55))
    m_real, m_gen = draw(ax, jumps)
    # Abgleich mit dem Sprungfaktor aus evaluation_metrics
    assert abs(m_gen / m_real - chosen["jump_factor"]) < 1e-4, (m_gen / m_real, chosen["jump_factor"])
    despine(fig)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"jump_factor_example_{args.version}_{args.variant}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")
    report(chosen, per_sensor, jumps, m_real, m_gen, category_of(chosen["sensor_id"]), args)


if __name__ == "__main__":
    main()
