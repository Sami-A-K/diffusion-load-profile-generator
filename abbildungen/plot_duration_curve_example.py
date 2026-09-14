"""
Jahresdauerlinie der Messung und des erzeugten Profils eines Testgebäudes, mit der Fläche
zwischen beiden Kurven.

Die y-Achse ist auf die Jahresmittellast normiert (Skala des Modells). Die Fläche geteilt durch
die Stundenzahl entspricht dem globalen Dauerlinienfehler (error_global).

Gezeigt wird ein vollständiges Kalenderjahr im Einsatzfall (deployment). Ausgewählt wird das
Sensor-Jahr, dessen globaler Fehler dem Quantil QUANTILE aller vollständigen Test-Sensor-Jahre am
nächsten liegt (Default: Median); alternativ mit --sensor/--year.

Datenquelle: data/samples/<version>/.

    python -m abbildungen.plot_duration_curve_example [--version main] [--variant deployment]
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

from abbildungen.plot_style import COLORS, GENERATED_COLOR, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure
from src.data.encode_covariates import decode_category
from src.evaluation.compare_cached_samples import SENSOR_TABLE, load_cache
from src.evaluation.evaluation_metrics import duration_curve_error

OUT_DIR = Path("abbildungen/outputs")
QUANTILE = 0.5
AREA_ALPHA = 0.30


def num(v: float, fmt: str) -> str:
    """Deutsches Zahlenformat: Tausenderpunkt, Dezimalkomma."""
    return fmt.format(v).replace(",", "_").replace(".", ",").replace("_", ".")


def sensor_years(cache: dict) -> pd.DataFrame:
    """Je (Sensor, Kalenderjahr): Tageszahl, Vollständigkeit und globaler Dauerlinienfehler."""
    day = pd.DatetimeIndex(cache["day"])
    df = pd.DataFrame({"sensor_id": cache["sensor_id"], "year": day.year, "row": np.arange(len(day))})
    rows = []
    for (sid, year), g in df.groupby(["sensor_id", "year"]):
        idx = g["row"].to_numpy()
        err = duration_curve_error([cache["y_real"][idx].ravel()], [cache["y_sample"][idx].ravel()])[0]
        rows.append({"sensor_id": sid, "year": year, "n_days": len(idx),
                     "full": len(idx) == pd.Timestamp(year, 12, 31).dayofyear, "error_global": err})
    return pd.DataFrame(rows)


def pick(years: pd.DataFrame, quantile: float) -> pd.Series:
    full = years[years["full"]]
    if full.empty:
        raise SystemExit("Kein vollstaendiges Kalenderjahr im Cache - Jahresdauerlinie nicht darstellbar.")
    target = full["error_global"].quantile(quantile)
    return full.loc[(full["error_global"] - target).abs().idxmin()]


def curves(cache: dict, sensor_id: str, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Reale und erzeugte Jahresdauerlinie (absteigend sortierte Stundenwerte)."""
    m = (cache["sensor_id"] == sensor_id) & (pd.DatetimeIndex(cache["day"]).year == year)
    if not m.any():
        raise SystemExit(f"Keine Tage fuer Sensor {sensor_id} im Jahr {year} im Cache.")
    real = np.sort(cache["y_real"][m].ravel())[::-1].astype(np.float64)
    gen = np.sort(cache["y_sample"][m].ravel())[::-1].astype(np.float64)
    return real, gen


def category_of(sensor_id: str) -> str:
    """Kategorie eines Sensors aus der Sensor-Tabelle."""
    cat = pd.read_parquet(SENSOR_TABLE, columns=["category"])["category"]
    return decode_category(np.asarray(cat.xs(sensor_id, level=0).iloc[0]))


def draw(ax, real: np.ndarray, gen: np.ndarray) -> float:
    hours = np.arange(1, len(real) + 1)
    error = float(np.mean(np.abs(real - gen)))   # = Fläche / Stundenzahl = error_global

    ax.fill_between(hours, real, gen, color=GENERATED_COLOR, alpha=AREA_ALPHA, lw=0, zorder=1)
    ax.plot(hours, real, color=REAL_COLOR, lw=1.6, zorder=3)
    ax.plot(hours, gen, color=GENERATED_COLOR, lw=1.6, zorder=2)
    ax.axhline(1.0, color=COLORS["gray"], lw=0.8, ls=(0, (2, 3)), zorder=0)
    ax.text(len(real), 1.0, "Jahresmittellast", ha="right", va="bottom", fontsize=10, color=COLORS["gray"])

    ax.set_xlim(0, len(real))
    ax.set_ylim(bottom=min(0.0, gen.min(), real.min()), top=max(real[0], gen[0]) * 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: num(x, "{:,.0f}")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))
    ax.set_xlabel("Stunden im Jahr [h]")
    ax.set_ylabel("Last / Jahresmittellast [-]")

    handles = [Line2D([], [], color=REAL_COLOR, lw=1.6, label="Messung"),
               Line2D([], [], color=GENERATED_COLOR, lw=1.6, label="Erzeugtes Profil"),
               Patch(facecolor=GENERATED_COLOR, alpha=AREA_ALPHA, lw=0,
                     label="Fläche zwischen den Dauerlinien")]
    ax.legend(handles=handles, loc="upper right")
    return error


def report(chosen: pd.Series, years: pd.DataFrame, real: np.ndarray, gen: np.ndarray, error: float,
           category, args) -> None:
    full = years[years["full"]]
    rank = (full["error_global"] < chosen["error_global"]).mean()
    print(f"\n[{args.version} / {args.variant}] Sensor {chosen['sensor_id']}, Jahr {chosen['year']}, "
          f"Kategorie {category!r}, {chosen['n_days']} Tage = {len(real)} h")
    print(f"  globaler Dauerlinienfehler {error * 100:.2f} % der Jahresmittellast "
          f"(Flaeche {np.sum(np.abs(real - gen)):.0f} h, Rang {rank:.0%} unter {len(full)} vollst. Sensor-Jahren; "
          f"Median dort {full['error_global'].median() * 100:.2f} %)")
    # Aufteilung in zu hoch und zu niedrig erzeugte Last, bezogen auf den Jahresverbrauch
    over = np.clip(gen - real, 0, None).sum() / real.sum()
    under = np.clip(real - gen, 0, None).sum() / real.sum()
    print(f"  Anteil am Jahresverbrauch: zu hoch erzeugt {over * 100:.2f} %  + zu niedrig {under * 100:.2f} %  "
          f"= {(over + under) * 100:.2f} %   (Energiebilanz {(over - under) * 100:+.2f} %)")
    print(f"  Mittel Messung {real.mean():.3f} (Soll 1)  erzeugt {gen.mean():.3f}   "
          f"Spitze Messung {real[0]:.2f}  erzeugt {gen[0]:.2f}   Minimum {real[-1]:.2f} / {gen[-1]:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="main", help="version_suffix des Checkpoints")
    ap.add_argument("--variant", default="deployment", choices=["deployment", "oracle"])
    ap.add_argument("--quantile", type=float, default=QUANTILE,
                    help="Quantil des globalen Fehlers ueber die vollstaendigen Sensor-Jahre (Default Median)")
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
    real, gen = curves(cache, chosen["sensor_id"], int(chosen["year"]))

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=fig_size(1.0, 0.55))
    error = draw(ax, real, gen)
    despine(fig)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"duration_curve_example_{args.version}_{args.variant}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")
    report(chosen, years, real, gen, error, category_of(chosen["sensor_id"]), args)


if __name__ == "__main__":
    main()
