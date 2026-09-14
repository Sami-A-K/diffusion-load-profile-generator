"""
Zwei Realisierungen des Modells für dasselbe Testgebäude und dieselbe Woche, die sich nur im
Generierungsrauschen unterscheiden (--noise-seed in sample_population.py; Kovariaten und
Messfenster identisch). Variante oracle: jeder Tag unabhängig mit echtem Vortageswert.

Die Konsolenausgabe stellt den stundenweisen Fehler dem Dauerlinienfehler derselben Woche gegenüber.

Auswahl:
  1. Gebäude: Zeitkontext-Fehler (results/cached_eval_per_sensor.csv, main/oracle) am nächsten am
     Quantil SENSOR_QUANTILE (Default: Median).
  2. Woche: unter den vollständigen Wochen (Mo-So) mit einer Spannweite zwischen den Quantilen
     RANGE_QUANTILE_LOW und RANGE_QUANTILE_HIGH des Gebäudes diejenige mit der größten Differenz
     zwischen stundenweisem Fehler und Dauerlinienfehler.
--sensor/--start wählen eine bestimmte Woche.

Datenquelle: data/samples/main/best_oracle_test.npz und best_oracle_test_nseed1.npz.

    python -m abbildungen.plot_stochasticity_example [--sensor <id> --start <JJJJ-MM-TT>]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from abbildungen.plot_duration_curve_example import category_of
from abbildungen.plot_style import ACCENT_CYCLE, COLORS, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure
from src.evaluation.compare_cached_samples import load_cache

OUT_DIR = Path("abbildungen/outputs")
PER_SENSOR_RESULTS = Path("results/cached_eval_per_sensor.csv")
VERSION = "main"
SENSOR_QUANTILE = 0.5       # Quantil des Zeitkontext-Fehlers für die Gebäudeauswahl
RANGE_QUANTILE_LOW = 0.5    # untere Grenze der Wochen-Spannweite
RANGE_QUANTILE_HIGH = 0.95  # obere Grenze der Wochen-Spannweite
REALIZATION_TAGS = ["", "_nseed1"]   # gleiches Messfenster, unterschiedliches Rauschen
REALIZATION_LABELS = ["Realisierung 1", "Realisierung 2"]
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def load_realizations(version: str) -> dict[str, dict]:
    caches = {}
    for tag in REALIZATION_TAGS:
        cache = load_cache(version, "oracle", cache_tag=tag)
        if cache is None:
            raise SystemExit(f"Cache best_oracle_test{tag}.npz fuer {version} fehlt unter data/samples/{version}/ - "
                             f"mit sample_population.py (--variant oracle --noise-seed ...) erzeugen.")
        caches[tag] = cache
    # Messwerte, Sensoren und Tage müssen in allen Caches übereinstimmen
    base_real = caches[REALIZATION_TAGS[0]]["y_real"]
    for tag in REALIZATION_TAGS[1:]:
        if caches[tag]["y_real"].shape != base_real.shape or not np.array_equal(caches[tag]["y_real"], base_real):
            raise SystemExit(f"y_real in best_oracle_test{tag}.npz weicht von der Basis ab - Caches passen nicht "
                             f"zusammen (unterschiedliche Population/Version?).")
    for tag in REALIZATION_TAGS[1:]:
        if not (np.array_equal(caches[tag]["sensor_id"], caches[REALIZATION_TAGS[0]]["sensor_id"])
                and np.array_equal(caches[tag]["day"], caches[REALIZATION_TAGS[0]]["day"])):
            raise SystemExit(f"Zeilenreihenfolge in best_oracle_test{tag}.npz weicht von der Basis ab.")
    return caches


def typical_sensor(version: str, quantile: float = SENSOR_QUANTILE) -> tuple[str, pd.Series]:
    """Testgebäude, dessen Zeitkontext-Fehler (oracle) dem Quantil quantile am nächsten liegt."""
    if not PER_SENSOR_RESULTS.exists():
        raise SystemExit(f"{PER_SENSOR_RESULTS} fehlt - erst compare_cached_samples.py laufen lassen.")
    per_sensor = pd.read_csv(PER_SENSOR_RESULTS)
    ok = per_sensor[(per_sensor["version_suffix"] == version) & (per_sensor["variant"] == "oracle")]
    if ok.empty:
        raise SystemExit(f"Keine oracle-Zeile fuer {version} in {PER_SENSOR_RESULTS}.")
    target = ok["error_context"].quantile(quantile)
    row = ok.loc[(ok["error_context"] - target).abs().idxmin()]
    rank = (ok["error_context"] < row["error_context"]).mean()
    print(f"[typical_sensor] {row['sensor_id']}: Zeitkontext-Fehler {row['error_context'] * 100:.2f} % "
          f"(Rang {rank:.0%} unter {len(ok)} Test-Gebaeuden, Ziel Quantil {quantile:.0%})")
    return row["sensor_id"], row


def full_weeks(caches: dict[str, dict], sensor_id: str) -> pd.DataFrame:
    """Vollständige Wochen (Mo-So) eines Sensors: Zeilenpositionen, Spannweite der Messung sowie
    stundenweiser Fehler und Dauerlinienfehler, gemittelt über die Realisierungen."""
    base = caches[REALIZATION_TAGS[0]]
    mask = base["sensor_id"] == sensor_id
    idx = pd.DataFrame({"sensor_id": base["sensor_id"][mask], "day": pd.DatetimeIndex(base["day"][mask]),
                        "row": np.flatnonzero(mask)}).sort_values(["sensor_id", "day"])
    sensor_ids, starts, row_windows = [], [], []
    for sid, g in idx.groupby("sensor_id", sort=False):
        g = g.reset_index(drop=True)
        mondays = np.flatnonzero(g["day"].dt.dayofweek.to_numpy() == 0)
        for i in mondays:
            if i + 7 > len(g):
                continue
            window = g.iloc[i:i + 7]
            if not np.all(np.diff(window["day"].to_numpy()) == np.timedelta64(1, "D")):
                continue
            sensor_ids.append(sid)
            starts.append(window["day"].iloc[0])
            row_windows.append(window["row"].to_numpy())

    rows = np.stack(row_windows)                                   # (n_weeks, 7)
    real = base["y_real"][rows].reshape(len(rows), 168).astype(np.float64)
    real_sorted = np.sort(real, axis=1)[:, ::-1]

    pointwise_error = np.zeros(len(rows))
    sorted_error = np.zeros(len(rows))
    for tag in REALIZATION_TAGS:
        gen = caches[tag]["y_sample"][rows].reshape(len(rows), 168).astype(np.float64)
        pointwise_error += np.mean(np.abs(real - gen), axis=1)
        sorted_error += np.mean(np.abs(real_sorted - np.sort(gen, axis=1)[:, ::-1]), axis=1)
    pointwise_error /= len(REALIZATION_TAGS)
    sorted_error /= len(REALIZATION_TAGS)

    return pd.DataFrame({"sensor_id": sensor_ids, "start": starts, "rows": row_windows,
                         "range": real.max(axis=1) - real.min(axis=1),
                         "pointwise_error": pointwise_error, "sorted_error": sorted_error,
                         "gap": pointwise_error - sorted_error})


def pick_week(weeks: pd.DataFrame) -> pd.Series:
    """Woche im Spannweitenbereich mit der größten Differenz zwischen stundenweisem Fehler und
    Dauerlinienfehler."""
    lo, hi = weeks["range"].quantile([RANGE_QUANTILE_LOW, RANGE_QUANTILE_HIGH])
    candidates = weeks[(weeks["range"] >= lo) & (weeks["range"] <= hi)]
    if candidates.empty:
        candidates = weeks
    return candidates.loc[candidates["gap"].idxmax()]


def week_rows(base: dict, sensor_id: str, start: pd.Timestamp) -> np.ndarray:
    """Zeilenpositionen einer bestimmten Woche (für --sensor/--start)."""
    days = pd.date_range(start, periods=7, freq="D")
    idx = pd.MultiIndex.from_arrays([base["sensor_id"], pd.DatetimeIndex(base["day"])])
    pos = pd.Series(np.arange(len(base["sensor_id"])), index=idx)
    key = pd.MultiIndex.from_arrays([[sensor_id] * 7, days])
    if key.difference(pos.index).size:
        raise SystemExit(f"Sensor {sensor_id} hat keine vollstaendige Woche ab {start.date()} im oracle-Test-Cache.")
    return pos.loc[key].to_numpy()


def week_values(caches: dict[str, dict], rows: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reale und je Realisierung erzeugte Stundenwerte der Woche (168 Werte, chronologisch)."""
    real_week = caches[REALIZATION_TAGS[0]]["y_real"][rows].ravel().astype(np.float64)
    gen_week = {tag: caches[tag]["y_sample"][rows].ravel().astype(np.float64) for tag in REALIZATION_TAGS}
    return real_week, gen_week


def draw(ax_time, real_week: np.ndarray, gen_week: dict[str, np.ndarray]) -> None:
    hours = np.arange(168)
    for d in range(1, 7):
        ax_time.axvline(d * 24, color=COLORS["grid"], lw=0.8, zorder=0)
    for tag, color in zip(REALIZATION_TAGS, ACCENT_CYCLE):
        ax_time.plot(hours, gen_week[tag], color=color, lw=1.2, zorder=3)
    ax_time.plot(hours, real_week, color=REAL_COLOR, lw=1.2, zorder=5)
    ax_time.set_xlim(0, 167)
    ax_time.set_xticks(np.arange(12, 168, 24))
    ax_time.set_xticklabels(WEEKDAYS)
    ax_time.set_ylabel("Last / Jahresmittellast [-]")

    handles = [Line2D([], [], color=REAL_COLOR, lw=1.8, label="Messung")]
    handles += [Line2D([], [], color=c, lw=1.3, label=lbl) for c, lbl in zip(ACCENT_CYCLE, REALIZATION_LABELS)]
    # Legende über der Achse
    ax_time.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=10)


def report(sensor_id: str, start: pd.Timestamp, real_week: np.ndarray, gen_week: dict[str, np.ndarray],
          weeks: pd.DataFrame, category: str) -> None:
    this = weeks[(weeks["sensor_id"] == sensor_id) & (weeks["start"] == start)].iloc[0]
    rank_range = (weeks["range"] < this["range"]).mean()
    rank_gap = (weeks["gap"] < this["gap"]).mean()
    print(f"\nSensor {sensor_id}, Kategorie {category!r}, Woche {start.date()} - {(start + pd.Timedelta(days=6)).date()}")
    print(f"  Spannweite Messung {this['range']:.2f} (Rang {rank_range:.0%} unter {len(weeks)} vollst. Wochen DIESES "
          f"Gebaeudes); Fehlerluecke Rang {rank_gap:.0%}")

    pointwise = {tag: np.mean(np.abs(real_week - gen_week[tag])) for tag in REALIZATION_TAGS}
    sorted_err = {tag: np.mean(np.abs(np.sort(real_week)[::-1] - np.sort(gen_week[tag])[::-1])) for tag in REALIZATION_TAGS}
    print("  Realisierung   stundenweise |Δ| zur Messung   Dauerlinien-|Δ| (sortiert)")
    for tag, label in zip(REALIZATION_TAGS, REALIZATION_LABELS):
        print(f"    {label:<14} {pointwise[tag] * 100:6.2f} %                    {sorted_err[tag] * 100:6.2f} %")

    pairs = [(a, b) for i, a in enumerate(REALIZATION_TAGS) for b in REALIZATION_TAGS[i + 1:]]
    among = np.mean([np.mean(np.abs(gen_week[a] - gen_week[b])) for a, b in pairs])
    print(f"  stundenweise |Δ| ZWISCHEN den Realisierungen (Mittel aller Paare): {among * 100:.2f} % "
          f"- vergleichbar mit |Δ| zur Messung oben, obwohl beide dieselbe Konditionierung teilen.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", help="bestimmtes Gebaeude statt automatischer Auswahl (mit --start)")
    ap.add_argument("--start", help="Montag der Woche, JJJJ-MM-TT (mit --sensor)")
    args = ap.parse_args()
    if (args.sensor is None) != (args.start is None):
        ap.error("--sensor und --start nur gemeinsam")

    caches = load_realizations(VERSION)
    if args.sensor is None:
        sensor_id, _ = typical_sensor(VERSION)
        weeks = full_weeks(caches, sensor_id)
        chosen = pick_week(weeks)
        start, rows = chosen["start"], chosen["rows"]
    else:
        start = pd.Timestamp(args.start)
        if start.dayofweek != 0:
            ap.error(f"--start {args.start} ist kein Montag.")
        sensor_id = args.sensor
        weeks = full_weeks(caches, sensor_id)
        rows = week_rows(caches[REALIZATION_TAGS[0]], sensor_id, start)
    real_week, gen_week = week_values(caches, rows)

    apply_thesis_style()
    fig, ax_time = plt.subplots(figsize=fig_size(1.0, 0.55))
    draw(ax_time, real_week, gen_week)
    despine(fig)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"stochasticity_example_{VERSION}"
    save_figure(fig, out)
    print(f"-> {out}.pdf / .png")
    report(sensor_id, start, real_week, gen_week, weeks, category_of(sensor_id))


if __name__ == "__main__":
    main()
