"""
Mittlerer Sprung je Stundenübergang, real gegen erzeugt: die 23 Übergänge innerhalb des Tages und
der Übergang über die Tagesgrenze (Stunde 23 -> Stunde 0 des Folgetags). Der Wert an der
Tagesgrenze entspricht dem Sprungfaktor (evaluation_metrics.day_boundary_error_per_sensor).

Oben der mittlere |Sprung| je Übergang in Prozent der Jahresmittellast, unten das Verhältnis
erzeugt / real. Beides je Testgebäude berechnet, Median über die Gebäude, dann Mittel über die
Trainingsläufe je Modell. Die Messkurve stammt aus den main-Caches.

Datenquelle: data/samples/<version>/, Läufe aus results/cached_eval_summary.csv.

    python -m abbildungen.plot_jump_factor_transitions
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from abbildungen.plot_style import COLORS, REAL_COLOR, apply_thesis_style, despine, fig_size, save_figure
from src.evaluation.compare_cached_samples import load_cache

IN_CSV = Path("results/cached_eval_summary.csv")
OUT = Path("abbildungen/outputs/jump_factor_transitions")
VARIANT = "deployment"
# Läufe werden über (arch, patch) ausgewählt
MODELS = {
    "baseline": {"label": "Baseline", "color": COLORS["magenta"], "marker": "s", "arch": "baseline", "patch": "noenc"},
    "main": {"label": "mit Encoder", "color": COLORS["red"], "marker": "o", "arch": "base", "patch": "p8"},
}
BOUNDARY = 23                       # Spalte h = Übergang h -> h+1, Spalte 23 = 23 -> 0 des Folgetags
BAND_COLOR = COLORS["grid"]


def transition_means(y: np.ndarray, sensor_ids: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    """Mittlerer |Sprung| je Sensor (Zeile) und Stundenübergang (Spalte). Die Tagesgrenze zählt nur
    zwischen aufeinanderfolgenden Tagen desselben Sensors."""
    order = np.lexsort((dates, sensor_ids))
    s, d, y = sensor_ids[order], dates[order], y[order].astype(np.float64)
    steps = np.full((len(y), BOUNDARY + 1), np.nan)
    steps[:, :BOUNDARY] = np.abs(np.diff(y, axis=1))
    consecutive = (s[1:] == s[:-1]) & ((d[1:] - d[:-1]) == np.timedelta64(1, "D"))
    steps[:-1, BOUNDARY] = np.where(consecutive, np.abs(y[1:, 0] - y[:-1, 23]), np.nan)
    return pd.DataFrame(steps).groupby(s).mean()


def run_curves(version: str) -> dict[str, pd.Series]:
    """Median über die Testgebäude: mittlerer Sprung real, erzeugt und Verhältnis je Gebäude."""
    cache = load_cache(version, VARIANT)
    if cache is None:
        raise SystemExit(f"Kein {VARIANT}-Cache fuer {version} unter data/samples/.")
    real = transition_means(cache["y_real"], cache["sensor_id"], cache["day"])
    gen = transition_means(cache["y_sample"], cache["sensor_id"], cache["day"])
    # Unendliche Verhältnisse (realer Sprung 0) ausschließen
    factor = (gen / real).replace([np.inf, -np.inf], np.nan)
    return {"real": real.median(), "gen": gen.median(), "factor": factor.median()}


def load() -> dict[str, dict]:
    summary = pd.read_csv(IN_CSV)
    summary = summary[summary["variant"] == VARIANT]
    out = {}
    for key, m in MODELS.items():
        runs = summary[(summary["arch"] == m["arch"]) & (summary["patch"] == m["patch"])]
        assert len(runs) > 1, f"{key}: nur {len(runs)} Lauf/Laeufe in {IN_CSV}"
        curves = {row.version_suffix: run_curves(row.version_suffix) for row in runs.itertuples()}
        # Abgleich des Werts an der Tagesgrenze mit jump_factor_median
        for row in runs.itertuples():
            got = curves[row.version_suffix]["factor"][BOUNDARY]
            assert abs(got - row.jump_factor_median) < 1e-4, \
                f"{row.version_suffix}: Grenzpunkt {got:.5f} != jump_factor_median {row.jump_factor_median:.5f}"
        out[key] = {name: pd.concat([c[name] for c in curves.values()], axis=1)
                    for name in ("real", "gen", "factor")}
    return out


def comma(fmt: str):
    return FuncFormatter(lambda v, _: fmt.format(v).replace(".", ","))


def draw(ax_abs, ax_fac, data: dict[str, dict]) -> None:
    x = np.arange(BOUNDARY + 1)
    for ax in (ax_abs, ax_fac):
        ax.axvspan(BOUNDARY - 0.5, BOUNDARY + 0.5, color=BAND_COLOR, alpha=0.6, lw=0, zorder=0)

    ax_abs.plot(x, data["main"]["real"].mean(axis=1) * 100, color=REAL_COLOR, lw=1.6, zorder=3)
    ax_fac.axhline(1.0, color=REAL_COLOR, lw=0.8, ls=(0, (5, 4)), zorder=1)
    for key, m in MODELS.items():
        gen = data[key]["gen"].mean(axis=1) * 100
        factor = data[key]["factor"].mean(axis=1)
        style = dict(color=m["color"], lw=1.6, zorder=2)
        ax_abs.plot(x, gen, **style)
        ax_fac.plot(x, factor, **style)
        # Punkt an der Tagesgrenze, unten mit Wert beschriftet
        ax_abs.plot(BOUNDARY, gen[BOUNDARY], m["marker"], ms=6.5, color=m["color"], zorder=4)
        ax_fac.plot(BOUNDARY, factor[BOUNDARY], m["marker"], ms=6.5, color=m["color"], zorder=4)
        ax_fac.annotate(f"{factor[BOUNDARY]:.2f}".replace(".", ","), (BOUNDARY, factor[BOUNDARY]),
                        xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=10)

    ax_abs.text(BOUNDARY, ax_abs.get_ylim()[1], "Tages-\ngrenze", ha="center", va="bottom", fontsize=10,
                color=COLORS["gray"])
    ax_fac.annotate("real", (-0.3, 1.0), xytext=(0, -3), textcoords="offset points", ha="left", va="top",
                    fontsize=10, color=REAL_COLOR)

    ticks = list(range(0, BOUNDARY, 4)) + [BOUNDARY]
    ax_fac.set_xticks(ticks)
    ax_fac.set_xticklabels([f"{h}→{(h + 1) % 24}" for h in ticks])
    ax_fac.set_xlim(-0.5, BOUNDARY + 1.6)          # Platz für die Beschriftung rechts
    ax_fac.set_xlabel("Stundenübergang")
    ax_abs.set_ylim(bottom=0)
    ax_fac.set_ylim(bottom=0.8)
    ax_abs.set_ylabel("Mittlerer Sprung [%]")
    ax_fac.set_ylabel("erzeugt / real [-]")
    ax_abs.yaxis.set_major_formatter(comma("{:g}"))
    ax_fac.yaxis.set_major_formatter(comma("{:g}"))

    handles = [Line2D([], [], color=REAL_COLOR, lw=1.6, label="Messung")] + [
        Line2D([], [], color=m["color"], lw=1.6, marker=m["marker"], ms=6.5, label=m["label"])
        for m in MODELS.values()]
    ax_abs.legend(handles=handles, loc="upper left", ncol=3)


def report(data: dict[str, dict]) -> None:
    print(f"\n[{VARIANT}] Median ueber die Test-Gebaeude, Spanne = ueber die Trainingslaeufe")
    for key, m in MODELS.items():
        f = data[key]["factor"]
        inner = f.iloc[:BOUNDARY]
        print(f"  {m['label']:<12} n={f.shape[1]}  Tagesgrenze {f.iloc[BOUNDARY].mean():.3f} "
              f"(Spanne {f.iloc[BOUNDARY].min():.3f}..{f.iloc[BOUNDARY].max():.3f})   "
              f"innerhalb des Tages {inner.mean(axis=1).min():.3f}..{inner.mean(axis=1).max():.3f}")
        print(f"  {'':<12} Sprung an der Grenze real {data[key]['real'].iloc[BOUNDARY].mean() * 100:.1f} %  "
              f"erzeugt {data[key]['gen'].iloc[BOUNDARY].mean() * 100:.1f} % der Jahresmittellast")


def main() -> None:
    data = load()
    apply_thesis_style()
    fig, (ax_abs, ax_fac) = plt.subplots(2, 1, sharex=True, figsize=fig_size(1.0, 0.78))
    draw(ax_abs, ax_fac, data)
    despine(fig)
    fig.tight_layout(h_pad=1.2)
    save_figure(fig, OUT)
    print(f"-> {OUT}.pdf / .png   (VARIANT = {VARIANT})")
    report(data)


if __name__ == "__main__":
    main()
