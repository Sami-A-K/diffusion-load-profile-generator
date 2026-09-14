"""
Erkennung von Strom-Sensoren mit PV-Eigenverbrauch.

Ein Sensor wird markiert, wenn beide Kriterien zutreffen:

1. Negative Korrelation zwischen täglicher Energie und täglicher Sonnenscheindauer (Meteostat
   tsun, ersatzweise 100 - Wolkenbedeckung) im Sommerhalbjahr, bereinigt um den Wochentagseffekt.
2. Mittagseinbruch im Sommer-Werktagsprofil: mittlere Last 11-15 Uhr im Verhältnis zu
   7-10 Uhr und 16-19 Uhr.

Die Korrelation allein liefert bei öffentlichen Gebäuden falsch-positive Treffer, da
Sommerferien und sonnige Wochen zeitlich zusammenfallen. Das Profilkriterium schließt diese
Fälle aus.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


SUMMER_MONTHS = {4, 5, 6, 7, 8, 9}
MIN_DAYS = 60  # Mindestanzahl gemeinsamer Sommertage für die Korrelation
TSUN_COVERAGE_THRESHOLD = 0.3  # Mindestanteil valider tsun-Werte, sonst Wolkenbedeckung
DEFAULT_CORRELATION_THRESHOLD = -0.10

# Mittagsstunden vs. Vormittag/Abend; Nachtstunden bleiben unberücksichtigt.
MIDDAY_HOURS = slice(11, 16)
SHOULDER_HOURS = (slice(7, 11), slice(16, 20))
DEFAULT_DIP_RATIO_THRESHOLD = 0.9


def daily_energy_from_series(hourly: pd.Series) -> pd.Series:
    """Tägliche Summe einer stündlichen kWh-Reihe."""
    daily = hourly.resample("D").sum(min_count=20)
    return daily.dropna()


def summer_weekday_dip_ratio(hourly: pd.Series) -> float | None:
    """
    Verhältnis Mittag / Vor- und Nachmittag im mittleren Sommer-Werktagsprofil (normiert auf das
    Jahresmaximum). Werte < 1 bedeuten einen Einbruch zur Mittagszeit.
    """
    normalized = hourly / hourly.groupby(hourly.index.year).transform("max")
    summer_weekday = normalized[normalized.index.month.isin(SUMMER_MONTHS) & (normalized.index.dayofweek < 5)]
    if summer_weekday.empty:
        return None
    profile = summer_weekday.groupby(summer_weekday.index.hour).mean().to_numpy()
    if len(profile) < 24 or np.isnan(profile).any() or np.all(profile == 0):
        return None
    midday = profile[MIDDAY_HOURS]
    shoulders = np.concatenate([profile[SHOULDER_HOURS[0]], profile[SHOULDER_HOURS[1]]])
    return float(midday.mean() / shoulders.mean())


def _read_bulk_hourly(station_id: str, bulk_dir: Path) -> pd.DataFrame:
    """Liest tsun/cldc aus allen Jahresdateien einer Meteostat-Station."""
    station_dir = bulk_dir / str(station_id)
    files = sorted(station_dir.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f, compression="gzip")
        cols = [c for c in ("tsun", "cldc") if c in df.columns]
        if not cols:
            continue
        df["timestamp"] = pd.to_datetime(df[["year", "month", "day"]]) + pd.to_timedelta(df["hour"], unit="h")
        dfs.append(df[["timestamp", *cols]])

    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["timestamp"] = df_all["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    return df_all


def daily_sunshine_proxy(station_id: str, bulk_dir: Path, cache_dir: Path) -> pd.Series:
    """
    Tägliche Sonnenscheindauer einer Station (Summe tsun), bei geringer tsun-Abdeckung
    100 - mittlere Wolkenbedeckung. Wird je Station gecacht.
    """
    cache_path = cache_dir / f"{station_id}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["timestamp"]).set_index("timestamp")["sun"]

    df_all = _read_bulk_hourly(station_id, bulk_dir)
    if df_all.empty:
        return pd.Series(dtype=float)

    tsun_cov = df_all["tsun"].notna().mean() if "tsun" in df_all.columns else 0.0
    if tsun_cov > TSUN_COVERAGE_THRESHOLD:
        daily = df_all.set_index("timestamp")["tsun"].resample("D").sum(min_count=10)
    elif "cldc" in df_all.columns:
        daily = 100 - df_all.set_index("timestamp")["cldc"].resample("D").mean()
    else:
        return pd.Series(dtype=float)

    daily.index = daily.index.tz_localize(None)
    daily.name = "sun"
    cache_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache_path)
    return daily.dropna()


def weekday_residual(series: pd.Series) -> pd.Series:
    """Entfernt den Wochentagseffekt (Wert minus Mittelwert des jeweiligen Wochentags)."""
    weekday = series.index.to_series().dt.weekday.values
    means = pd.Series(series.values, index=weekday).groupby(level=0).transform("mean")
    return series - means.values


def correlation_for_sensor(energy: pd.Series, sunshine: pd.Series) -> dict | None:
    """Korrelation zwischen täglicher Energie (roh und wochentagsbereinigt) und Sonnenschein im
    Sommerhalbjahr. None bei zu wenigen gemeinsamen Tagen."""
    joined = pd.DataFrame({"energy": energy, "sun": sunshine}).dropna()
    joined = joined[joined.index.month.isin(SUMMER_MONTHS)]
    if len(joined) < MIN_DAYS:
        return None

    energy_resid = weekday_residual(joined["energy"])
    r_raw = float(np.corrcoef(joined["energy"], joined["sun"])[0, 1])
    r_resid = float(np.corrcoef(energy_resid, joined["sun"])[0, 1])
    return {"n_days": len(joined), "r_raw": r_raw, "r_resid": r_resid}


def plot_correlation_distribution(df_corr: pd.DataFrame, correlation_threshold: float, dip_ratio_threshold: float, output_path: Path) -> None:
    """Streudiagramm r_resid vs. dip_ratio je location_category mit Schwellwerten."""
    categories = sorted(df_corr["location_category"].unique())
    fig, axes = plt.subplots(1, len(categories), figsize=(5 * len(categories), 4.5), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, category in zip(axes, categories):
        cell = df_corr[df_corr["location_category"] == category]
        colors = np.where(cell["flagged"], "crimson", "steelblue")
        ax.scatter(cell["r_resid"], cell["dip_ratio"], c=colors, alpha=0.5, s=12)
        ax.axvline(correlation_threshold, color="grey", linestyle="--", linewidth=1)
        ax.axhline(dip_ratio_threshold, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{category} (n={len(cell)}, geflaggt={int(cell['flagged'].sum())})", fontsize=10)
        ax.set_xlabel("r_resid (Energie vs. Sonnenschein)")

    axes[0].set_ylabel("dip_ratio (Mittag / Schultern, Sommer-Werktag)")
    fig.suptitle("PV-Erkennung: Korrelation x Profilform, rot = geflaggt (beide Kriterien)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def detect_pv_sensors(
    df_sensors: pd.DataFrame,
    load_hourly,
    output_ids_path: Path,
    output_plot_path: Path,
    sunshine_cache_dir: Path,
    bulk_weather_dir: Path,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    dip_ratio_threshold: float = DEFAULT_DIP_RATIO_THRESHOLD,
) -> set[str]:
    """
    Bestimmt die PV-verdächtigen Strom-Sensoren aus df_sensors (Index = sensor_id, mit Spalte
    'station_id') und schreibt die Liste nach output_ids_path.

    Returns: Menge der PV-verdächtigen sensor_ids.
    """
    df_strom = df_sensors[df_sensors["energy_type"] == "Strom"]

    rows = []
    for sensor_id, row in tqdm(df_strom.iterrows(), total=len(df_strom), desc="PV-Erkennung"):
        hourly = load_hourly(sensor_id)
        energy = daily_energy_from_series(hourly)
        sunshine = daily_sunshine_proxy(row["station_id"], bulk_weather_dir, sunshine_cache_dir)
        if energy.empty or sunshine.empty:
            continue

        result = correlation_for_sensor(energy, sunshine)
        if result is None:
            continue
        result["dip_ratio"] = summer_weekday_dip_ratio(hourly)
        rows.append({"sensor_id": sensor_id, "location_category": row["location_category"], **result})

    df_corr = pd.DataFrame(rows)
    df_corr["flagged"] = (df_corr["r_resid"] < correlation_threshold) & (df_corr["dip_ratio"] < dip_ratio_threshold)
    detected_ids = set(df_corr.loc[df_corr["flagged"], "sensor_id"])

    output_ids_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sensor_id": sorted(detected_ids)}).to_csv(output_ids_path, index=False)
    plot_correlation_distribution(df_corr, correlation_threshold, dip_ratio_threshold, output_plot_path)

    print(f"  {len(df_corr)} Strom-Sensoren ausgewertet, {len(detected_ids)} als PV-verdaechtig geflaggt")
    return detected_ids
