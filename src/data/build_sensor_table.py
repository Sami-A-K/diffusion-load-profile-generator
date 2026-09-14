"""
baut aus Stundenreihen, Wetterdaten und Cluster-Zuordnung die
sensor_table.parquet mit einer Zeile je Sensor und validem Tag (MultiIndex sensor_id/day).

Spalten:
  - Ziel: y_hourly (24 Stundenwerte in kWh) und y_scale (Jahresmittellast des Kalenderjahres).
    Die Normierung y_hourly / y_scale erfolgt in build_training_data.build_y.
  - Kovariaten des Decoders: cyclical_time, is_holiday, energy_type, category, cluster,
    daily_weather, prev_h23.
  - Eingangsgrößen des Encoder-Messfensters: sin_hour, cos_hour, hourly_temp, y_hourly.

Normiert wird auf die Jahresmittellast, da diese beim Einsatz aus dem vorgegebenen
Jahresenergiebedarf folgt (E_a / Stundenzahl). Die Temperaturen sind mit festen Grenzen
(lookup_tables.WEATHER_TEMP_MIN_C/MAX_C) skaliert.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.encode_covariates import encode_category, encode_cluster, encode_cyclical_hour, encode_cyclical_time, encode_daily_temps, encode_energy_type, encode_holidays, encode_hourly_temps, encode_prev_h23
from src.data.lookup_tables import WEATHER_TEMP_MIN_C, WEATHER_TEMP_MAX_C
from src.data.get_weather_data import load_hourly_weather_for_station, load_weather_for_station


def load_hourly_series(sensor_id: str, hourly_dir: Path) -> pd.Series:
    """Lädt die stündliche kWh-Reihe eines Sensors (naiver Index in lokaler Zeit)."""
    df = pd.read_csv(hourly_dir / f"{sensor_id}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")["values"]


def get_valid_days(df_hourly: pd.Series, df_weather: pd.DataFrame) -> pd.DatetimeIndex:
    """Tage mit 24 vollständigen Stundenwerten und vorhandenen Wetterdaten."""
    dates = df_hourly.index.normalize()
    size = df_hourly.groupby(dates).size()
    n_valid = df_hourly.groupby(dates).count()
    complete = (size == 24) & (n_valid == size)
    complete_dates = complete[complete].index

    weather_dates = pd.DatetimeIndex(df_weather.index).normalize()
    valid = complete_dates[complete_dates.isin(weather_dates)]
    return pd.DatetimeIndex(sorted(valid))


MIN_DAYS_FOR_YEAR_SCALE = 30  # Jahre mit weniger validen Tagen erhalten das Mittel über alle Jahre des Sensors


def year_mean_load(valid_days: pd.DatetimeIndex, y_hourly_arr: np.ndarray) -> np.ndarray:
    """
    Mittlere Last je Kalenderjahr (Spalte y_scale), gemittelt über die validen Tage.

    Jahre mit weniger als MIN_DAYS_FOR_YEAR_SCALE validen Tagen erhalten das Mittel über alle
    Jahre des Sensors.

    Returns: [len(valid_days)] - ein y_scale-Wert je Zeile.
    """
    daily_mean = y_hourly_arr.mean(axis=1)
    years = valid_days.year
    sensor_wide_mean = float(daily_mean.mean())

    daily_mean_by_year = pd.Series(daily_mean, index=years)
    year_mean = daily_mean_by_year.groupby(level=0).transform("mean")
    year_count = daily_mean_by_year.groupby(level=0).transform("count")
    year_mean = year_mean.where(year_count >= MIN_DAYS_FOR_YEAR_SCALE, sensor_wide_mean)
    return year_mean.to_numpy()


# Ausschlusskriterien für Sensor-Jahre ohne auswertbares Lastprofil.
DEAD_YEAR_MAX_ZERO_FRACTION = 0.5   # Anteil der Stunden mit exakt 0
MIN_MEAN_ANNUAL_LOAD = 0.05         # kWh/h, entspricht 50 W mittlerer Last


def implausible_years(valid_days: pd.DatetimeIndex, y_hourly_arr: np.ndarray, y_scale: np.ndarray) -> np.ndarray:
    """
    Maske [len(valid_days)]: True für Zeilen, deren Sensor-Jahr über 50 % Nullstunden oder eine
    mittlere Last unter 50 W hat. Bei solchen Jahren ist y_scale sehr klein und die normierten
    Werte werden extrem groß. Gefiltert wird je Sensor-Jahr, damit brauchbare Jahre desselben
    Sensors erhalten bleiben.
    """
    years = valid_days.year
    zero_hours = pd.Series((y_hourly_arr <= 1e-9).sum(axis=1), index=years)
    zero_fraction = zero_hours.groupby(level=0).transform("sum") / (zero_hours.groupby(level=0).transform("count") * y_hourly_arr.shape[1])
    bad = (zero_fraction.to_numpy() >= DEAD_YEAR_MAX_ZERO_FRACTION) | (y_scale < MIN_MEAN_ANNUAL_LOAD)
    return bad


def build_sensor_features(sensor_id: str, df_hourly: pd.Series, df_weather: pd.DataFrame, df_weather_hourly: pd.Series, metadata_row: pd.Series, df_cluster: pd.DataFrame, n_clusters: int) -> pd.DataFrame | None:
    """
    Tagestabelle eines Sensors: eine Zeile je validem Tag, mehrwertige Features als Listen.

    prev_h23 ist NaN, wenn der Vortag kein valider Tag ist. Stundenfeatures werden für alle
    validen Tage gemeinsam berechnet und zu (n_days, 24) umgeformt.
    """
    valid_days = get_valid_days(df_hourly, df_weather)
    if len(valid_days) == 0:
        return None

    n_days = len(valid_days)
    valid_hours_mask = df_hourly.index.normalize().isin(valid_days)
    df_hourly_valid = df_hourly[valid_hours_mask].sort_index()
    hours_index = df_hourly_valid.index

    y_hourly_arr = df_hourly_valid.to_numpy().reshape(n_days, 24)
    y_scale = year_mean_load(valid_days, y_hourly_arr)

    keep = ~implausible_years(valid_days, y_hourly_arr, y_scale)
    if not keep.any():
        return None

    cyclical_time = encode_cyclical_time(valid_days)
    is_holiday = encode_holidays(valid_days, metadata_row["location_federal_state"])
    energy_type = encode_energy_type(valid_days, metadata_row["energy_type"])
    category = encode_category(valid_days, metadata_row["location_usage"], metadata_row["location_usage_detail"])
    cluster = encode_cluster(valid_days, sensor_id, df_cluster, n_clusters)
    daily_weather = (encode_daily_temps(valid_days, df_weather) - WEATHER_TEMP_MIN_C) / (WEATHER_TEMP_MAX_C - WEATHER_TEMP_MIN_C)
    prev_h23 = encode_prev_h23(valid_days, y_hourly_arr, y_scale)
    hourly_temp_arr = (encode_hourly_temps(hours_index, df_weather_hourly) - WEATHER_TEMP_MIN_C) / (WEATHER_TEMP_MAX_C - WEATHER_TEMP_MIN_C)
    hourly_temp = hourly_temp_arr.reshape(n_days, 24)
    cyclical_hour = encode_cyclical_hour(hours_index)
    sin_hour = cyclical_hour[:, 0].reshape(n_days, 24)
    cos_hour = cyclical_hour[:, 1].reshape(n_days, 24)

    index = pd.MultiIndex.from_arrays(
        [[sensor_id] * n_days, valid_days], names=["sensor_id", "day"]
    )

    sensor_features = pd.DataFrame(
        {
            "cyclical_time": cyclical_time.tolist(),
            "is_holiday": is_holiday,
            "energy_type": energy_type,
            "category": category.tolist(),
            "cluster": cluster.tolist(),
            "prev_h23": prev_h23,
            "daily_weather": daily_weather.tolist(),
            "sin_hour": sin_hour.tolist(),
            "cos_hour": cos_hour.tolist(),
            "hourly_temp": hourly_temp.tolist(),
            "y_hourly": y_hourly_arr.tolist(),
            "y_scale": y_scale,
        }, index=index)

    return sensor_features.loc[keep]


def build_sensor_table(df_sensors: pd.DataFrame, hourly_dir: Path, df_cluster: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Baut die Tagestabellen aller Sensoren aus df_sensors (Index = sensor_id, mit Spalte
    'station_id') und schreibt sie als gemeinsame Parquet-Datei.

    df_cluster: MultiIndex (sensor_id, year) mit Spalte 'cluster'.
    """
    n_clusters = int(df_cluster["cluster"].max()) + 1

    station_cache: dict[str, pd.DataFrame] = {}
    station_hourly_cache: dict[str, pd.Series] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_tables: list[pd.DataFrame] = []
    n_skipped = 0

    for sensor_id, row in tqdm(df_sensors.iterrows(), total=len(df_sensors), desc="Sensor-Tabelle"):
        station_id = row["station_id"]
        if station_id not in station_cache:
            station_cache[station_id] = load_weather_for_station(station_id)
            station_hourly_cache[station_id] = load_hourly_weather_for_station(station_id)

        df_hourly = load_hourly_series(sensor_id, hourly_dir)
        df_sensor = build_sensor_features(
            sensor_id, df_hourly, station_cache[station_id], station_hourly_cache[station_id],
            row, df_cluster, n_clusters,
        )

        if df_sensor is None or df_sensor.empty:
            n_skipped += 1
            continue
        sensor_tables.append(df_sensor)

    sensor_table = pd.concat(sensor_tables)
    sensor_table.to_parquet(output_path)

    print(f"  {len(sensor_table)} Sensor-Tage über {len(sensor_tables)} Sensoren ({n_skipped} ohne valide Tage übersprungen)")
    print(f"  Geschrieben nach: {output_path}")
    return sensor_table
