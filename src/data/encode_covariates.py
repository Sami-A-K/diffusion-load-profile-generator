"""
Encoding-Funktionen für die Kovariaten: zyklische Zeit, Feiertage, energy_type, category,
cluster, prev_h23 und Wetter. Die Funktionen arbeiten jeweils auf den Tagen eines Sensors.
"""

import numpy as np
import pandas as pd
import holidays

from src.data.lookup_tables import CATEGORIES, LOCATION_USAGE_FIXES, LOCATION_USAGE_TO_CATEGORY, LOCATION_USAGE_DETAIL_TO_CATEGORY, FEDERAL_STATE_CODES


def encode_energy_type(days: pd.DatetimeIndex, energy_type: str) -> np.ndarray:
    """
    Encodes energy_type as 0 (Strom) / 1 (Waerme), broadcast über alle Tage.
    """
    if energy_type == 'Strom':
        encoded_energy_type = 0
    elif energy_type == 'Waerme':
        encoded_energy_type = 1
    else:
        raise ValueError(f"Unbekannter energy_type: {energy_type}")

    return np.full(len(days), encoded_energy_type)


def resolve_category(location_usage: str, location_usage_detail: str | None = None) -> str:
    """
    Ordnet location_usage einer der zehn Kategorien zu. Ist location_usage_detail gesetzt und
    bekannt, hat es Vorrang (z.B. Sporthalle, die unter "Gymnasium" geführt wird).
    """
    if location_usage_detail is not None and pd.notna(location_usage_detail) and location_usage_detail in LOCATION_USAGE_DETAIL_TO_CATEGORY:
        return LOCATION_USAGE_DETAIL_TO_CATEGORY[location_usage_detail]
    fixed = LOCATION_USAGE_FIXES.get(location_usage, location_usage)
    return LOCATION_USAGE_TO_CATEGORY.get(fixed, "Sonstige")


def encode_category(days: pd.DatetimeIndex, location_usage: str, location_usage_detail: str | None = None) -> np.ndarray:
    """
    Zuordnung von der location_usage (ggf. verfeinert durch location_usage_detail) aus den
    Metadaten zu einer Kategorie. One-hot-Encoding der Kategorie.
    """
    cleaned = resolve_category(location_usage, location_usage_detail)
    one_hot_row = np.zeros(len(CATEGORIES))
    one_hot_row[CATEGORIES.index(cleaned)] = 1
    return np.tile(one_hot_row, (len(days), 1))


def decode_category(one_hot: np.ndarray) -> str:
    """Kehrt encode_category um: one-hot-Vektor -> Kategorie."""
    return CATEGORIES[int(np.argmax(one_hot))]


def encode_cluster(days: pd.DatetimeIndex, sensor_id: str, df_cluster: pd.DataFrame, n_clusters: int) -> np.ndarray:
    """
    One-hot-Encoding des global eindeutigen Lastform-Clusters je Tag (aus build_clusters.py).

    df_cluster: MultiIndex (sensor_id, year) -> Spalte 'cluster'. Tage ohne Eintrag ergeben
    NaN-Zeilen.
    """
    one_hot = np.full((len(days), n_clusters), np.nan)
    for i, year in enumerate(days.year):
        if (sensor_id, year) not in df_cluster.index:
            continue
        one_hot[i] = 0
        one_hot[i, df_cluster.loc[(sensor_id, year), "cluster"]] = 1
    return one_hot


def encode_prev_h23(days: pd.DatetimeIndex, y_hourly: np.ndarray, y_scale: np.ndarray) -> np.ndarray:
    """
    Last in Stunde 23 des Vortags, normiert mit der Jahresmittellast des aktuellen Tages.
    NaN, wenn der Vortag nicht in days enthalten ist.

    Args:
        days: sortierter DatetimeIndex der validen Tage eines Sensors.
        y_hourly: [len(days), 24] Stundenwerte in kWh.
        y_scale: [len(days)] Jahresmittellast je Zeile.
    """
    y_scale = np.asarray(y_scale, dtype=float)
    scale = np.where(y_scale > 0, y_scale, 1.0)
    raw_h23 = y_hourly[:, 23]

    ordinals = np.array([d.toordinal() for d in days])
    is_consecutive = np.zeros(len(days), dtype=bool)
    is_consecutive[1:] = (ordinals[1:] - ordinals[:-1]) == 1
    cur_idx = np.where(is_consecutive)[0]
    prev_idx = cur_idx - 1

    prev_h23 = np.full(len(days), np.nan)
    prev_h23[cur_idx] = raw_h23[prev_idx] / scale[cur_idx]
    return prev_h23


def encode_cyclical_time(days: pd.DatetimeIndex) -> np.ndarray:
    """
    Zyklische sin/cos-Encodings für Wochentag und Tag-im-Jahr, pro Tag.
    """
    day_of_week = days.day_of_week
    sin_day_of_week = np.sin(2 * np.pi * day_of_week / 7)
    cos_day_of_week = np.cos(2 * np.pi * day_of_week / 7)

    day_of_year = days.day_of_year
    sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365)
    cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365)

    return np.column_stack([sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year])


def encode_cyclical_hour(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Zyklisches sin/cos-Encoding der Stunde-des-Tages, pro Stunde. Im MeasurementSeriesEncoder verwendet
    """
    hour = timestamps.hour
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)

    return np.column_stack([sin_hour, cos_hour])


def encode_holidays(days: pd.DatetimeIndex, federal_state: str) -> np.ndarray:
    """
    Encoding to differentiate working from non working days i.e. holidays and sundays.
    Gets federal holidays in Germany from holidays library.
    Args:
        days: pandas DatetimeIndex
        federal_state: String with Federal-State
    Returns:
        holiday_list: Numpy Array, where the index value equals 1 if it is a non working day, else 0.
    """
    state_code = FEDERAL_STATE_CODES.get(federal_state)
    years_in_data = {d.year for d in days}
    fed_holidays = holidays.country_holidays("DE", subdiv=state_code, years=years_in_data)
    holiday_list = np.array([1 if (d.dayofweek == 6 or d.date() in fed_holidays) else 0 for d in days])

    return holiday_list


def encode_daily_temps(days: pd.DatetimeIndex, df_weather: pd.DataFrame) -> np.ndarray:
    """
    Liefert tmin/temp/tmax (Meteostat, Tagesauflösung) für die gegebenen Tage.
    Tage ohne Wetterdaten ergeben NaN-Zeilen.
    """
    aligned = df_weather.reindex(days.normalize())
    return aligned[["tmin", "temp", "tmax"]].values


def encode_hourly_temps(hours: pd.DatetimeIndex, df_weather_hourly: pd.Series) -> np.ndarray:
    """
    Liefert die stündliche Temperatur (Meteostat) für die gegebenen Stunden, verwendet im
    Messfenster des Encoders. Stunden ohne Wetterdaten ergeben NaN.
    """
    aligned = df_weather_hourly.reindex(hours)
    return aligned.to_numpy().reshape(-1, 1)
