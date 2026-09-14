import numpy as np
import pandas as pd

"""
Zieht zufällige Messreihen-Fenster aus den stündlichen Spalten der sensor_table.parquet
(y_hourly/sin_hour/cos_hour/hourly_temp) als Eingabe des MeasurementSeriesPatchEncoder.
Die Sensordaten werden als numpy-Arrays gehalten.
"""


# Kanalgruppen (encoder.token_columns) und ihre feste Spaltenreihenfolge im Fenster:
# [Last, Temperatur, sin(Stunde), cos(Stunde)].
_TOKEN_COLUMN_ORDER = ["hourly_load", "hourly_weather", "hourly_time"]
_TOKEN_COLUMN_WIDTH = {"hourly_load": 1, "hourly_weather": 1, "hourly_time": 2}


def n_series_features(token_columns: list[str] | set[str] | None) -> int:
    """Anzahl Spalten im Encoder-Fenster für die gewählten token_columns (None = alle Kanäle)."""
    if token_columns is None:
        token_columns = _TOKEN_COLUMN_ORDER
    return sum(_TOKEN_COLUMN_WIDTH[name] for name in _TOKEN_COLUMN_ORDER if name in token_columns)


def build_sensor_arrays(sensor_days: pd.DataFrame) -> dict:
    """
    Wandelt die sensor_table-Zeilen eines Sensors in numpy-Arrays um. Tage mit fehlenden
    stündlichen Temperaturwerten werden ausgeschlossen.
    """
    has_complete_temp = np.array([not np.isnan(t).any() for t in sensor_days["hourly_temp"]])
    valid = sensor_days[has_complete_temp].sort_index()

    ordinals = np.array([d.toordinal() for d in valid.index], dtype=np.int64)
    y_hourly = np.array(valid["y_hourly"].tolist(), dtype=np.float32).reshape(len(valid), -1)
    hourly_temp = np.array(valid["hourly_temp"].tolist(), dtype=np.float32).reshape(len(valid), -1)
    sin_hour = np.array(valid["sin_hour"].tolist(), dtype=np.float32).reshape(len(valid), -1)
    cos_hour = np.array(valid["cos_hour"].tolist(), dtype=np.float32).reshape(len(valid), -1)
    category = np.asarray(sensor_days["category"].iloc[0], dtype=np.float32)

    return {
        "ordinals": ordinals,
        "y_hourly": y_hourly,
        "hourly_temp": hourly_temp,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "category": category,
        "runs": _contiguous_runs(ordinals),
    }


def _contiguous_runs(ordinals: np.ndarray) -> list[tuple[int, int]]:
    """
    Zerlegt sortierte Tages-Ordinalzahlen in zusammenhängende Abschnitte.
    Gibt (start_idx, length) als Indizes in die Arrays zurück.
    """
    runs = []
    n = len(ordinals)
    start = 0
    for i in range(1, n + 1):
        if i == n or ordinals[i] - ordinals[i - 1] != 1:
            runs.append((start, i - start))
            start = i
    return runs


def _exclude_ordinal_from_runs(runs: list[tuple[int, int]], ordinals: np.ndarray, target_ordinal: int) -> list[tuple[int, int]]:
    """Entfernt target_ordinal aus den Abschnitten und teilt den betroffenen Abschnitt bei Bedarf."""
    result = []
    for start, length in runs:
        end = start + length  # exklusiv
        if not (ordinals[start] <= target_ordinal <= ordinals[end - 1]):
            result.append((start, length))
            continue
        idx = start + (target_ordinal - ordinals[start])
        if idx - start > 0:
            result.append((start, idx - start))
        if end - (idx + 1) > 0:
            result.append((idx + 1, end - (idx + 1)))
    return result


def _restrict_runs_to_ordinal_range(runs: list[tuple[int, int]], ordinals: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    """Beschneidet jeden Abschnitt auf das Ordinalzahl-Intervall [lo, hi]."""
    result = []
    for start, length in runs:
        run_lo, run_hi = ordinals[start], ordinals[start] + length - 1
        new_lo, new_hi = max(run_lo, lo), min(run_hi, hi)
        if new_lo > new_hi:
            continue
        result.append((start + (new_lo - run_lo), new_hi - new_lo + 1))
    return result


def _pick_window(runs: list[tuple[int, int]], min_days: int, max_days: int, rng: np.random.Generator) -> tuple[int, int]:
    """
    Wählt einen Abschnitt gewichtet nach Länge und zieht darin Fensterlänge (min_days..max_days)
    und Startposition.
    Returns: (start_idx, window_len).
    """
    weights = np.array([length for _, length in runs], dtype=float)
    start, length = runs[rng.choice(len(runs), p=weights / weights.sum())]

    window_len = int(rng.integers(min_days, min(max_days, length) + 1))
    offset = int(rng.integers(0, length - window_len + 1))
    return start + offset, window_len


def _extract_window(sensor_arrays: dict, idx: int, window_len: int, token_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    columns_by_group = {
        "hourly_load": [sensor_arrays["y_hourly"][idx: idx + window_len].reshape(-1)],
        "hourly_weather": [sensor_arrays["hourly_temp"][idx: idx + window_len].reshape(-1)],
        "hourly_time": [
            sensor_arrays["sin_hour"][idx: idx + window_len].reshape(-1),
            sensor_arrays["cos_hour"][idx: idx + window_len].reshape(-1),
        ],
    }
    columns = [col for name in _TOKEN_COLUMN_ORDER if name in token_columns for col in columns_by_group[name]]

    series = np.column_stack(columns)
    static = sensor_arrays["category"]

    return series, static


def sample_measurement_window(
    sensor_arrays: dict,
    target_day: pd.Timestamp,
    min_days: int = 1,
    max_days: int = 14,
    rng: np.random.Generator | None = None,
    token_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Zieht ein zusammenhängendes Fenster von min_days bis max_days Tagen aus dem Kalenderjahr des
    Zieltags, ohne den Zieltag selbst. Länge und Position werden bei jedem Aufruf neu gezogen.
    Das Fenster stammt aus demselben Jahr, da es mit der Jahresmittellast des Zieltags normiert
    wird.

    Args:
        sensor_arrays: Ausgabe von build_sensor_arrays().
        target_day: Zieltag, wird aus den Kandidaten ausgeschlossen.
        min_days/max_days: Fensterlänge in Tagen.
        rng: numpy Random Generator.
        token_columns: aktive Kanalgruppen ("hourly_load"/"hourly_weather"/"hourly_time"),
            None = alle.

    Returns:
        (series, static) oder None, falls kein Fenster >= min_days im Zieljahr verfügbar ist.
        series: [window_hours, n_series_features(token_columns)], Spalten in der Reihenfolge
            [Last, Temperatur, sin(Stunde), cos(Stunde)]. Die Last ist unskaliert (siehe scale_window).
        static: [n_static_features] One-Hot-Category.
    """
    if rng is None:
        rng = np.random.default_rng()
    if token_columns is None:
        token_columns = _TOKEN_COLUMN_ORDER

    target_ordinal = target_day.toordinal()
    year = target_day.year
    runs = _restrict_runs_to_ordinal_range(
        sensor_arrays["runs"], sensor_arrays["ordinals"],
        pd.Timestamp(year=year, month=1, day=1).toordinal(),
        pd.Timestamp(year=year, month=12, day=31).toordinal(),
    )
    runs = _exclude_ordinal_from_runs(runs, sensor_arrays["ordinals"], target_ordinal)
    runs = [(s, length) for s, length in runs if length >= min_days]
    if not runs:
        return None

    idx, window_len = _pick_window(runs, min_days, max_days, rng)
    return _extract_window(sensor_arrays, idx, window_len, token_columns)


def scale_window(series: np.ndarray, scale: float, token_columns: list[str] | None = None) -> np.ndarray:
    """
    Teilt den Lastkanal des Fensters durch die Jahresmittellast des Zieltags (y_scale).
    Die übrigen Kanäle bleiben unverändert.

    Args:
        series: [window_hours, n_series_features(token_columns)], Last unskaliert.
        scale: Jahresmittellast des Zieltags.
        token_columns: Kanalauswahl wie bei sample_measurement_window (None = alle).
    Returns:
        Skaliertes Fenster gleicher Form.
    """
    if token_columns is not None and "hourly_load" not in token_columns:
        return series.copy()

    scaled = series.copy()
    scaled[:, 0] = series[:, 0] / scale if scale > 0 else 0.0
    return scaled
