"""
Rohe OpenMeter-Zeitreihen -> bereinigte stündliche kWh-Reihen je Sensor (Stufe 2 und 4 der
Pipeline).
- conversion: Rohwerte (Zählerstand/Differenzmengenwert/Differenzmittelwert/kW, 15min/1h) in eine
  stündliche kWh-Reihe mit naivem Index in lokaler Zeit umwandeln (24 Werte je Kalendertag).
- filter: Werte-Filter (Ausreißer, hängende Sensoren, Lücken) und Metadatenfilter.
"""

from pathlib import Path

import pandas as pd

# Zeitstempel vor 1995 oder in der Zukunft gelten als fehlerhaft.
MIN_PLAUSIBLE_TIMESTAMP = pd.Timestamp("1995-01-01", tz="Europe/Berlin")


# --- conversion -------------------------------------------------------------------------------

def regularize_dst_days(hourly: pd.Series) -> pd.Series:
    """
    Überführt eine tz-aware Europe/Berlin-Reihe in ein naives Stundenraster mit 24 Werten je Tag.

    Am Umstellungstag im Herbst werden die beiden Werte der doppelten Stunde gemittelt; im
    Frühjahr entsteht für die fehlende Stunde ein NaN, das fill_short_gaps interpoliert. Die
    lokale Zeit bleibt erhalten, damit die Stunde-des-Tages-Kovariate dem Nutzungsverhalten folgt.
    """
    naive = hourly.index.tz_localize(None)
    hourly = hourly.groupby(naive).mean()

    full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_range)
    hourly.index.name = "timestamp"
    return hourly


def load_raw_timeseries(path: Path) -> pd.Series:
    """
    Liest eine rohe Zeitreihen-CSV (timestamps,values; Zeitstempel bereits mit Offset).
    Index wird nach Europe/Berlin konvertiert (DST-aware), Duplikate entfernt,
    unplausible Zeitstempel (vor 1995 oder in der Zukunft) verworfen.
    """
    df = pd.read_csv(path, parse_dates=["timestamps"])
    series = df.set_index("timestamps")["values"]
    series.index = pd.to_datetime(series.index, utc=True).tz_convert("Europe/Berlin")
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series.index.name = "timestamp"
    series.name = "values"

    now = pd.Timestamp.now(tz="Europe/Berlin")
    series = series[(series.index >= MIN_PLAUSIBLE_TIMESTAMP) & (series.index <= now)]
    return series


def unify_to_hourly(raw: pd.Series, unit: str, value_type: str, frequency: str) -> pd.Series:
    """
    Vereinheitlicht eine Rohreihe zu stündlichem kWh-Verbrauch und wendet die Werte-Filter an.

    Args:
        raw: Rohreihe (Index = Europe/Berlin Timestamps).
        unit: measurement_unit aus den Metadaten (z.B. 'kWh', 'kW', 'MWh').
        value_type: measurement_value_type ('Zaehlerstand', 'Differenzmengenwert', 'Differenzmittelwert').
        frequency: measurement_frequency der Rohdaten ('15min' oder '1h').
    """
    if frequency not in {"15min", "1h"}:
        raise ValueError(f"frequency '{frequency}' nicht unterstützt (nur 15min, 1h).")

    if unit in {"kWh", "MWh", "kWh(Hs)"}:
        if value_type == "Zaehlerstand":
            # min_count=1, da manche Sensoren zwischen 15min- und 1h-Abtastung wechseln
            hourly = raw.diff().resample("h").sum(min_count=1)
        elif value_type == "Differenzmengenwert":
            hourly = raw.resample("h").sum()
        elif value_type == "Differenzmittelwert":
            hourly = raw.resample("h").mean()
        else:
            raise ValueError(f"Unbekannter value_type: {value_type}")

        if unit == "MWh":
            hourly = hourly * 1000
    else:  # kW (Leistungsdaten): Durchschnittsleistung über die Stunde ≈ kWh in der Stunde
        hourly = raw.resample("h").mean()

    hourly = regularize_dst_days(hourly)
    hourly = filter_bad_values(hourly)
    hourly = filter_rolling_range(hourly)
    hourly = fill_short_gaps(hourly)
    return hourly


# --- filter -------------------------------------------------------------------------------------

def filter_bad_values(series: pd.Series, iqr_factor: float = 5.0) -> pd.Series:
    """
    Setzt negative Werte und IQR-Ausreißer auf NaN. Der Faktor liegt über dem üblichen 1.5,
    da Lastspitzen legitime Werte sind.
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    mask_bad = (series < 0) | (series < q1 - iqr_factor * iqr) | (series > q3 + iqr_factor * iqr)
    return series.mask(mask_bad)


def filter_rolling_range(series: pd.Series, window: int = 8, rel_threshold: float = 1e-6) -> pd.Series:
    """
    Setzt Werte in 8h-Fenstern ohne Änderung (hängender Sensor) auf NaN. Kriterium ist die
    Spannweite relativ zum Fenstermittel.
    """
    half_window = window // 2
    rolling_range = series.rolling(window=window, center=True, min_periods=window).apply(
        lambda w: w.max() - w.min(), raw=True
    )
    rolling_mean = series.rolling(window=window, center=True, min_periods=window).mean()
    mask_bad = (rolling_range / rolling_mean.abs()) < rel_threshold
    mask_bad = mask_bad.rolling(window=2 * half_window + 1, center=True, min_periods=1).max().astype(bool)
    return series.mask(mask_bad)


def fill_short_gaps(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """
    Interpoliert linear über Lücken bis max_gap Stunden; längere Lücken bleiben vollständig NaN.
    """
    is_nan = series.isna()
    gap_id = (is_nan != is_nan.shift()).cumsum()
    gap_size = is_nan.groupby(gap_id).transform("sum")
    fillable = is_nan & (gap_size <= max_gap)

    interpolated = series.interpolate(method="linear", limit_direction="both")
    return series.where(~fillable, interpolated)


def drop_redundant_wirkleistung(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Wirkleistung-Sensoren an (location_id, energy_type)-Kombinationen, für die bereits
    ein Wirkarbeit-Sensor vorliegt.
    """
    has_arbeit = set(
        map(tuple, df_meta.loc[df_meta["measurement_type"] == "Wirkarbeit", ["location_id", "energy_type"]].to_numpy())
    )
    is_redundant = (df_meta["measurement_type"] == "Wirkleistung") & pd.Series(
        list(zip(df_meta["location_id"], df_meta["energy_type"])), index=df_meta.index
    ).isin(has_arbeit)
    return df_meta[~is_redundant]


def filter_relevant_sensors(df_meta: pd.DataFrame, energy_types: set[str] = {"Strom", "Waerme", "Fernwaerme"}) -> pd.DataFrame:
    """
    Filtert die Metadaten auf Verbrauchssensoren der gewählten energy_types mit Wirkarbeit oder
    Wirkleistung (siehe drop_redundant_wirkleistung). Fernwärme wird als "Waerme" geführt.
    """
    df_meta = df_meta[
        df_meta["energy_type"].isin(energy_types)
        & (df_meta["measurement_category"] == "Verbrauch")
        & df_meta["measurement_type"].isin({"Wirkarbeit", "Wirkleistung"})
    ].copy()
    df_meta["energy_type"] = df_meta["energy_type"].replace({"Fernwaerme": "Waerme"})
    return drop_redundant_wirkleistung(df_meta)
