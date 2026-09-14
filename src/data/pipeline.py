"""
Datenpipeline: von den Rohdaten unter data/raw/ zur sensor_table.parquet.

    python -m src.data.pipeline

Eine Sensorliste wird durch die Stufen gereicht; jede Stufe filtert sie und protokolliert den Abgang.

Stufen:
  1. Metadaten laden
  2. Metadatenfilter    - energy_type (inkl. Fernwärme), Verbrauch, Wirkarbeit/Wirkleistung,
                          Messfrequenz, Land
  3. Wetterstation      - Zuordnung Sensor -> Meteostat-Station
  4. Stundenreihen      - Rohreihen -> stündliche kWh
  5. PV-Erkennung       - Strom-Sensoren mit PV-Eigenverbrauchssignatur entfernen
  6. Clustering         - Lastform-Cluster auf der finalen Population
  7. Sensor-Tabelle     - sensor_table.parquet

Die PV-Erkennung steht am Ende, da sie selbst Stundenreihen und Stationszuordnung benötigt.
Die Stationszuordnung liest den Messzeitraum aus der Rohzeitreihe und kann daher vor der
Stundenaufbereitung laufen. Das Clustering folgt nach der PV-Erkennung, damit PV-verzerrte
Profile die Zentroide nicht beeinflussen.
"""

import json
from pathlib import Path

import pandas as pd

from src.data.build_clusters import build_clusters
from src.data.build_sensor_table import build_sensor_table, load_hourly_series
from src.data.detect_pv_sensors import detect_pv_sensors
from src.data.get_weather_data import map_sensors_to_stations
from src.data.process_load_data import filter_relevant_sensors, load_raw_timeseries, unify_to_hourly

RAW_DIR = Path("data/raw/openmeter")
PROCESSED_DIR = Path("data/processed/openmeter")
TIMESERIES_DIR = RAW_DIR / "timeseries"
HOURLY_DIR = PROCESSED_DIR / "hourly"
METADATA_PATH = RAW_DIR / "meta_data/meta_data_all_sensors.csv"

# Version der Stundenreihen-Aufbereitung; bei Änderungen an Konvertierung oder Filtern erhöhen,
# damit Stufe 4 vorhandene Dateien neu berechnet.
HOURLY_FORMAT_VERSION = 2
HOURLY_MANIFEST_PATH = HOURLY_DIR / "_manifest.json"

SUPPORTED_FREQUENCIES = {"15min", "1h"}
FREQUENCY_ALIASES = {"15m": "15min"}  # abweichende Schreibweise in den Rohmetadaten


class Attrition:
    """Protokolliert je Stufe die Anzahl der Sensoren vor und nach dem Filter."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, stage: str, n_before: int, n_after: int, reason: str) -> None:
        self.rows.append({"stufe": stage, "vorher": n_before, "nachher": n_after, "abgang": n_before - n_after, "grund": reason})
        print(f"[{stage}] {n_before} -> {n_after}  (-{n_before - n_after}: {reason})")

    def to_csv(self, path: Path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"\nAbgangs-Report: {path}")


def measures_range_from_raw(sensor_id: str, timeseries_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Messzeitraum (erster/letzter Zeitstempel) aus der Rohzeitreihe, falls er in den Metadaten fehlt."""
    series = load_raw_timeseries(timeseries_dir / f"{sensor_id}.csv")
    if series.empty:
        return pd.NaT, pd.NaT
    return series.index[0], series.index[-1]


def stage_metadata_filter(attrition: Attrition) -> pd.DataFrame:
    """Stufe 1+2: Metadaten laden und auf die verwendbaren Sensoren filtern."""
    df = pd.read_csv(METADATA_PATH)
    n_start = len(df)

    df = filter_relevant_sensors(df)
    attrition.record("Metadatenfilter", n_start, len(df), "energy_type/Verbrauch/Wirkarbeit+Wirkleistung(dedupliziert)")

    n_before = len(df)
    df["measurement_frequency"] = df["measurement_frequency"].replace(FREQUENCY_ALIASES)
    df = df[df["measurement_frequency"].isin(SUPPORTED_FREQUENCIES)]
    attrition.record("Messfrequenz", n_before, len(df), f"nicht in {sorted(SUPPORTED_FREQUENCIES)}")

    n_before = len(df)
    df = df[df["location_country"] == "Deutschland"]
    attrition.record("Land", n_before, len(df), "ausserhalb Deutschlands (Feiertags-/Wetterlogik ist DE-spezifisch)")

    n_before = len(df)
    df = df[df["id"].map(lambda sid: (TIMESERIES_DIR / f"{sid}.csv").exists())]
    attrition.record("Rohdaten", n_before, len(df), "keine Rohzeitreihe heruntergeladen")

    return df.set_index("id")


def stage_weather(df_sensors: pd.DataFrame, attrition: Attrition) -> pd.DataFrame:
    """Stufe 3: Wetterstation je Sensor bestimmen; Sensoren ohne passende Station entfallen."""
    n_before = len(df_sensors)
    mapping = map_sensors_to_stations(df_sensors, measures_range_fallback=lambda sid: measures_range_from_raw(sid, TIMESERIES_DIR))

    df_sensors = df_sensors.join(mapping.rename("station_id"), how="inner")
    attrition.record("Wetterstation", n_before, len(df_sensors), "keine Meteostat-Station mit Zeitraumueberlappung")
    return df_sensors


def stage_hourly(df_sensors: pd.DataFrame, attrition: Attrition) -> pd.DataFrame:
    """
    Stufe 4: Rohreihen -> stündliche kWh-Reihen. Neu berechnet werden fehlende Dateien und
    Dateien mit älterer HOURLY_FORMAT_VERSION.
    """
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(HOURLY_MANIFEST_PATH.read_text()) if HOURLY_MANIFEST_PATH.exists() else {}

    n_before = len(df_sensors)
    valid_ids, n_reused, n_built = [], 0, 0

    for sensor_id, row in df_sensors.iterrows():
        out_path = HOURLY_DIR / f"{sensor_id}.csv"
        if out_path.exists() and manifest.get(sensor_id) == HOURLY_FORMAT_VERSION:
            valid_ids.append(sensor_id)
            n_reused += 1
            continue

        raw = load_raw_timeseries(TIMESERIES_DIR / f"{sensor_id}.csv")
        hourly = unify_to_hourly(
            raw, unit=row["measurement_unit"], value_type=row["measurement_value_type"],
            frequency=row["measurement_frequency"],
        )
        if hourly.dropna().empty:
            manifest.pop(sensor_id, None)
            out_path.unlink(missing_ok=True)
            continue

        hourly.to_frame("values").to_csv(out_path)
        manifest[sensor_id] = HOURLY_FORMAT_VERSION
        valid_ids.append(sensor_id)
        n_built += 1

    HOURLY_MANIFEST_PATH.write_text(json.dumps(manifest, indent=0, sort_keys=True))
    attrition.record("Stundenreihen", n_before, len(valid_ids), f"keine validen Werte nach Bereinigung (neu: {n_built}, wiederverwendet: {n_reused})")
    return df_sensors.loc[valid_ids]


def stage_pv(df_sensors: pd.DataFrame, attrition: Attrition) -> pd.DataFrame:
    """Stufe 5: Sensoren mit PV-Eigenverbrauch erkennen und ausschließen."""
    n_before = len(df_sensors)
    pv_ids = detect_pv_sensors(
        df_sensors=df_sensors,
        load_hourly=lambda sid: load_hourly_series(sid, HOURLY_DIR),
        output_ids_path=RAW_DIR / "meta_data/pv_affected_sensor_ids.csv",
        output_plot_path=PROCESSED_DIR / "pv_detection_plot.png",
        sunshine_cache_dir=PROCESSED_DIR / "pv_detection_sunshine_cache",
        bulk_weather_dir=Path("data/raw/meteostat/meteostat_bulk_hourly"),
    )
    df_sensors = df_sensors[~df_sensors.index.isin(pv_ids)]
    attrition.record("PV-Erkennung", n_before, len(df_sensors), "PV-Eigenverbrauchssignatur (Korrelation UND Mittagseinbruch)")
    return df_sensors


def run(sensor_table_path: Path = PROCESSED_DIR / "sensor_table.parquet") -> None:
    """Führt alle Stufen aus. sensor_table_path: Zielpfad der Sensor-Tabelle."""
    attrition = Attrition()

    df_sensors = stage_metadata_filter(attrition)
    df_sensors = stage_weather(df_sensors, attrition)
    df_sensors = stage_hourly(df_sensors, attrition)
    df_sensors = stage_pv(df_sensors, attrition)

    print("\n[Clustering]")
    df_cluster = build_clusters(
        df_sensors=df_sensors,
        load_hourly=lambda sid: load_hourly_series(sid, HOURLY_DIR),
        output_cluster_path=PROCESSED_DIR / "clusters_per_sensor_and_year.csv",
        output_proportions_path=PROCESSED_DIR / "cluster_proportions.csv",
        output_plot_path=PROCESSED_DIR / "clusters_plot.png",
    )

    print("\n[Sensor-Tabelle]")
    build_sensor_table(
        df_sensors=df_sensors,
        hourly_dir=HOURLY_DIR,
        df_cluster=df_cluster,
        output_path=sensor_table_path,
    )

    attrition.to_csv(PROCESSED_DIR / "pipeline_attrition.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=PROCESSED_DIR / "sensor_table.parquet",
        help="Zielpfad der Sensor-Tabelle (Stufe 7). Fuer einen Probe-Rebuild auf eine Seitendatei "
             "setzen und erst nach einem Vergleich gegen die produktive Datei uebernehmen.",
    )
    run(sensor_table_path=parser.parse_args().out)
