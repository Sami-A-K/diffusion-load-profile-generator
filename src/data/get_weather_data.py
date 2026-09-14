import pandas as pd
import numpy as np
import sqlite3
import requests
from pathlib import Path
from tqdm import tqdm
import json

METEOSTAT_BASE_URL = "https://data.meteostat.net/hourly"

def get_meteostat_stations(save_file: bool = False):
    """Load meteostat stations and save as CSV file.
    stations.db is available at: https://data.meteostat.net/stations.db
    """
    con = sqlite3.connect("data/raw/meteostat/meta_data/stations.db")

    # nur deutsche Stationen
    df_de = pd.read_sql_query(
            "SELECT * FROM stations WHERE country = 'DE';",
            con)

    df_time = pd.read_sql_query("""
    SELECT station, start, end, parameter
    FROM inventory
    """, con)

    con.close()
    df_time = df_time[df_time["parameter"].isin(["temp", "tmin", "tmax"])]

    agg = df_time.groupby("station").agg(
        start=("start", "max"),   # spätester Start aller drei
        end=("end", "min"),       # frühestes Ende aller drei
        n_params=("parameter", "nunique"),
    ).reset_index()

    # Nur Stationen mit allen drei Parametern und gemeinsamem Verfügbarkeitszeitraum
    agg["start"] = pd.to_datetime(agg["start"])
    agg["end"] = pd.to_datetime(agg["end"])
    agg = agg[(agg["n_params"] == 3) & (agg["start"] <= agg["end"])].drop(columns="n_params")

    df_all = df_de.merge(agg, left_on="id", right_on="station").drop(columns="station")

    if save_file:
        df_all.to_csv('data/raw/meteostat/meta_data/stations_de.csv', index=False)

    return df_all

def download_station_year(
    station_id: str,
    year: int,
    cache_dir: Path,
) -> bool:
    """
    Lädt eine Jahres-CSV einer Meteostat-Station ins lokale Cache-Verzeichnis.
    Returns True bei Erfolg, False sonst.
    """
    file = cache_dir / station_id / f"{year}.csv.gz"
    if file.exists():
        return True

    url = f"{METEOSTAT_BASE_URL}/{year}/{station_id}.csv.gz"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"  Download-Fehler {station_id}/{year}: {e}")
    return False


def download_with_fallback(
    primary_station: str,
    sensor_lat: float,
    sensor_lon: float,
    start_year: int,
    end_year: int,
    df_stations: pd.DataFrame,
    cache_dir: Path,
    max_fallbacks: int = 5,
) -> dict[int, str]:
    """
    Lädt für jedes Jahr im Bereich die Daten für die primäre Station.
    Bei Fehlschlag wird die jeweils nächste Station versucht.

    Returns:
        Dict {jahr: tatsächlich genutzte station_id} für alle Jahre,
        in denen ein Download geklappt hat.
    """
    successful = {}

    for year in range(start_year, end_year + 1):
        # Primärversuch
        if download_station_year(primary_station, year, cache_dir):
            successful[year] = primary_station
            continue

        # Fallback: nächste Stationen mit gültigem Zeitraum für dieses Jahr
        year_start = pd.Timestamp(year, 1, 1)
        year_end   = pd.Timestamp(year, 12, 31)

        valid_mask = (df_stations["start"] <= year_start) & (df_stations["end"] >= year_end)
        candidates = df_stations.loc[valid_mask]
        if candidates.empty:
            continue

        distances = haversine_km(
            sensor_lat, sensor_lon,
            candidates["latitude"].values, candidates["longitude"].values,
        )
        # Ausschluss der Primärstation, dann die nächsten N
        order = np.argsort(distances)
        alt_ids = [
            candidates.iloc[idx]["id"] for idx in order
            if candidates.iloc[idx]["id"] != primary_station
        ][:max_fallbacks]

        for alt_id in alt_ids:
            if download_station_year(alt_id, year, cache_dir):
                # Speichern unter dem Pfad der Primärstation, damit das Sensor-Mapping gilt
                src = cache_dir / alt_id / f"{year}.csv.gz"
                dst = cache_dir / primary_station / f"{year}.csv.gz"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
                successful[year] = alt_id
                break

    return successful


def _daily_from_hourly(df_hourly: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """
    Leitet tmin/temp/tmax je Kalendertag (Europe/Berlin) aus den stündlichen Temperaturen ab.
    Tage mit weniger als min_hours Stunden ergeben NaN.
    """
    day = df_hourly["timestamp"].dt.normalize()
    grouped = df_hourly.groupby(day)["temp"]

    daily = grouped.agg(tmin="min", temp="mean", tmax="max")
    daily.loc[grouped.count() < min_hours] = np.nan

    daily.index = daily.index.tz_localize(None)
    daily.index.name = "timestamp"
    return daily.reset_index()


def consolidate_station(station_id: str, cache_dir: Path, output_dir: Path, hourly_output_dir: Path) -> bool:
    """
    Fasst alle Jahres-CSVs einer Station zu einer stündlichen Temperaturreihe und einer
    Tagesreihe (tmin/temp/tmax) zusammen. Returns True bei Erfolg.
    """
    station_dir = cache_dir / station_id
    if not station_dir.exists():
        return False

    files = sorted(station_dir.glob("*.csv.gz"))
    if not files:
        return False

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, compression="gzip")
            dfs.append(df)
        except Exception as e:
            print(f"  Lese-Fehler {f.name}: {e}")
            continue

    if not dfs:
        return False

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all[["year", "month", "day"]]) + pd.to_timedelta(df_all["hour"], unit="h")
    df_all["timestamp"] = df_all["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    df_hourly = df_all[["timestamp", "temp"]].sort_values("timestamp")

    hourly_output_dir.mkdir(parents=True, exist_ok=True)
    df_hourly.to_csv(hourly_output_dir / f"{station_id}.csv", index=False)

    df_daily = _daily_from_hourly(df_hourly)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(output_dir / f"{station_id}.csv", index=False)
    return True

# Koordinatengrenzen für Deutschland, um fehlerhafte Geocoding-Einträge in den Metadaten auszuschließen
GERMANY_BOUNDS = {"lat_min": 47.0, "lat_max": 56.0, "lon_min": 5.0, "lon_max": 16.0}


def metadata_to_coords(row: pd.Series) -> tuple[float | None, float | None]:
    """
    Liefert Sensor-Koordinaten aus location_latitude/location_longitude, falls vorhanden und
    innerhalb Deutschlands. Sonst (None, None).
    """
    lat, lon = row.get("location_latitude"), row.get("location_longitude")
    if pd.isna(lat) or pd.isna(lon):
        return None, None
    if not (GERMANY_BOUNDS["lat_min"] <= lat <= GERMANY_BOUNDS["lat_max"]):
        return None, None
    if not (GERMANY_BOUNDS["lon_min"] <= lon <= GERMANY_BOUNDS["lon_max"]):
        return None, None
    return float(lat), float(lon)


def measures_range_from_hourly(sensor_id: str, hourly_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Zeitstempelbereich aus der verarbeiteten Stundendatei, falls measures_from/measures_to in
    den Metadaten fehlen.
    """
    path = hourly_dir / f"{sensor_id}.csv"
    if not path.exists():
        return pd.NaT, pd.NaT
    timestamps = pd.read_csv(path, usecols=["timestamp"])["timestamp"]
    timestamps = pd.to_datetime(timestamps)
    return timestamps.min(), timestamps.max()


def bulk_download_for_sensors(
    df_metadata: pd.DataFrame,
    df_stations: pd.DataFrame,
    cache_dir: Path = Path("data/raw/meteostat/meteostat_bulk_hourly"),
    output_dir: Path = Path("data/raw/meteostat"),
    hourly_output_dir: Path = Path("data/raw/meteostat/hourly"),
    measures_range_fallback=None,
) -> dict:
    """
    Download aller Wetterdaten für die Sensoren in df_metadata (Meteostat Hourly).

    Für jeden Sensor:
    1. Koordinaten aus den Metadaten (Fallback: PLZ → Koordinaten via Geocoding)
    2. Nächste Wetterstation finden
    3. Download (mit Fallback bei fehlenden Jahren)
    4. Konsolidierung zu data/raw/meteostat/hourly/{station_id}.csv (stündlich)
       und data/raw/meteostat/{station_id}.csv (tmin/temp/tmax pro Tag)

    measures_range_fallback(sensor_id) -> (from, to) liefert den Messzeitraum, falls
    measures_from/measures_to in den Metadaten fehlen.

    Returns:
        Mapping {sensor_id: station_id} für erfolgreich verarbeitete Sensoren.
    """
    print(f"Bulk-Download für {len(df_metadata)} Sensoren\n")

    sensor_to_station = {}
    stations_to_consolidate = set()
    skipped = {"coords": 0, "zeitraum": 0, "station": 0, "download": 0}

    for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc="Sensoren"):
        sensor_id = row["id"]

        sensor_lat, sensor_lon = metadata_to_coords(row)
        if sensor_lat is None:
            sensor_lat, sensor_lon = plz_to_coords(row.get("location_post_code"), row.get("location_city"))
        if sensor_lat is None:
            skipped["coords"] += 1
            continue

        measures_from = pd.Timestamp(row["measures_from"])
        measures_to = pd.Timestamp(row["measures_to"])
        if (pd.isna(measures_from) or pd.isna(measures_to)) and measures_range_fallback is not None:
            measures_from, measures_to = measures_range_fallback(sensor_id)
        if pd.isna(measures_from) or pd.isna(measures_to):
            skipped["zeitraum"] += 1
            continue

        # Nächste Wetterstation
        station_id, distance = find_nearest_station(
            sensor_lat, sensor_lon,
            measures_from, measures_to,
            df_stations,
        )
        if station_id is None:
            skipped["station"] += 1
            continue

        # Download mit Fallback (bereits vorhandene Dateien werden übersprungen)
        successful = download_with_fallback(
            primary_station=station_id,
            sensor_lat=sensor_lat,
            sensor_lon=sensor_lon,
            start_year=measures_from.year,
            end_year=measures_to.year,
            df_stations=df_stations,
            cache_dir=cache_dir,
        )
        if not successful:
            skipped["download"] += 1
            continue

        sensor_to_station[sensor_id] = station_id
        stations_to_consolidate.add(station_id)

    # Konsolidierung pro genutzter Station
    print(f"\nKonsolidierung von {len(stations_to_consolidate)} Stationen...")
    consolidated = 0
    for station_id in tqdm(stations_to_consolidate, desc="Konsolidierung"):
        if consolidate_station(station_id, cache_dir, output_dir, hourly_output_dir):
            consolidated += 1

    print(f"\n{'='*50}")
    print(f"Sensoren erfolgreich:        {len(sensor_to_station)}/{len(df_metadata)}")
    print(f"Geskippt - Koordinaten:      {skipped['coords']}")
    print(f"Geskippt - kein Zeitraum:    {skipped['zeitraum']}")
    print(f"Geskippt - keine Station:    {skipped['station']}")
    print(f"Geskippt - Download:         {skipped['download']}")
    print(f"Stationen konsolidiert:      {consolidated}/{len(stations_to_consolidate)}")

    mapping_path = output_dir / "sensor_to_station.csv"
    pd.Series(sensor_to_station, name="station_id").rename_axis("sensor_id").to_csv(mapping_path)

    return sensor_to_station


def map_sensors_to_stations(df_sensors: pd.DataFrame, measures_range_fallback=None, stations_path: Path = Path("data/raw/meteostat/meta_data/stations_de.csv")) -> pd.Series:
    """
    Stationszuordnung für die Pipeline: Sensorliste (Index = sensor_id) -> Series
    sensor_id -> station_id. Sensoren ohne passende Station fehlen im Ergebnis.
    Fehlende Wetterdaten werden bei Bedarf heruntergeladen.
    """
    if not stations_path.exists():
        print("Erzeuge stations_de.csv...")
        get_meteostat_stations(save_file=True)

    df_stations = pd.read_csv(stations_path, parse_dates=["start", "end"])
    mapping = bulk_download_for_sensors(
        df_metadata=df_sensors.reset_index().rename(columns={"sensor_id": "id"}),
        df_stations=df_stations,
        measures_range_fallback=measures_range_fallback,
    )
    return pd.Series(mapping, name="station_id").rename_axis("sensor_id")

def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Distanz in km zwischen einem Punkt und einem Array von Punkten.
    """
    R = 6371.0  # Erdradius in km
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def find_nearest_station(sensor_lat: float, sensor_lon: float, measures_from: pd.Timestamp, measures_to: pd.Timestamp, df_stations: pd.DataFrame) -> tuple[str | None, float | None]:
    """
    Findet die nächstgelegene Wetterstation, deren Zeitraum sich mit dem Messzeitraum des
    Sensors überschneidet. Fehlende Jahre ergänzt download_with_fallback.

    Args:
        sensor_lat, sensor_lon: Koordinaten des Sensors.
        measures_from, measures_to: Messzeitraum des Sensors.
        df_stations: DataFrame mit Spalten id, latitude, longitude, start, end
                     (start/end als pd.Timestamp).

    Returns:
        (station_id, distance_km) oder (None, None) wenn keine Station passt.
    """
    # Zeitfilter: Stationszeitraum überlappt mit dem Sensor-Messzeitraum
    valid_mask = (df_stations["start"] <= measures_to) & (df_stations["end"] >= measures_from)
    candidates = df_stations.loc[valid_mask]

    if candidates.empty:
        return None, None

    # Räumliche Distanz
    distances = haversine_km(
        sensor_lat, sensor_lon,
        candidates["latitude"].values, candidates["longitude"].values,
    )

    idx = np.argmin(distances)
    return candidates.iloc[idx]["id"], float(distances[idx])

def _geocode(name: str) -> tuple[float | None, float | None]:
    """Abfrage der Open-Meteo-Geocoding-API nach Ortsname oder PLZ; (None, None) ohne Treffer."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        r = requests.get(url, params={"name": name, "country": "DE", "count": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None, None

    if not data.get("results"):
        return None, None
    hit = data["results"][0]
    return hit["latitude"], hit["longitude"]


def plz_to_coords(plz: str, city: str | None = None, cache_path: Path = Path("data/cache/plz_to_coords.json")) -> tuple[float | None, float | None]:
    if not plz:
        return (None, None)

    plz = str(plz).strip()
    cache_key = f"{plz}_{city}" if city else plz

    # Cache laden
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    if cache_key in cache:
        entry = cache[cache_key]
        return (entry[0], entry[1]) if entry[0] is not None else (None, None)

    lat, lon = (None, None)
    if city:
        lat, lon = _geocode(city)
    if lat is None:
        lat, lon = _geocode(plz)

    cache[cache_key] = [lat, lon]
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    return (lat, lon)

def load_weather_for_station(station_id: str, weather_dir: Path = Path("data/raw/meteostat")) -> pd.DataFrame:
    path = weather_dir / f"{station_id}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["tmin", "temp", "tmax"])
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")[["tmin", "temp", "tmax"]]


def load_hourly_weather_for_station(station_id: str, hourly_weather_dir: Path = Path("data/raw/meteostat/hourly")) -> pd.Series:
    """
    Lädt die stündliche Temperaturreihe einer Station. Die Zeitstempel werden über UTC geparst
    und in naive lokale Zeit umgerechnet, passend zum Index der Lastreihen. Die doppelte Stunde
    bei der Zeitumstellung im Herbst wird gemittelt.
    """
    path = hourly_weather_dir / f"{station_id}.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="temp")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    series = df.groupby("timestamp")["temp"].mean()
    series.index.name = "timestamp"
    return series
