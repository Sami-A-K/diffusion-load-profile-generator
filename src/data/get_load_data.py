import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openmeter import OpenMeterClient

load_dotenv()
client = OpenMeterClient(os.environ["OPENMETER_TOKEN"])

# Funktion um alle sensoren via Filter: federal_state
def get_metadata_all_sensors(force: bool = False, path: str = 'data/raw/openmeter/meta_data/meta_data_all_sensors.csv'):
    """
    Download aller Metadaten über die Bundesländer
    Args:
        force (bool): Hiermit kann der Download erzwungen werden, falls die Daten bereits unter path gespeichert sind
        path (str): Speicherort der Metadaten

    Es ist nicht möglich die Metadaten alle auf einmal zu erhalten.
    Über 'get_meta_data_distict()' sind die Metadaten über ein Unterscheidungsmerkmal abrufbar.
    Als Unterscheidungsmerkmal wurde 'federal_state' gewählt, da dies die einzige möglichkeit war viele Daten ohne Fehlermeldung zu erhalten.\n
    Falls andere attribute_name ausprobiert werden wollen, siehe hier:\n
    https://api.openmeter.de/v1/docs#/Meta%20Data/read_attributes_meta_data_distinct_values_get
    """
    out = Path(path)
    if out.exists() and not force:
        return pd.read_csv(out)

    # Wenn Datei nicht vorhanden, dann download
    # Liste mit allen verfügbaren Bundesländern
    list_states = client.get_meta_data_distinct(attribute_name="federal_state")

    # Alle Metadaten über die Bundesländer erhalten
    dfs: list[pd.DataFrame] = []    # list[pd.DataFrame] habe ich nur gemacht, um zu definieren, das in der Liste df's sind.
                                    # Sonst wurde unter fillna() (Zeile 40) die Funktion nicht angezeigt.
                                    # Ist nicht zwingend notwendig aber schöner im Code.
    for state in list_states:
        df = client.get_meta_data(federal_state = state)
        dfs.append(df)

    df  = pd.concat(dfs, axis = 0, ignore_index = True)
    df = df.fillna(pd.NA)

    # Speichern
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df

def get_timeseries_from_ids(ids: pd.Series,
                            base_path: str = 'data/raw/openmeter/timeseries/'):
    '''
    Downlaod und speichern der Zeitreihen über die Sensor IDs
    Args:
        ids (pd.Series): Series mit den abzufragenden IDs.
        base_path (str): Grundpfad, wird mit der Sensor ID ergänzt und den Speicherort zu definieren.
    Returns:
        list: Eine liste mit allen Sensor IDs die nicht heruntergeladen werden konnten.
    '''
    list_sensor_id_not_working = []
    for sensor_id in ids:
        out_path = Path(f'{base_path}/{sensor_id}.csv')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not out_path.exists():
            try:
                client.get_timeseries(sensor_id = sensor_id).to_csv(out_path, index=False)
            except Exception: # Falls die API keine Zeitreihe übergibt, kann keine csv erzeugt werden.
                    #Wenn dieser Fehler auftritt, wird die ID abgespeichert um ggf. Daten anzupassen.
                list_sensor_id_not_working.append(sensor_id)
    return list_sensor_id_not_working


if __name__ == "__main__":
    from src.data.process_load_data import filter_relevant_sensors

    df_metadata = get_metadata_all_sensors()
    # Zeitreihen der Sensoren, die den Metadatenfilter der Pipeline passieren
    df_relevant = filter_relevant_sensors(df_metadata)
    df_relevant = df_relevant[df_relevant["location_country"] == "Deutschland"]

    list_sensor_id_not_working = get_timeseries_from_ids(df_relevant['id'])
    print(f"{len(df_relevant)} Sensoren, {len(list_sensor_id_not_working)} ohne Zeitreihe")
