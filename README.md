# Konditionierung eines Diffusionsmodells durch Lastgangmessungen zur Erzeugung synthetischer Gebäudelastprofile

## Pipeline

```
Rohdaten (OpenMeter, Meteostat)
  └─ src/data/pipeline.py                         Stufen 1-7 -> data/processed/openmeter/sensor_table.parquet
       └─ src/training/build_training_data.py     Sensor-Split, X/Y-Arrays je Datenversion
            └─ src/training/train_model.py        Training (configs/*.yml)
                 └─ src/evaluation/sample_population.py      Sample-Caches je Checkpoint (data/samples/)
                      └─ src/evaluation/compare_cached_samples.py   Metriken -> results/
                           └─ abbildungen/*.py                     Abbildungen -> abbildungen/outputs/
```

Alle Aufrufe aus dem Projektordner:

```bash
python -m src.data.pipeline                                   # Datenpipeline, Stufe 1-7
python -m src.training.build_training_data                    # Trainingsdaten aller Configs aus configs/runs.yml
python -m src.training.train_model configs/config_main.yml    # ein Modell trainieren (ohne Argument: alle)
python -m src.evaluation.sample_population --configs configs/config_main.yml
python -m src.evaluation.compare_cached_samples               # Zeitkontext-Fehler und Sprungfaktor je Sensor
python -m abbildungen.plot_main_vs_baseline                   # Abbildung nach abbildungen/outputs/
```

## Daten beziehen

Im Repository enthalten sind Code, Configs, Ergebnistabellen (`results/`), Abbildungen, Trainingshistorien
und die Checkpoints `*_best.pth`, außerdem:

- `data/processed/openmeter/training_split_99.json`: Sensor-IDs je Split (873 train / 109 val / 110 test)
- `data/processed/openmeter/cluster_proportions.csv`: Cluster-Anteile je Kategorie und Energieart, benötigt
  für das Sampling mit `--cluster-source sampled`
- `data/processed/openmeter/training_data/x0bounds_<version>.json`: Wertebereich von x0 im Trainings-Split,
  benötigt für das Sampling mit den Checkpoints

Abbildungen, die nur auf den Ergebnistabellen beruhen, lassen sich damit direkt erzeugen. Rohdaten,
Sensor-Tabelle, Trainingsdaten und Sample-Caches werden nicht weitergegeben, sondern lokal erzeugt:

1. OpenMeter: API-Token unter https://www.openmeter.de beantragen, als `OPENMETER_TOKEN` in einer
   `.env`-Datei hinterlegen und `python -m src.data.get_load_data` ausführen
   (Metadaten und Zeitreihen nach `data/raw/openmeter/`).
2. Meteostat: `stations.db` von https://data.meteostat.net/stations.db nach
   `data/raw/meteostat/meta_data/` legen. Die Wetterdaten lädt `src.data.pipeline` selbst.
3. `python -m src.data.pipeline` erzeugt `data/processed/openmeter/sensor_table.parquet`.
4. `python -m src.training.build_training_data` erzeugt `data/processed/openmeter/training_data/` und
   verwendet dabei den mitgelieferten Split.
5. `python -m src.evaluation.sample_population` erzeugt die Sample-Caches (`data/samples/`), die
   `compare_cached_samples` und die Abbildungen mit Einzelprofilen benötigen.


**Reproduzierbarkeit:** Die Zeitreihen der Arbeit wurden zwischen November 2025 und März 2026 von openMeter abgerufen. Über `training_split_99.json` ist die Zuordnung der damals verfügbaren Sensoren zu train/val/test festgehalten. 


## Ordnerstruktur
```
src/data/          Datenpipeline: Metadatenfilter, Wetterdaten, Stundenreihen, PV-Erkennung,
                   Clustering, Sensor-Tabelle, Kovariaten-Encoding
src/training/      Trainingsdaten, Sampling der Messfenster, Config-Loader, Training
src/model/         Diffusions-Decoder (diffusion_model.py), Encoder (measurement_series_patch_encoder.py),
                   model_weights/ mit Checkpoints (*.pth) und Trainingshistorien (training_history_<name>.json)
src/evaluation/    Sampling, Metriken, Auswertung der Sample-Caches
configs/           eine Config je trainiertem Modell, Liste aller Configs in runs.yml
results/           Ergebnistabellen der Auswertung (CSV)
abbildungen/       Skripte der Abbildungen (plot_style.py = gemeinsame Formatierung), outputs/ = erzeugte Abbildungen
data/              raw/ (Rohdaten), processed/ (Sensor-Tabelle, Trainingsdaten), samples/ (Sample-Caches),
                   cache/ (Geocoding-Cache)
```

## Modelle

Der Dateiname der Config (`config_<name>.yml`) benennt Checkpoint, Trainingshistorie und Sample-Cache.

| Config | Unterschied |
|---|---|
| `config_main.yml` | Hauptmodell: Encoder (1 Layer, 16 Heads, Patch 8 h) + Decoder mit Cross-Attention; Kovariaten `energy_type, cyclical_time, is_holiday, daily_weather, category, prev_h23` |
| `config_patch_8h_seed{05,07,12,20}.yml` | Hauptmodell mit anderem Trainings-Seed |
| `config_patch_{1,2,4,6,12,24}h.yml` | Patch-Länge |
| `config_patch_{1,24}h_seed{12,20}.yml` | Patch-Länge mit anderem Trainings-Seed |
| `config_main_l3.yml` | 3 Encoder-Layer |
| `config_main_noenc.yml` | ohne Encoder |
| `config_main_noh23.yml` | ohne `prev_h23` |
| `config_main_nocat.yml` | ohne `category` |
| `config_main_cluster.yml` | `cluster` statt `category` |
| `config_main_all_covariates.yml` | `category` und `cluster` |
| `config_baseline.yml` | Modell der vorangegangenen Arbeit: ohne Encoder, Kovariate `cluster`, ohne `prev_h23` |
| `config_baseline_seed{05,07,12,20}.yml` | Baseline mit anderem Trainings-Seed |
| `config_baseline_h23.yml` | Baseline mit `prev_h23` |
| `config_baseline_category.yml` | Baseline mit `category` statt `cluster` |

Alle Modelle: 50 Epochen, Batch 512, Adam 1e-3, 500 Diffusionsschritte (linearer Schedule),
EMA-Gewichte für Validierung und Checkpoints, sensorbasierter 80/10/10-Split (Seed 99,
stratifiziert nach Strom/Wärme).

## Festgelegte Verarbeitungsschritte

- **Normierung auf die Jahresmittellast:** `y = y_hourly / y_scale`, `y_scale` = mittlere Last des
  Kalenderjahres (1.0 = Jahresmittellast, Nullpunkt 0 kWh). Beim Einsatz ergibt sich die Skala aus dem
  Jahresenergiebedarf `E_a / 8760`; die Rückskalierung ist `y_kW = y * E_a / 8760`.
- **Encoder-Fenster** stammen aus dem Kalenderjahr des Zieltags und werden mit derselben
  Jahresmittellast normiert (`measurement_window.py`).
- **Sensor-Jahre ohne auswertbares Lastprofil** (über 50 % Nullstunden oder mittlere Last unter 50 W)
  sind ausgeschlossen (`build_sensor_table.implausible_years`).
- **Decoder:** ResBlock-MLP mit FiLM-Konditionierung und einem Cross-Attention-Block direkt nach der
  Eingangsprojektion.
- **Sampling:** Die x0-Schätzung wird in jedem Reverse-Schritt auf den Wertebereich des
  Trainings-Splits begrenzt (`x0bounds_<version>.json`).
- **Jahresprofile** entstehen autoregressiv: Jeder Folgetag erhält den generierten Wert der Stunde 23
  des Vortags als `prev_h23`; Ketten beginnen am Jahreswechsel neu (`sample_population.py`, Variante
  `deployment`). Die Variante `oracle` generiert jeden Tag unabhängig mit dem echten Vortageswert.

## Grundlage

Der Code baut auf der Codebasis der vorangegangenen Masterarbeit von Pascal Ruhl auf (Datenabruf, Clustering, Diffusionsmodell der Baseline). 


## Datenquellen und Lizenzen

- **Code:** MIT, siehe [LICENSE](LICENSE).
- **Lastgangdaten:** [openMeter](https://www.openmeter.de), Open Database License
  ([ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)). Der daraus abgeleitete Sensor-Split (`training_split_99.json`) steht ebenfalls unter ODbL 1.0.
- **Wetterdaten:** [Meteostat](https://meteostat.net) und die jeweiligen Datenanbieter, u. a. der Deutsche Wetterdienst ([Lizenz](https://dev.meteostat.net/license)).

