"""
Direkte Auswertung trainierter Checkpoints: lädt ein Modell aus src/model/model_weights/,
generiert Tagesprofile für ausgewählte (sensor_id, day)-Zeilen des Test-Splits und berechnet den
Zeitkontext-Fehler je Sensor. Die Architektur wird aus training_history_<version_suffix>.json
rekonstruiert.

Die Auswertung der vollständigen Test-Population erfolgt über die gecachten Samples
(sample_population.py -> compare_cached_samples.py). generate_independent() erlaubt zusätzlich
eine feste Fensterlänge (window_override) und Messfenster anderer Gebäude (encoder_keys).

    python -m src.evaluation.compare_models        # Stichprobe (N_SENSORS_PER_EVAL x N_DAYS_PER_SENSOR) fuer VERSIONS
"""
import gc
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import src.model.diffusion_model as dm
from src.training.build_training_data import build_x, build_y, load_splits, select_valid_rows, NON_COVARIATE_COLUMNS, PARTIAL_COVERAGE_COVARIATES
from src.training.measurement_window import build_sensor_arrays, sample_measurement_window, scale_window, n_series_features
from src.evaluation.evaluation_metrics import duration_curve_error_per_sensor
from src.model.measurement_series_patch_encoder import MeasurementSeriesPatchEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENSOR_TABLE_PATH = Path("data/processed/openmeter/sensor_table.parquet")
SPLIT_DIR = Path("data/processed/openmeter")
DATA_DIR = Path("data/processed/openmeter/training_data")
WEIGHTS_DIR = Path("src/model/model_weights")
RESULTS_DIR = Path("results")

# Checkpoints für den __main__-Lauf
VERSIONS = ["main", "main_noenc"]

SPLIT = "test"
WEIGHTS_SUFFIX = "best"

SEQ_LEN = 24
DIFFUSION_STEPS = 500
BATCH_SIZE = 512
# Seed für die Auswahl von Sensoren, Tagen und Messfenstern
SEED = 48

# True = alle gemeinsamen Test-Zeilen, False = Stichprobe
EVAL_FULL_POPULATION = False
N_SENSORS_PER_EVAL = 30
N_DAYS_PER_SENSOR = 200

# Deterministische Algorithmen und kein TF32 für reproduzierbare Ergebnisse
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_x0_bounds(version: str) -> tuple[float, float]:
    """(x0_min, x0_max) aus x0bounds_<version>.json für dm.sample()."""
    path = DATA_DIR / f"x0bounds_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt - mit python -m src.training.build_training_data neu bauen.")
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    return float(b["x0_min"]), float(b["x0_max"])


@lru_cache(maxsize=1)
def get_sensor_table() -> pd.DataFrame:
    return pd.read_parquet(SENSOR_TABLE_PATH)


def build_measurement_batch(
    sensor_ids: np.ndarray, target_days: np.ndarray, window_scales: np.ndarray, sensor_table: pd.DataFrame,
    min_days: int, max_days: int, seed: int, token_columns: list[str] | None,
):
    """
    Encoder-Eingabe für Modelle mit Encoder: je (sensor_id, target_day) ein zufälliges
    Messfenster aus dem Kalenderjahr des Zieltags, skaliert mit window_scales[i], wie im Training.
    Returns: (series [n, max_days*24, n_series_features], static [n, n_static_features], padding_mask [n, max_days*24])
    """
    rng = np.random.default_rng(seed)
    unique_sensors = set(np.unique(sensor_ids))
    relevant = sensor_table[sensor_table.index.get_level_values("sensor_id").isin(unique_sensors)]
    sensor_arrays_cache = {
        sensor_id: build_sensor_arrays(group.droplevel("sensor_id").sort_index())
        for sensor_id, group in relevant.groupby(level="sensor_id")
    }

    series_list, static_list = [], []
    for sensor_id, day, scale in zip(sensor_ids, target_days, window_scales):
        arrays = sensor_arrays_cache[sensor_id]
        sampled = sample_measurement_window(arrays, pd.Timestamp(day), min_days, max_days, rng, token_columns=token_columns)
        if sampled is None:
            series = np.zeros((1, n_series_features(token_columns)), dtype=np.float32)
            static = arrays["category"]
        else:
            series, static = sampled
            series = scale_window(series, float(scale), token_columns=token_columns).astype(np.float32)
        series_list.append(torch.from_numpy(series.astype(np.float32)))
        static_list.append(torch.from_numpy(static.astype(np.float32)))

    static_batch = torch.stack(static_list)
    max_len = max_days * 24
    padded = torch.zeros(len(series_list), max_len, series_list[0].shape[1])
    padding_mask = torch.ones(len(series_list), max_len, dtype=torch.bool)
    for i, s in enumerate(series_list):
        padded[i, : s.shape[0]] = s
        padding_mask[i, : s.shape[0]] = False
    return padded, static_batch, padding_mask


def generate_for_dataset(model, encoder, schedule, conditions, series, static, padding_mask,
                         x0_bounds: tuple[float, float], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Generiert ein Sample je Kovariatenzeile, zurückgegeben in der normierten Skala (y >= 0).
    encoder=None für Modelle ohne Encoder."""
    model.eval()
    if encoder is not None:
        encoder.eval()
    cond_t = torch.FloatTensor(conditions)
    x0_min, x0_max = x0_bounds

    out = []
    if encoder is None:
        loader = DataLoader(TensorDataset(cond_t), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (cond_batch,) in loader:
                gen = dm.sample(model, schedule, cond_batch.to(device), device, x0_min=x0_min, x0_max=x0_max)
                out.append(gen.numpy())
    else:
        loader = DataLoader(TensorDataset(cond_t, series, static, padding_mask), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for cond_batch, series_batch, static_batch, mask_batch in loader:
                context, context_padding_mask = encoder(series_batch.to(device), static_batch.to(device), mask_batch.to(device))
                gen = dm.sample(model, schedule, cond_batch.to(device), device, context, context_padding_mask, x0_min=x0_min, x0_max=x0_max)
                out.append(gen.numpy())

    return np.maximum(np.vstack(out) + 1.0, 0.0)


def load_run_metadata(version_suffix: str, weights_dir: Path = WEIGHTS_DIR) -> tuple[dict, int, int]:
    """(run_config, n_features, n_static_features) aus training_history_<version_suffix>.json."""
    with open(weights_dir / f"training_history_{version_suffix}.json") as f:
        history = json.load(f)
    return history["run_config"], history["n_features"], history["n_static_features"]


def load_model(version: str, n_features: int, n_static_features: int, run_config: dict, weights_suffix: str = WEIGHTS_SUFFIX):
    """Rekonstruiert (model, encoder) aus einem Checkpoint; encoder ist None bei Modellen ohne Encoder."""
    checkpoint = torch.load(WEIGHTS_DIR / f"model_weights_{version}_{weights_suffix}.pth", map_location=device)

    model = dm.ConditionedDenoiser(seq_len=SEQ_LEN, n_features=n_features, **run_config["denoiser"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    encoder = None
    if run_config["encoder"] == "patch":
        encoder = MeasurementSeriesPatchEncoder(
            n_series_features=n_series_features(run_config["token_columns"]), n_static_features=n_static_features,
            **run_config["encoder_params"],
        )
        encoder.load_state_dict(checkpoint["encoder"])
        encoder.to(device)

    return model, encoder


def common_eval_keys(versions: list[str], split_name: str, split_seed: int):
    """
    Schnittmenge der gültigen (sensor_id, day)-Zeilen aller versions im Split, nach denselben
    Regeln wie build_training_data.build_and_save_training_data.
    Returns: (common_keys, {version: version_index}).
    """
    df = get_sensor_table()
    splits = load_splits(SPLIT_DIR / f"training_split_{split_seed}.json")
    all_covariate_columns = [c for c in df.columns if c not in NON_COVARIATE_COLUMNS]
    universal_index = select_valid_rows(df, splits[split_name], all_covariate_columns)

    per_version_index = {}
    for version in versions:
        run_config, _, _ = load_run_metadata(version)
        partial = [c for c in run_config["covariate_columns"] if c in PARTIAL_COVERAGE_COVARIATES]
        index = universal_index
        if partial:
            index = index.intersection(select_valid_rows(df, splits[split_name], partial))
        per_version_index[version] = index

    common = per_version_index[versions[0]]
    for version in versions[1:]:
        common = common.intersection(per_version_index[version])
    return common.sort_values(), per_version_index


def sample_per_sensor_keys(common_keys_full: pd.MultiIndex, n_sensors: int, n_days_per_sensor: int, seed: int) -> pd.MultiIndex:
    """n_sensors zufällige Sensoren mit mindestens n_days_per_sensor Tagen, jeweils die ersten
    n_days_per_sensor Tage in chronologischer Reihenfolge."""
    rng = np.random.default_rng(seed)
    sensor_ids = common_keys_full.get_level_values("sensor_id")
    counts = pd.Series(sensor_ids).value_counts()
    eligible = np.sort(counts[counts >= n_days_per_sensor].index.to_numpy())
    if len(eligible) == 0:
        raise ValueError(f"Kein Sensor hat >= {n_days_per_sensor} gueltige Tage im Eval-Set.")
    chosen = rng.choice(eligible, size=min(n_sensors, len(eligible)), replace=False)
    selected = [common_keys_full[sensor_ids == sid][:n_days_per_sensor] for sid in chosen]
    return selected[0].append(selected[1:]).sort_values()


def generate_independent(version: str, sensor_table: pd.DataFrame, keys: pd.MultiIndex, gen_seed: int,
                         window_override: tuple[int, int] | None = None,
                         encoder_keys: pd.MultiIndex | None = None) -> np.ndarray:
    """
    Generiert jede Zeile unabhängig mit dem echten Vortageswert als prev_h23. X wird aus der
    Sensor-Tabelle gebaut.

    window_override: (min_days, max_days) des Messfensters statt der Werte aus der Config.
    encoder_keys: (sensor_id, day)-Zeilen, aus denen das Messfenster gezogen wird (gleiche Länge
        wie keys). Das Fenster wird mit dem y_scale der encoder_keys-Zeile skaliert.
    """
    run_config, n_features, n_static_features = load_run_metadata(version)
    model, encoder = load_model(version, n_features=n_features, n_static_features=n_static_features, run_config=run_config)
    schedule = dm.DiffusionSchedule(timesteps=DIFFUSION_STEPS, schedule_type=run_config.get("noise_schedule", "linear")).to(device)

    x = build_x(sensor_table, keys, covariate_columns=run_config["covariate_columns"])
    series = static = padding_mask = None
    if encoder is not None:
        window_keys = keys if encoder_keys is None else encoder_keys
        assert len(window_keys) == len(keys), "encoder_keys muss dieselbe Laenge wie keys haben."
        series, static, padding_mask = build_measurement_batch(
            window_keys.get_level_values("sensor_id").to_numpy(), window_keys.get_level_values("day").to_numpy(),
            sensor_table.loc[window_keys, "y_scale"].to_numpy(dtype=float), sensor_table,
            *(window_override or (run_config["measurement_min_days"], run_config["measurement_max_days"])),
            seed=SEED, token_columns=run_config["token_columns"],
        )

    torch.manual_seed(gen_seed)
    y_gen = generate_for_dataset(model, encoder, schedule, x, series, static, padding_mask, x0_bounds=load_x0_bounds(run_config["version"]))

    del model, encoder, schedule
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_gen


def evaluate_version(version: str, sensor_table: pd.DataFrame, keys: pd.MultiIndex, gen_seed: int = SEED) -> pd.DataFrame:
    """Zeitkontext-Fehler je Sensor für keys (unabhängige Generierung)."""
    y_real = build_y(sensor_table, keys)
    y_gen = generate_independent(version, sensor_table, keys, gen_seed)
    per_sensor, _ = duration_curve_error_per_sensor(
        y_real, y_gen, keys.get_level_values("sensor_id").to_numpy(), keys.get_level_values("day").to_numpy(),
        sensor_table.loc[keys, "is_holiday"].to_numpy().astype(int),
    )
    return per_sensor.assign(version_suffix=version)


if __name__ == "__main__":
    print(device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    split_seeds = {load_run_metadata(v)[0]["split"]["seed"] for v in VERSIONS}
    if len(split_seeds) != 1:
        raise ValueError(f"VERSIONS haben unterschiedliche split-Seeds ({split_seeds}) - keine identische Sensor-Population.")

    common_keys_full, _ = common_eval_keys(VERSIONS, SPLIT, split_seeds.pop())
    print(f"Gemeinsamer Eval-Pool ueber {len(VERSIONS)} Versionen: {len(common_keys_full)} Zeilen")
    keys = common_keys_full if EVAL_FULL_POPULATION else sample_per_sensor_keys(common_keys_full, N_SENSORS_PER_EVAL, N_DAYS_PER_SENSOR, SEED)
    print(f"Eval-Set: {len(keys)} Zeilen ueber {keys.get_level_values('sensor_id').nunique()} Sensoren")

    sensor_table = get_sensor_table()
    per_sensor = pd.concat([evaluate_version(v, sensor_table, keys) for v in VERSIONS], ignore_index=True)
    summary = per_sensor.groupby("version_suffix")[["error_global", "error_context", "error_structure", "error_hour"]].mean() * 1e2
    print("\nZeitkontext-Fehler je Version, Mittel ueber Sensoren [%]:")
    print(summary.round(3).to_string())

    tag = "full" if EVAL_FULL_POPULATION else f"{N_SENSORS_PER_EVAL}x{N_DAYS_PER_SENSOR}"
    per_sensor.to_csv(RESULTS_DIR / f"live_eval_{SPLIT}_{tag}_per_sensor.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"live_eval_{SPLIT}_{tag}.csv")
    print(f"geschrieben: {RESULTS_DIR}/live_eval_{SPLIT}_{tag}*.csv")
