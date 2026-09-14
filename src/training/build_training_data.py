"""
Baut X/Y-Trainingsarrays aus sensor_table.parquet für einen sensorbasierten Split.

    python -m src.training.build_training_data                 # alle Versionen aus configs/runs.yml
    python -m src.training.build_training_data configs/<cfg>   # nur diese Config

Artefakte je Datenversion <version> unter data/processed/openmeter/training_data/:
    x_<split>_<version>.npy       Kovariaten je Zeile (Reihenfolge wie covariate_columns)
    y_<split>_<version>.npy       [n, 24] normiertes Ziel y_hourly / y_scale
    yscale_<split>_<version>.npy  y_scale je Zeile, zur Skalierung des Encoder-Fensters
    index_<split>_<version>.csv   (sensor_id, day) je Zeile, gleiche Reihenfolge wie x/y
    x0bounds_<version>.json       Wertebereich von x0 im Trainings-Split
    measurement_windows.pkl       Arrays je Sensor für das Fenster-Sampling (für alle Versionen gleich)
"""
import json
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.training.measurement_window import build_sensor_arrays


def _sensors_with_cluster(df: pd.DataFrame) -> set[str]:
    """
    Sensoren mit mindestens einem Cluster-Eintrag. Sensoren ohne Eintrag werden vor dem Split
    ausgeschlossen, da sie in select_valid_rows ohnehin keine gültigen Zeilen hätten.
    """
    valid = set()
    for sensor_id, group in df.groupby(level="sensor_id"):
        if not np.isnan(np.vstack(group["cluster"].to_numpy())).all():
            valid.add(sensor_id)
    return valid


def _sensor_strata(df: pd.DataFrame, sensor_ids: list[str]) -> dict[str, str]:
    """
    Stratifizierungslabel (energy_type) je Sensor, damit Strom und Wärme anteilig auf
    train/val/test verteilt werden.
    """
    strata = {}
    for sensor_id in sensor_ids:
        first_row = df.loc[sensor_id].iloc[0]
        strata[sensor_id] = "Waerme" if first_row["energy_type"] == 1 else "Strom"
    return strata


def create_splits(sensor_ids: list[str], save_dir: Path, ratio_train: float, ratio_val: float, ratio_test: float, seed: int, overwrite: bool = False, strata: dict[str, str] | None = None) -> dict[str, list[str]]:
    """
    Sensorbasierter Split: jeder Sensor liegt vollständig in train, val oder test. Mit strata
    wird jedes Stratum getrennt im angegebenen Verhältnis aufgeteilt.
    """
    if abs(ratio_train + ratio_val + ratio_test - 1.0) > 1e-9:
        raise ValueError("Ratios müssen sich zu 1 summieren.")

    save_dir = Path(save_dir)
    save_path = save_dir / f"training_split_{seed}.json"

    if save_path.exists() and not overwrite:
        raise FileExistsError(f"{save_path} existiert bereits. Mit overwrite=True überschreiben.")

    ids = sorted(sensor_ids)
    if not ids:
        raise ValueError("sensor_ids ist leer.")

    rng = random.Random(seed)

    if strata is None:
        groups = [ids]
    else:
        grouped: dict[str, list[str]] = {}
        for sensor_id in ids:
            grouped.setdefault(strata[sensor_id], []).append(sensor_id)
        groups = [grouped[key] for key in sorted(grouped)]

    train, val, test = [], [], []
    for group in groups:
        shuffled = group.copy()
        rng.shuffle(shuffled)
        n_train = int(round(ratio_train * len(shuffled)))
        n_val = int(round(ratio_val * len(shuffled)))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_splits(path: Path) -> dict[str, list[str]]:
    """Lädt einen zuvor gespeicherten Split aus JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """Flacht eine Spalte zu einem 2D-Array ab (list-typed -> mehrspaltig, skalar -> einspaltig)."""
    values = df[column].to_numpy()
    if len(values) == 0:
        return values.reshape(0, 1).astype(float)
    if np.ndim(values[0]) == 0:
        return values.reshape(-1, 1).astype(float)
    return np.vstack(values).astype(float)


def select_valid_rows(
    df: pd.DataFrame, sensor_ids: list[str], covariate_columns: list[str]
) -> pd.MultiIndex:
    """
    (sensor_id, day)-Zeilen der gegebenen Sensoren, in denen alle covariate_columns ohne NaN
    sind. build_x und build_y verwenden denselben Index.
    """
    mask_sensor = df.index.get_level_values("sensor_id").isin(sensor_ids)
    df_subset = df.loc[mask_sensor]

    if df_subset.empty:
        return df_subset.index

    valid = np.ones(len(df_subset), dtype=bool)
    for column in covariate_columns:
        stacked = _stack_column(df_subset, column)
        valid &= ~np.isnan(stacked).any(axis=1)

    return df_subset.index[valid]


def save_sensor_arrays(
    df: pd.DataFrame,
    sensor_ids: list[str],
    save_path: Path,
    overwrite: bool = False,
) -> dict[str, dict]:
    """
    Speichert die Arrays je Sensor (measurement_window.build_sensor_arrays) nach save_path bzw.
    lädt sie, falls die Datei existiert. Die Fenster selbst werden im Training je Sample gezogen.

    Format: {"sensor_arrays": {sensor_id: build_sensor_arrays(...)}}
    """
    if save_path.exists() and not overwrite:
        return joblib.load(save_path)["sensor_arrays"]

    sensor_arrays: dict[str, dict] = {}
    mask = df.index.get_level_values("sensor_id").isin(sensor_ids)
    for sensor_id, group in df.loc[mask].groupby(level="sensor_id"):
        sensor_arrays[sensor_id] = build_sensor_arrays(group.droplevel("sensor_id").sort_index())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"sensor_arrays": sensor_arrays}, save_path)
    return sensor_arrays


def build_x(df: pd.DataFrame, index: pd.MultiIndex, covariate_columns: list[str]) -> np.ndarray:
    """X für die gegebenen Zeilen als hstack der Kovariaten in der Reihenfolge von covariate_columns."""
    df_subset = df.loc[index]
    parts = [_stack_column(df_subset, column) for column in covariate_columns]
    return np.hstack(parts)


def build_y(df: pd.DataFrame, index: pd.MultiIndex, scale: bool = True) -> np.ndarray:
    """
    Y (24 Stundenwerte je Tag) für die gegebenen Zeilen. scale=True normiert auf die
    Jahresmittellast (y_hourly / y_scale), scale=False liefert kWh.
    """
    df_subset = df.loc[index]
    y = _stack_column(df_subset, "y_hourly")
    if not scale:
        return y

    y_scale = df_subset["y_scale"].to_numpy(dtype=float).reshape(-1, 1)
    return y / np.where(y_scale > 0, y_scale, 1.0)


# Spalten der Sensor-Tabelle, die keine Kovariaten mit vollständiger Abdeckung sind.
NON_COVARIATE_COLUMNS = {"y_hourly", "sin_hour", "cos_hour", "hourly_temp", "y_scale", "prev_h23"}

# Kovariaten mit nur teilweiser Abdeckung (NaN, wenn der Vortag fehlt).
PARTIAL_COVERAGE_COVARIATES = {"prev_h23"}


def _write_x0_bounds(output_dir: Path, version: str) -> None:
    """
    Schreibt x0bounds_<version>.json mit Minimum und Maximum von x0 = y - 1 im Trainings-Split.
    dm.sample() clippt die x0-Schätzung auf diesen Bereich; ohne Obergrenze divergiert die
    verkettete Generierung.
    """
    y_train = np.load(output_dir / f"y_train_{version}.npy")
    x0 = y_train - 1
    bounds = {"x0_min": float(x0.min()), "x0_max": float(x0.max())}
    with open(output_dir / f"x0bounds_{version}.json", "w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=2)
    print(f"x0-Schranken: [{bounds['x0_min']:.4f}, {bounds['x0_max']:.4f}]")


def build_and_save_training_data(sensor_table_path: Path, split_dir: Path, output_dir: Path, covariate_columns: list[str], version: str, seed: int, ratio_train: float, ratio_val: float, ratio_test: float, build_windows: bool = True) -> None:
    """
    Baut X/Y/Index für eine Datenversion. Ein vorhandener Split wird geladen, sonst neu erzeugt.

    y/index/yscale werden nur geschrieben, wenn sie noch nicht existieren; X wird immer neu
    geschrieben. Alle Versionen nutzen dieselbe Zeilenpopulation, eingeschränkt nur durch
    prev_h23, falls es als Kovariate verwendet wird.

    build_windows: False überspringt measurement_windows.pkl (für Versionen ohne Encoder).
    """
    df = pd.read_parquet(sensor_table_path)
    sensor_ids = sorted(df.index.get_level_values("sensor_id").unique())

    sensor_ids = sorted(set(sensor_ids) & _sensors_with_cluster(df))

    split_path = split_dir / f"training_split_{seed}.json"
    if split_path.exists():
        splits = load_splits(split_path)
        splits = {name: [sid for sid in ids if sid in sensor_ids] for name, ids in splits.items()}
    else:
        splits = create_splits(
            sensor_ids, split_dir, ratio_train=ratio_train, ratio_val=ratio_val, ratio_test=ratio_test, seed=seed,
            strata=_sensor_strata(df, sensor_ids),
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    windows_path = output_dir / "measurement_windows.pkl"
    if build_windows and not windows_path.exists():
        save_sensor_arrays(df, sensor_ids, windows_path)

    all_covariate_columns = [c for c in df.columns if c not in NON_COVARIATE_COLUMNS]
    partial_columns = [c for c in covariate_columns if c in PARTIAL_COVERAGE_COVARIATES]

    for split_name, ids in splits.items():
        index = select_valid_rows(df, ids, covariate_columns=all_covariate_columns)
        if partial_columns:
            index = index.intersection(select_valid_rows(df, ids, covariate_columns=partial_columns))

        y_path = output_dir / f"y_{split_name}_{version}.npy"
        if not y_path.exists():
            np.save(y_path, build_y(df, index))
            pd.DataFrame({
                "sensor_id": index.get_level_values("sensor_id"),
                "day": index.get_level_values("day"),
            }).to_csv(output_dir / f"index_{split_name}_{version}.csv", index=False)
            np.save(output_dir / f"yscale_{split_name}_{version}.npy",
                    df.loc[index, "y_scale"].to_numpy(dtype=np.float64))

        x = build_x(df, index, covariate_columns=covariate_columns)
        np.save(output_dir / f"x_{split_name}_{version}.npy", x)

        print(f"{split_name}: X {x.shape} ({len(ids)} Sensoren)")

    _write_x0_bounds(output_dir, version)

    print(f"Geschrieben nach: {output_dir}")


if __name__ == "__main__":
    import argparse

    from src.training.run_config import load_model_config, resolve_runs

    parser = argparse.ArgumentParser(description="Baut X/Y/Index je Datenversion aus configs/runs.yml.")
    parser.add_argument(
        "configs", nargs="*", type=Path,
        help="Optional: einzelne Modell-Config-Dateien statt configs/runs.yml.",
    )
    config_paths = parser.parse_args().configs
    runs = ([load_model_config(p) for p in config_paths] if config_paths
            else resolve_runs(Path("configs/runs.yml")))

    # Eine Config je Datenversion; Messfenster nur, wenn ein Run dieser Version einen Encoder nutzt.
    run_by_version = {r["version"]: r for r in runs}
    versions_needing_windows = {r["version"] for r in runs if r["encoder"] is not None}

    for version in sorted(run_by_version):
        run = run_by_version[version]
        print(f"\n=== Datenversion: {version} (build_windows={version in versions_needing_windows}) ===")
        build_and_save_training_data(
            sensor_table_path=Path("data/processed/openmeter/sensor_table.parquet"),
            split_dir=Path("data/processed/openmeter"),
            output_dir=Path("data/processed/openmeter/training_data"),
            covariate_columns=run["covariate_columns"],
            version=version,
            seed=run["split"]["seed"],
            ratio_train=run["split"]["train"],
            ratio_val=run["split"]["val"],
            ratio_test=run["split"]["test"],
            build_windows=(version in versions_needing_windows),
        )
