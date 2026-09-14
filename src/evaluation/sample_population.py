"""
Generiert je Checkpoint Samples für die Test-Population und speichert sie, damit Metriken
(compare_cached_samples.py) und Abbildungen ohne erneute Diffusion berechnet werden können.

Varianten:
  "deployment": Einsatzfall. Bei Modellen mit prev_h23 werden die Tage autoregressiv verkettet
                (prev_h23 = generierter Stunde-23-Wert des Vortags); alle Zeilen gleicher
                Kettentiefe werden gemeinsam verarbeitet. Ohne prev_h23 sind die Tage unabhängig.
  "oracle":     jeder Tag unabhängig mit echtem Vortageswert als prev_h23; nur für Modelle mit
                prev_h23.

--cluster-source: "true" verwendet das beobachtete Cluster des Sensors, "sampled" zieht je Sensor
ein Cluster aus cluster_proportions.csv (Fall eines neuen Gebäudes ohne Lasthistorie). Nur für
Modelle mit cluster als Kovariate relevant.

--chain-anchor year: jede Kette beginnt am Jahreswechsel neu, da y_scale je Kalenderjahr definiert
ist. Datenlücken unterbrechen die Kette. Das Messfenster wird in beiden Varianten mit demselben
Seed je Zeile gezogen.

Aufruf (Repo-Root):
    python -m src.evaluation.sample_population                      # alle Configs aus configs/runs.yml
    python -m src.evaluation.sample_population --configs configs/config_main.yml
    python -m src.evaluation.sample_population --sensor-limit 5 --device cpu   # Kurztest
    python -m src.evaluation.sample_population --configs configs/config_main.yml \
        --variant oracle --noise-seed 1

Ausgabe: data/samples/<version_suffix>/best_deployment[_yearanchor]_test.npz und bei Modellen mit
prev_h23 best_oracle_test.npz (sensor_id/day/split/y_sample/y_real in der normierten Skala) plus
JSON-Metadaten. Abweichende --seed/--noise-seed/--population erzeugen eigene Dateinamen.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import src.model.diffusion_model as dm
from src.data.build_clusters import CLUSTER_K, sample_cluster
from src.data.encode_covariates import decode_category
from src.data.lookup_tables import CATEGORIES
from src.model.measurement_series_patch_encoder import MeasurementSeriesPatchEncoder
from src.training.measurement_window import n_series_features, sample_measurement_window, scale_window
from src.training.run_config import cache_stem, cache_variants, load_model_config, load_run_index, model_chains

DATA_DIR = Path("data/processed/openmeter/training_data")
WEIGHTS_DIR = Path("src/model/model_weights")
CACHE_VARIANTS_ORDER = ("deployment", "oracle")
SAMPLES_DIR = Path("data/samples")
CLUSTER_PROPORTIONS_PATH = DATA_DIR.parent / "cluster_proportions.csv"

# prev_h23 muss die letzte Kovariate sein; die Spalte wird zusätzlich über die Korrelation geprüft.
PREV_H23_CORR_MIN = 0.999

# Spaltenbreite je Kovariate in x (Reihenfolge wie covariate_columns).
_COVARIATE_WIDTHS = {
    "energy_type": 1,
    "cyclical_time": 4,
    "is_holiday": 1,
    "daily_weather": 3,
    "category": len(CATEGORIES),
    "cluster": sum(CLUSTER_K.values()),
    "prev_h23": 1,
}


def _load_population(version: str, sensor_limit: int | None, splits: tuple[str, ...]) -> dict:
    """Lädt index/y/x/yscale der angegebenen Splits als eine Population; sensor_limit wird auf alle
    Arrays gleich angewendet."""
    idx_parts, y_parts, x_parts, ys_parts, split_parts = [], [], [], [], []
    for split in splits:
        idx = pd.read_csv(DATA_DIR / f"index_{split}_{version}.csv", parse_dates=["day"])
        idx_parts.append(idx)
        y_parts.append(np.load(DATA_DIR / f"y_{split}_{version}.npy"))
        x_parts.append(np.load(DATA_DIR / f"x_{split}_{version}.npy"))
        ys_parts.append(np.load(DATA_DIR / f"yscale_{split}_{version}.npy"))
        split_parts.append(np.full(len(idx), split))

    index = pd.concat(idx_parts, ignore_index=True)
    index["split"] = np.concatenate(split_parts)
    y = np.concatenate(y_parts).astype(np.float32)
    x = np.concatenate(x_parts).astype(np.float32)
    yscale = np.concatenate(ys_parts).astype(np.float64)
    if not (len(index) == len(y) == len(x) == len(yscale)):
        raise ValueError(f"Artefakte der Version {version} haben unterschiedliche Zeilenzahlen.")

    if sensor_limit is not None:
        keep_sensors = set(sorted(index["sensor_id"].unique())[:sensor_limit])
        mask = index["sensor_id"].isin(keep_sensors).to_numpy()
        index, y, x, yscale = index[mask].reset_index(drop=True), y[mask], x[mask], yscale[mask]

    return {"index": index, "y": y, "x": x, "yscale": yscale}


def _load_x0_bounds(version: str) -> tuple[float, float]:
    """(x0_min, x0_max) aus x0bounds_<version>.json für dm.sample()."""
    path = DATA_DIR / f"x0bounds_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt - mit python -m src.training.build_training_data neu bauen.")
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    return float(b["x0_min"]), float(b["x0_max"])


def _prev_h23_column(run_cfg: dict, population: dict, sample_size: int = 20_000) -> int | None:
    """Spaltenindex von prev_h23 in x (None ohne prev_h23). Die Spalte wird über die Korrelation mit
    dem echten Stunde-23-Wert des Vortags geprüft. Paare über den Jahreswechsel bleiben dabei
    unberücksichtigt, da sich dort die Skala ändert."""
    cols = run_cfg["covariate_columns"]
    if "prev_h23" not in cols:
        return None
    if cols[-1] != "prev_h23":
        raise ValueError("prev_h23 muss die letzte Kovariate der Config sein (Spaltenposition in x).")
    col = population["x"].shape[1] - 1

    index = population["index"]
    n = len(index)
    lookup = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([index["sensor_id"], index["day"]]))
    rng = np.random.default_rng(0)
    sample = rng.choice(n, size=min(sample_size, n), replace=False)
    day = index["day"].to_numpy()[sample]
    prev_day = day - np.timedelta64(1, "D")
    same_year = pd.DatetimeIndex(day).year == pd.DatetimeIndex(prev_day).year
    rows = lookup.reindex(list(zip(index["sensor_id"].to_numpy()[sample], prev_day)))
    valid = rows.notna().to_numpy() & same_year
    if valid.sum() < 100:
        print("[verify] zu wenige Vortag-Ueberschneidungen in der Stichprobe (--sensor-limit?) - Verifikation uebersprungen.")
        return col
    y_prev = population["y"][rows[valid].to_numpy().astype(int), 23]
    corr = np.corrcoef(population["x"][sample][valid][:, col], y_prev)[0, 1]
    if corr < PREV_H23_CORR_MIN:
        raise RuntimeError(f"prev_h23-Spalte {col} nicht bestaetigt: Korrelation mit dem echten Vortageswert nur {corr:.4f}.")
    print(f"[verify] prev_h23-Spalte {col} bestaetigt (r={corr:.6f}, n={valid.sum()})")
    return col


def _covariate_slice(run_cfg: dict, name: str) -> slice | None:
    """Spaltenbereich einer Kovariate in x (None, falls nicht in covariate_columns)."""
    cols = run_cfg["covariate_columns"]
    if name not in cols:
        return None
    start = sum(_COVARIATE_WIDTHS[c] for c in cols[: cols.index(name)])
    return slice(start, start + _COVARIATE_WIDTHS[name])


def _load_cluster_proportions() -> pd.DataFrame:
    if not CLUSTER_PROPORTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{CLUSTER_PROPORTIONS_PATH} fehlt - mit python -m src.data.pipeline neu bauen."
        )
    return pd.read_csv(CLUSTER_PROPORTIONS_PATH)


def _resample_cluster_column(population: dict, run_cfg: dict, sensor_arrays: dict,
                             proportions: pd.DataFrame, seed: int) -> None:
    """Ersetzt die Cluster-Spalten in population["x"] durch ein je Sensor gezogenes Cluster
    (build_clusters.sample_cluster), konstant über alle Zeilen des Sensors. category stammt aus
    measurement_windows.pkl, energy_type aus x. Ohne cluster-Kovariate wirkungslos.
    """
    cluster_slice = _covariate_slice(run_cfg, "cluster")
    if cluster_slice is None:
        return
    energy_slice = _covariate_slice(run_cfg, "energy_type")
    if energy_slice is None:
        raise ValueError(
            "--cluster-source sampled braucht energy_type als Kovariate, um die (category, "
            "energy_type)-Zelle in cluster_proportions.csv zu bestimmen."
        )
    energy_col = energy_slice.start

    index = population["index"]
    x = population["x"]
    cluster_width = cluster_slice.stop - cluster_slice.start
    new_cluster = np.zeros((len(index), cluster_width), dtype=np.float32)

    sensor_ids = index["sensor_id"].to_numpy()
    rng = np.random.default_rng(seed)
    for sensor_id in sorted(set(sensor_ids)):
        rows = np.flatnonzero(sensor_ids == sensor_id)
        arrays = sensor_arrays.get(sensor_id)
        if arrays is None:
            raise KeyError(f"Kein measurement_windows-Eintrag fuer Sensor {sensor_id!r} - category nicht bestimmbar.")
        category = decode_category(arrays["category"])
        energy_type = "Waerme" if x[rows[0], energy_col] >= 0.5 else "Strom"
        cluster_id = sample_cluster(category, energy_type, proportions, rng)
        new_cluster[rows, cluster_id] = 1.0

    x[:, cluster_slice] = new_cluster
    print(f"[cluster] {len(set(sensor_ids))} Sensoren: Cluster aus cluster_proportions.csv gezogen (statt wahrem Wert).")


def _compute_chain_metadata(index: pd.DataFrame, max_chain_depth: int | None, anchor_year: bool) -> tuple[np.ndarray, np.ndarray]:
    """Je Zeile parent_row (Zeile des Vortags desselben Sensors, sonst -1) und depth (Kettentiefe,
    0 = Kettenstart). anchor_year startet Ketten am Jahreswechsel neu; max_chain_depth begrenzt die
    Kettenlänge."""
    order = index[["sensor_id", "day"]].sort_values(["sensor_id", "day"], kind="mergesort")
    prev_day = order.groupby("sensor_id")["day"].shift(1)
    consecutive = (order["day"] - prev_day) == pd.Timedelta(days=1)
    if anchor_year:
        consecutive &= order["day"].dt.year == prev_day.dt.year

    grp = (~consecutive).cumsum()
    depth_sorted = grp.groupby(grp).cumcount().to_numpy()
    orig_idx = order.index.to_numpy()
    parent_sorted = np.where(consecutive.to_numpy(), np.roll(orig_idx, 1), -1)

    if max_chain_depth is not None:
        period = max_chain_depth + 1
        is_resync = (depth_sorted % period) == 0
        parent_sorted = np.where(is_resync, -1, parent_sorted)
        depth_sorted = depth_sorted % period

    parent_row = np.full(len(index), -1, dtype=np.int64)
    depth = np.zeros(len(index), dtype=np.int64)
    parent_row[orig_idx] = parent_sorted
    depth[orig_idx] = depth_sorted
    return parent_row, depth


def _build_model(run_cfg: dict, n_features: int, n_static_features: int, series_features: int, device):
    """Erzeugt Denoiser und ggf. Encoder wie in train_model.build_run."""
    model = dm.ConditionedDenoiser(seq_len=run_cfg["seq_len"], n_features=n_features, **run_cfg["denoiser"]).to(device)
    encoder = None
    if run_cfg["encoder"] == "patch":
        encoder = MeasurementSeriesPatchEncoder(
            n_series_features=series_features, n_static_features=n_static_features, **run_cfg["encoder_params"],
        ).to(device)
    return model, encoder


def _load_checkpoint(model, encoder, path: Path, device) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    if encoder is not None:
        encoder.load_state_dict(state["encoder"])
        encoder.eval()


def _encode_batch(row_ids: np.ndarray, population: dict, sensor_arrays: dict, run_cfg: dict, seed: int):
    """Zieht je Zeile ein Messfenster mit dem Seed row_id + seed, damit das Fenster in beiden
    Varianten und bei wiederholten Aufrufen gleich ist."""
    if run_cfg["encoder"] is None:
        return None, None, None

    token_columns = run_cfg["token_columns"]
    min_days, max_days = run_cfg["measurement_min_days"], run_cfg["measurement_max_days"]
    max_len = max_days * 24
    sensor_ids = population["index"]["sensor_id"].to_numpy()[row_ids]
    days = population["index"]["day"].to_numpy()[row_ids]

    feat = n_series_features(token_columns)
    static_dim = next(iter(sensor_arrays.values()))["category"].shape[0]
    padded = np.zeros((len(row_ids), max_len, feat), dtype=np.float32)
    padding_mask = np.ones((len(row_ids), max_len), dtype=bool)
    static = np.zeros((len(row_ids), static_dim), dtype=np.float32)

    for i, (row_id, sensor_id, day) in enumerate(zip(row_ids, sensor_ids, days)):
        arrays = sensor_arrays.get(sensor_id)
        result = None
        if arrays is not None:
            rng = np.random.default_rng(int(row_id) + seed)
            result = sample_measurement_window(arrays, pd.Timestamp(day), min_days, max_days, rng, token_columns=token_columns)
        if result is not None:
            series, stat = result
            series = scale_window(series, float(population["yscale"][row_id]), token_columns=token_columns).astype(np.float32)
            length = min(len(series), max_len)
            padded[i, :length] = series[:length]
            padding_mask[i, :length] = False
            static[i] = stat.astype(np.float32)
        else:
            padding_mask[i, 0] = False  # Nullfenster, wenn kein Messfenster verfügbar ist

    return torch.from_numpy(padded), torch.from_numpy(static), torch.from_numpy(padding_mask)


def _run_model(model, encoder, schedule, cond, series, static, padding_mask, device, x0_bounds) -> np.ndarray:
    """Reverse-Diffusion für einen Batch; Rückgabe in der normierten Skala (y >= 0)."""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        context, context_padding_mask = None, None
        if encoder is not None:
            series, static, padding_mask = series.to(device), static.to(device), padding_mask.to(device)
            context, context_padding_mask = encoder(series, static, padding_mask)
        out = dm.sample(model, schedule, cond, device, context, context_padding_mask, x0_min=x0_bounds[0], x0_max=x0_bounds[1])
    return (out + 1.0).clamp(min=0.0).float().numpy()  # Umkehrung von train_model._to_x0


def _sample_real(model, encoder, schedule, population, sensor_arrays, run_cfg, device, batch_size, seed, x0_bounds) -> np.ndarray:
    n = len(population["x"])
    out = np.zeros((n, 24), dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="real", unit="batch"):
        row_ids = np.arange(start, min(start + batch_size, n))
        cond = torch.from_numpy(population["x"][row_ids])
        series, static, padding_mask = _encode_batch(row_ids, population, sensor_arrays, run_cfg, seed)
        out[row_ids] = _run_model(model, encoder, schedule, cond, series, static, padding_mask, device, x0_bounds)
    return out


def _sample_synthetic(model, encoder, schedule, population, sensor_arrays, parent_row, depth, prev_col: int,
                      run_cfg, device, batch_size, seed, x0_bounds) -> np.ndarray:
    """Verkettete Generierung: prev_h23 wird durch den generierten Wert des Vortags ersetzt, sofern
    ein Vortag in der Kette existiert. Je Kettentiefe werden alle Zeilen gemeinsam generiert."""
    n = len(population["x"])
    h23_sampled = np.zeros(n, dtype=np.float32)
    out = np.zeros((n, 24), dtype=np.float32)

    pbar = tqdm(range(int(depth.max()) + 1 if n else 0), desc="synthetic (Kettentiefe)", unit="Tiefe")
    for d in pbar:
        rows_at_depth = np.flatnonzero(depth == d)
        pbar.set_postfix(zeilen=len(rows_at_depth))
        for start in range(0, len(rows_at_depth), batch_size):
            row_ids = rows_at_depth[start:start + batch_size]
            cond_np = population["x"][row_ids].copy()
            parents = parent_row[row_ids]
            has_parent = parents >= 0
            if has_parent.any():
                cond_np[has_parent, prev_col] = h23_sampled[parents[has_parent]]
            series, static, padding_mask = _encode_batch(row_ids, population, sensor_arrays, run_cfg, seed)
            y = _run_model(model, encoder, schedule, torch.from_numpy(cond_np), series, static, padding_mask, device, x0_bounds)
            out[row_ids] = y
            h23_sampled[row_ids] = y[:, 23]
    return out


def _cache_paths(version_suffix: str, checkpoint: str, variant: str, sensor_limit: int | None,
                 max_chain_depth: int | None, population: str, anchor_year: bool, chained: bool,
                 seed: int = 0, noise_seed: int | None = None, cluster_source: str = "true") -> tuple[Path, Path]:
    """Pfade (.npz, .json) des Sample-Caches; Benennung über run_config.cache_stem."""
    d = SAMPLES_DIR / version_suffix
    d.mkdir(parents=True, exist_ok=True)
    stem = d / cache_stem(variant, chained=chained, checkpoint=checkpoint, population=population,
                          max_chain_depth=max_chain_depth, anchor_year=anchor_year, seed=seed,
                          noise_seed=noise_seed, sensor_limit=sensor_limit, cluster_source=cluster_source)
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def _save_cache(npz_path: Path, json_path: Path, population: dict, y_sample: np.ndarray, run_cfg: dict,
                checkpoint_path: Path, variant: str, elapsed_s: float, args, anchor_year: bool,
                chained: bool, cluster_source: str) -> None:
    idx = population["index"]
    np.savez_compressed(
        npz_path,
        sensor_id=idx["sensor_id"].to_numpy(), day=idx["day"].to_numpy(), split=idx["split"].to_numpy(),
        y_sample=y_sample.astype(np.float32), y_real=population["y"].astype(np.float32),
    )
    meta = {
        "version_suffix": run_cfg["version_suffix"],
        "checkpoint": str(checkpoint_path),
        "variant": variant,
        "n_rows": int(len(idx)),
        "diffusion_steps": run_cfg["diffusion_steps"],
        "sensor_limit": args.sensor_limit,
        "chained": chained,
        "max_chain_depth": args.max_chain_depth if chained else None,
        "chain_anchor": ("year" if anchor_year else "none") if chained else None,
        "population": args.population,
        "window_seed": args.seed,
        "noise_seed": args.noise_seed,
        "cluster_source": cluster_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_s, 1),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=None, help="Config-Dateipfade (Default: alle aus configs/runs.yml)")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--variant", choices=["deployment", "oracle", "both"], default="both",
                        help="'deployment' (Einsatzfall) | 'oracle' (echter Vortageswert, nur mit "
                             "prev_h23) | 'both' (Default). Siehe Modul-Docstring.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sensor-limit", type=int, default=None, help="Nur die ersten N Sensoren - fuer Smoke-Tests")
    parser.add_argument("--population", choices=["all", "test", "val", "train"], default="test",
                        help="'test' (Default) = nur der Test-Split (ungesehene Gebaeude); 'val' = nur der "
                             "Validierungs-Split (Replikation auf disjunkten Sensoren); 'train' = nur der "
                             "Trainings-Split (dem Modell bekannte Gebaeude - z.B. um die Sensor-Population "
                             "einer Abbildung zu vergroessern, keine Generalisierungsaussage); 'all' = train+val+test.")
    parser.add_argument("--chain-anchor", choices=["none", "year"], default="year",
                        help="Nur bei verketteten Laeufen: 'year' (Default) startet jede Kette am Jahreswechsel neu, siehe Modul-Docstring.")
    parser.add_argument("--max-chain-depth", type=int, default=None,
                        help="Nur bei verketteten Laeufen: zusaetzlicher harter Resync auf den echten Vortageswert nach N Tagen.")
    parser.add_argument("--seed", type=int, default=0, help="Seed-Offset fuer die Messreihenfenster-Ziehung")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Seed fuer das Rauschen des Reverse-Diffusionsprozesses (dm.sample zieht aus dem "
                             "globalen torch-RNG). Default None = ungeseedet wie bisher. Zusammen mit --seed "
                             "trennbar: --seed variiert NUR das Encoder-Fenster, --noise-seed NUR das "
                             "Generierungsrauschen - Voraussetzung fuer die Varianzzerlegung.")
    parser.add_argument("--overwrite", action="store_true", help="Bereits gecachte Samples erneut erzeugen")
    parser.add_argument("--device", default=None, help="Default: cuda falls verfuegbar, sonst cpu")
    parser.add_argument("--cluster-source", choices=["true", "sampled"], default="true",
                        help="Nur bei Modellen mit `cluster` als Kovariate: 'true' (Default) = das wahre, "
                             "historisch beobachtete Cluster (wie im Training). 'sampled' = je Sensor EINMAL "
                             "aus cluster_proportions.csv gezogen - realistischer Kaltstart-Pfad, siehe "
                             "Modul-Docstring.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Anhang an version_suffix (Checkpoint- UND Cache-Pfad), analog zu "
                             "train_model.py --suffix - fuer Probelaeufe/Benchmarks, die den Checkpoint/Cache "
                             "eines vollen Laufs nicht ueberschreiben duerfen.")
    args = parser.parse_args()
    anchor_year = args.chain_anchor == "year"

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = list(CACHE_VARIANTS_ORDER) if args.variant == "both" else [args.variant]
    splits = ("train", "val", "test") if args.population == "all" else (args.population,)

    config_paths = [Path(p) for p in args.configs] if args.configs else load_run_index(Path("configs/runs.yml"))
    runs = [load_model_config(p) for p in config_paths]
    if args.suffix:
        for run in runs:
            run["version_suffix"] += args.suffix
    print(f"Device: {device} | Modelle: {len(runs)} | Varianten: {requested} | Checkpoint: {args.checkpoint} | Population: {args.population}")

    cluster_proportions = _load_cluster_proportions() if args.cluster_source == "sampled" else None
    population_cache: dict[str, dict] = {}
    chain_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    resampled_cluster_versions: set[str] = set()
    sensor_arrays = None

    for run_cfg in runs:
        version_suffix, version = run_cfg["version_suffix"], run_cfg["version"]

        available = cache_variants(run_cfg["covariate_columns"])
        chains = model_chains(run_cfg["covariate_columns"])
        has_cluster = "cluster" in run_cfg["covariate_columns"]
        cluster_source = args.cluster_source if has_cluster else "true"
        pending = []
        for variant in requested:
            if variant not in available:
                print(f"[skip] {version_suffix}/{variant}: Lauf ohne prev_h23 kennt nur {available}.")
                continue
            chained = variant == "deployment" and chains
            npz_path, json_path = _cache_paths(version_suffix, args.checkpoint, variant, args.sensor_limit,
                                               args.max_chain_depth, args.population, anchor_year, chained,
                                               args.seed, args.noise_seed, cluster_source)
            if npz_path.exists() and not args.overwrite:
                print(f"[skip] {version_suffix}/{variant} bereits gecacht ({npz_path})")
                continue
            pending.append((variant, chained, npz_path, json_path))
        if not pending:
            continue

        checkpoint_path = WEIGHTS_DIR / f"model_weights_{version_suffix}_{args.checkpoint}.pth"
        if not checkpoint_path.exists():
            print(f"[warn] Checkpoint fehlt, ueberspringe {version_suffix}: {checkpoint_path}")
            continue

        if version not in population_cache:
            print(f"Lade Population fuer version={version} ({args.population}) ...")
            population_cache[version] = _load_population(version, args.sensor_limit, splits)
        population = population_cache[version]
        prev_col = _prev_h23_column(run_cfg, population)
        x0_bounds = _load_x0_bounds(version)

        # measurement_windows.pkl wird für den Encoder und für das Cluster-Sampling (category) benötigt.
        if (run_cfg["encoder"] is not None or cluster_source == "sampled") and sensor_arrays is None:
            print("Lade measurement_windows.pkl ...")
            sensor_arrays = joblib.load(DATA_DIR / "measurement_windows.pkl")["sensor_arrays"]

        if cluster_source == "sampled" and version not in resampled_cluster_versions:
            _resample_cluster_column(population, run_cfg, sensor_arrays, cluster_proportions, args.seed)
            resampled_cluster_versions.add(version)

        if any(c for _, c, _, _ in pending) and version not in chain_cache:
            chain_cache[version] = _compute_chain_metadata(population["index"], args.max_chain_depth, anchor_year)

        try:
            n_static = next(iter(sensor_arrays.values()))["category"].shape[0] if sensor_arrays else 0
            series_features = n_series_features(run_cfg["token_columns"]) if run_cfg["encoder"] is not None else 0
            model, encoder = _build_model(run_cfg, population["x"].shape[1], n_static, series_features, device)
            _load_checkpoint(model, encoder, checkpoint_path, device)
            schedule = dm.DiffusionSchedule(timesteps=run_cfg["diffusion_steps"], schedule_type=run_cfg["noise_schedule"]).to(device)

            for variant, chained, npz_path, json_path in pending:
                label = f"{variant} (verkettet)" if chained else variant
                print(f"\n=== {version_suffix} | {label} | {len(population['x'])} Zeilen ===")
                assert not (chained and prev_col is None), \
                    f"{version_suffix}: verketteter Lauf ohne prev_h23-Spalte - run_config und Population passen nicht zusammen."
                t0 = time.time()
                # Rauschen je Variante mit --noise-seed initialisieren (falls gesetzt)
                if args.noise_seed is not None:
                    torch.manual_seed(args.noise_seed)
                if not chained:
                    y_sample = _sample_real(model, encoder, schedule, population, sensor_arrays, run_cfg, device,
                                            args.batch_size, args.seed, x0_bounds)
                else:
                    parent_row, depth = chain_cache[version]
                    y_sample = _sample_synthetic(model, encoder, schedule, population, sensor_arrays, parent_row, depth,
                                                 prev_col, run_cfg, device, args.batch_size, args.seed, x0_bounds)
                elapsed = time.time() - t0
                _save_cache(npz_path, json_path, population, y_sample, run_cfg, checkpoint_path, variant, elapsed,
                           args, anchor_year, chained, cluster_source)
                print(f"[done] {npz_path} ({elapsed:.1f}s)")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
