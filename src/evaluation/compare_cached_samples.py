"""
Wertet die gecachten Samples unter data/samples/<version_suffix>/ (aus sample_population.py) aus.

Je Checkpoint und Cache-Variante ("deployment", bei Modellen mit prev_h23 zusätzlich "oracle")
werden je Test-Sensor berechnet:
  - Zeitkontext-Fehler (evaluation_metrics.duration_curve_error_per_sensor) mit Zerlegung in
    global und Struktur.
  - Tagesgrenzen-Sprungfaktor (day_boundary_error_per_sensor).
Alle Werte in der normierten Skala (1e-2 = ein Prozent der Jahresmittellast).

Aufruf (Repo-Root): python -m src.evaluation.compare_cached_samples [--versions main ...]
Ausgabe unter results/: cached_eval_summary.csv (eine Zeile je Checkpoint und Variante),
cached_eval_by_variant.csv (Mittel/SD über die Trainings-Seeds je Architekturvariante),
cached_eval_per_sensor.csv, cached_eval_per_context.csv.
"""
import argparse
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluation_metrics import duration_curve_error_per_sensor, day_boundary_error_per_sensor
from src.training.run_config import CACHE_VARIANTS, cache_stem, cache_variants, load_model_config, model_chains

SAMPLES_DIR = Path("data/samples")
RESULTS_DIR = Path("results")
CONFIG_DIR = Path("configs")
SENSOR_TABLE = Path("data/processed/openmeter/sensor_table.parquet")
VARIANTS = CACHE_VARIANTS


@lru_cache(maxsize=None)
def run_config(version_suffix: str) -> dict:
    """Lädt die Config zu einem Cache-Verzeichnis."""
    path = CONFIG_DIR / f"config_{version_suffix}.yml"
    if not path.exists():
        raise SystemExit(f"Keine Config zu data/samples/{version_suffix}/ gefunden ({path}) - "
                         f"Cache-Verzeichnis und configs/ sind auseinandergelaufen.")
    return load_model_config(path)


def available_variants(version_suffix: str) -> tuple[str, ...]:
    return cache_variants(run_config(version_suffix)["covariate_columns"])


def variant_stem(version_suffix: str, variant: str, population: str = "test", cache_tag: str = "") -> str:
    """Dateiname eines Caches; cache_tag (z.B. "_nseed1") wird angehängt."""
    chained = variant == "deployment" and model_chains(run_config(version_suffix)["covariate_columns"])
    return cache_stem(variant, chained=chained, population=population) + cache_tag

# Namensschema von version_suffix:
#   main[_<ablation>]                   - Hauptmodell (Patch 8 h, Seed 9) und Ablationen
#   patch_<K>h[_seed<NN>]               - Patch-Längen, ohne Seed-Suffix Seed 9
#   baseline[_h23|_category][_seed<NN>] - Modell ohne Encoder (Vorgänger)
MAIN_RE = re.compile(r"^main(?:_(\w+))?$")
PATCH_RE = re.compile(r"^patch_(\d+)h(?:_seed(\d+))?$")
BASELINE_RE = re.compile(r"^baseline(?:_(h23|category))?(?:_seed(\d+))?$")

# main-Suffix -> (patch, arch); arch == "base" kennzeichnet den Patch-Längen-Vergleich.
_MAIN_ARCH = {
    None: ("p8", "base"),                   # Hauptmodell
    "noenc": ("noenc", "base"),             # ohne Encoder
    "l3": ("p8", "l3"),                     # 3 Encoder-Layer
    "nocat": ("p8", "nocat"),               # ohne Kategorie
    "cluster": ("p8", "cluster"),           # Cluster statt Kategorie
    "all_covariates": ("p8", "all_cov"),    # Kategorie und Cluster
    "noh23": ("p8", "noh23"),               # ohne prev_h23
}


def parse_version_suffix(version_suffix: str) -> dict:
    m = MAIN_RE.match(version_suffix)
    if m:
        arch_suffix = m.group(1)
        if arch_suffix not in _MAIN_ARCH:
            raise ValueError(f"Unbekannte main-Ablation {version_suffix!r} - in _MAIN_ARCH eintragen.")
        patch, arch = _MAIN_ARCH[arch_suffix]
        seed, variant_tag = 9, (patch if arch == "base" else f"{patch}_{arch}")
    elif (m := PATCH_RE.match(version_suffix)):
        patch, arch = f"p{m.group(1)}", "base"
        seed, variant_tag = int(m.group(2)) if m.group(2) else 9, patch
    elif (m := BASELINE_RE.match(version_suffix)):
        patch = "noenc"
        arch = {"h23": "baseline_h23", "category": "baseline_category"}.get(m.group(1), "baseline")
        seed, variant_tag = int(m.group(2)) if m.group(2) else 9, arch
    else:
        raise ValueError(f"Unerwartetes version_suffix-Schema: {version_suffix!r} (erwartet "
                         f"'main'[_<ablation>], 'patch_<K>h'[_seed<NN>] oder 'baseline'[_h23][_seed<NN>])")
    return {"seed": seed, "patch": patch, "arch": arch, "model_variant": variant_tag}


def discover_version_suffixes() -> list[str]:
    return sorted(p.name for p in SAMPLES_DIR.iterdir()
                  if p.is_dir() and (MAIN_RE.match(p.name) or PATCH_RE.match(p.name)
                                     or BASELINE_RE.match(p.name)))


def load_cache(version_suffix: str, variant: str, population: str = "test", cache_tag: str = "") -> dict | None:
    path = SAMPLES_DIR / version_suffix / f"{variant_stem(version_suffix, variant, population, cache_tag)}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        cache = {k: data[k] for k in data.files}
    mask = cache["split"] == population
    return {k: v[mask] for k, v in cache.items()}


@lru_cache(maxsize=1)
def _holiday_table() -> pd.Series:
    return pd.read_parquet(SENSOR_TABLE, columns=["is_holiday"])["is_holiday"]


def holiday_flags(sensor_id: np.ndarray, day: np.ndarray) -> np.ndarray:
    """is_holiday je Zeile aus der Sensor-Tabelle."""
    table = _holiday_table()
    idx = pd.MultiIndex.from_arrays([sensor_id, pd.DatetimeIndex(day)], names=table.index.names)
    flags = table.reindex(idx)
    assert flags.notna().all(), f"{int(flags.isna().sum())} Zeilen ohne is_holiday in der Sensor-Tabelle"
    return flags.to_numpy().astype(int)


def evaluate(version_suffix: str, variant: str, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per_sensor, per_context) für einen Cache, inklusive Sprungfaktor je Sensor."""
    holiday = holiday_flags(cache["sensor_id"], cache["day"])
    per_sensor, per_context = duration_curve_error_per_sensor(cache["y_real"], cache["y_sample"], cache["sensor_id"], cache["day"], holiday)
    jumps = day_boundary_error_per_sensor(cache["y_real"], cache["y_sample"], cache["sensor_id"], cache["day"])
    per_sensor = per_sensor.merge(jumps[["sensor_id", "n_transitions", "jump_factor"]], on="sensor_id", how="left")
    meta = {"version_suffix": version_suffix, "variant": variant, **parse_version_suffix(version_suffix)}
    return per_sensor.assign(**meta), per_context.assign(**meta)


def summarize(per_sensor: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je (Checkpoint, Variante). Der Sprungfaktor wird als Median berichtet;
    unendliche Werte sind ausgeschlossen."""
    rows = []
    for (vs, variant), g in per_sensor.groupby(["version_suffix", "variant"], sort=True):
        jf = g["jump_factor"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "version_suffix": vs, "variant": variant, **parse_version_suffix(vs),
            "n_sensors": len(g), "n_days": int(g["n_days"].sum()),
            "error_context_mean": g["error_context"].mean(), "error_context_median": g["error_context"].median(),
            "error_global_mean": g["error_global"].mean(), "error_structure_mean": g["error_structure"].mean(),
            "error_hour_mean": g["error_hour"].mean(),
            "jump_factor_median": jf.median(), "jump_factor_mean": jf.mean(),
            "share_sensors_jump_gt_1_5": float((jf > 1.5).mean()),
        })
    return pd.DataFrame(rows)


def aggregate_seeds(summary: pd.DataFrame) -> pd.DataFrame:
    """Mittel und Standardabweichung über die Trainings-Seeds je Architektur- und Cache-Variante."""
    return summary.groupby(["model_variant", "variant"]).agg(
        n_seeds=("seed", "nunique"),
        error_context_mean=("error_context_mean", "mean"), error_context_sd=("error_context_mean", "std"),
        error_global_mean=("error_global_mean", "mean"), error_structure_mean=("error_structure_mean", "mean"),
        jump_factor_median=("jump_factor_median", "mean"),
    ).reset_index().sort_values(["variant", "error_context_mean"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="*", default=None, help="version_suffix-Liste (Default: alle unter data/samples/)")
    ap.add_argument("--population", choices=["test", "val"], default="test",
                    help="Welcher Split-Cache ausgewertet wird (Default test).")
    ap.add_argument("--cache-tag", default="",
                    help="Namenssuffix der Caches, z.B. '_nseed1' oder '_wseed1' (siehe sample_population).")
    ap.add_argument("--out-prefix", default="cached_eval",
                    help="Praefix der CSV-Ausgaben unter results/ - abweichend setzen, um die Hauptergebnisse "
                         "nicht zu ueberschreiben.")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    versions = args.versions or discover_version_suffixes()
    print(f"{len(versions)} Versionen: {versions}")

    sensor_tables, context_tables = [], []
    for vs in versions:
        for variant in available_variants(vs):
            cache = load_cache(vs, variant, args.population, args.cache_tag)
            if cache is None:
                mark = "FEHLT" if variant == "deployment" else "fehlt"
                print(f"  [{mark}] {vs}/{variant}")
                continue
            ps, pc = evaluate(vs, variant, cache)
            sensor_tables.append(ps)
            context_tables.append(pc)
            jf = ps["jump_factor"].replace([np.inf, -np.inf], np.nan).median()
            print(f"  {vs:<20} {variant:<9} Zeitkontext-Fehler {ps['error_context'].mean()*1e2:6.2f}%  "
                  f"global {ps['error_global'].mean()*1e2:6.2f}  Struktur {ps['error_structure'].mean()*1e2:6.2f}  "
                  f"Jump-Factor (Median) {jf:5.2f}  ({len(ps)} Sensoren)")

    if not sensor_tables:
        raise SystemExit(f"Kein Cache gefunden (population={args.population!r}, cache_tag={args.cache_tag!r}) - "
                         f"erst sample_population mit den passenden Optionen laufen lassen.")
    per_sensor = pd.concat(sensor_tables, ignore_index=True)
    per_context = pd.concat(context_tables, ignore_index=True)
    summary = summarize(per_sensor)
    by_variant = aggregate_seeds(summary)

    pre = args.out_prefix
    per_sensor.to_csv(RESULTS_DIR / f"{pre}_per_sensor.csv", index=False)
    per_context.to_csv(RESULTS_DIR / f"{pre}_per_context.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"{pre}_summary.csv", index=False)
    by_variant.to_csv(RESULTS_DIR / f"{pre}_by_variant.csv", index=False)

    print("\nMittel/SD ueber die Trainings-Seeds je Architekturvariante [W1 in %]:")
    show = by_variant.copy()
    for c in ("error_context_mean", "error_context_sd", "error_global_mean", "error_structure_mean"):
        show[c] = show[c] * 1e2
    print(show.round(3).to_string(index=False))
    print(f"\ngeschrieben: {RESULTS_DIR}/{pre}_*.csv")


if __name__ == "__main__":
    main()
