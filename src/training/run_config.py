"""
Laden der Modell-Configs: eine Datei je Modell (Schema siehe configs/config_main.yml), gelistet
in configs/runs.yml. Zusätzlich die Benennung der Sample-Caches.
"""
from pathlib import Path

import yaml


def load_run_index(path: Path) -> list[Path]:
    """Lädt configs/runs.yml - eine Liste von Config-Pfaden relativ zum Repo-Root."""
    with open(path, encoding="utf-8") as f:
        index = yaml.safe_load(f)
    return [Path(p) for p in index["configs"]]


def _active_names(flags: dict[str, bool]) -> list[str]:
    """{name: True/False} -> Liste der aktiven Namen."""
    return [name for name, active in flags.items() if active]


_EXPECTED_TOKEN_COLUMNS = {"hourly_load", "hourly_time", "hourly_weather"}

# Nicht mehr unterstützte Config-Schlüssel; ihr Vorkommen führt zu einem Fehler.
_REMOVED_TRAINING_KEYS = {"y_normalization", "sensor_year_filter", "prediction_type", "diff_loss_weight",
                          "learn_variance", "vlb_weight", "min_snr_gamma", "n_bins", "val_seed",
                          "val_noise_seed"}
_REMOVED_ENCODER_KEYS = {"measurement_same_year", "token_mixing", "norm_first"}
_REMOVED_DENOISER_KEYS = {"cross_attn_depth", "cross_attn_position", "n_cross_attn", "n_blocks"}


def load_model_config(path: Path) -> dict:
    """
    Lädt eine Modell-Config. version_suffix ergibt sich aus dem Dateinamen
    (config_<name>.yml -> <name>) und benennt Checkpoints und Historie; version benennt die
    Trainingsdaten und kann von mehreren Configs geteilt werden.

    encoder.type ist null oder "patch"; bei null wird cross_attn deaktiviert.

    Returns: dict mit version, version_suffix, split, covariate_columns, denoiser, encoder,
    encoder_params, token_columns, measurement_min_days, measurement_max_days sowie allen
    Schlüsseln aus `training:`.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    stem = path.stem
    version_suffix = stem[len("config_"):] if stem.startswith("config_") else stem

    denoiser = dict(raw["model"]["denoiser"])
    _reject_removed(path, "model.denoiser", denoiser, _REMOVED_DENOISER_KEYS)
    covariate_columns = _active_names(denoiser.pop("covariate_columns"))

    encoder_raw = dict(raw["model"]["encoder"])
    _reject_removed(path, "model.encoder", encoder_raw, _REMOVED_ENCODER_KEYS)
    encoder_type = encoder_raw.pop("type")
    if encoder_type not in (None, "patch"):
        raise ValueError(f"{path}: model.encoder.type={encoder_type!r} - erlaubt sind nur null oder \"patch\".")
    token_columns_flags = encoder_raw.pop("token_columns", {})
    measurement_min_days = encoder_raw.pop("measurement_min_days")
    measurement_max_days = encoder_raw.pop("measurement_max_days")
    if encoder_type != "patch":
        encoder_raw.pop("patch_len", None)
        encoder_raw.pop("stride", None)
    encoder_params = encoder_raw

    token_columns = _active_names(token_columns_flags)
    if encoder_type is not None:
        active_tokens = set(token_columns)
        if not active_tokens:
            raise ValueError(
                f"{path}: token_columns - mindestens ein Kanal muss aktiv sein, sonst haette der "
                f"Encoder kein Input-Fenster."
            )
        if not active_tokens <= _EXPECTED_TOKEN_COLUMNS:
            raise ValueError(
                f"{path}: token_columns={active_tokens} enthaelt unbekannte Kanaele - erlaubt sind "
                f"nur: {_EXPECTED_TOKEN_COLUMNS}."
            )

    if encoder_type is None:
        denoiser["cross_attn"] = False

    training = dict(raw["training"])
    _reject_removed(path, "training", training, _REMOVED_TRAINING_KEYS)
    training.setdefault("noise_schedule", "linear")
    if training["noise_schedule"] != "linear":
        raise NotImplementedError(
            f"{path}: noise_schedule={training['noise_schedule']!r} - dm.DiffusionSchedule "
            f"unterstuetzt nur \"linear\" (Ho et al. 2020)."
        )
    return {
        "version": raw["version"],
        "version_suffix": version_suffix,
        "split": raw["split"],
        "covariate_columns": covariate_columns,
        "denoiser": denoiser,
        "encoder": encoder_type,
        "encoder_params": encoder_params,
        "token_columns": token_columns,
        "measurement_min_days": measurement_min_days,
        "measurement_max_days": measurement_max_days,
        **training,
    }


def _reject_removed(path: Path, section: str, block: dict, removed: set[str]) -> None:
    present = sorted(removed & set(block))
    if present:
        raise ValueError(
            f"{path}: {section} enthaelt {present} - diese Schalter existieren nicht mehr (die Pipeline "
            f"ist auf die Jahresmittellast-Normierung und den statischen Cross-Attention-Block festgelegt, "
            f"siehe README). Zeilen entfernen."
        )


def _validate_runs(runs: list[dict]) -> None:
    """
    Prüft die Configs gegeneinander: alle nutzen denselben Split, und Configs mit gleicher
    version haben identische covariate_columns.
    """
    if not runs:
        return

    splits = {tuple(sorted(r["split"].items())) for r in runs}
    if len(splits) != 1:
        raise ValueError(
            f"configs/runs.yml: nicht alle Modell-Configs haben denselben split - "
            f"gefunden: {[dict(s) for s in splits]}. Fuer einen fairen Vergleich muessen alle "
            f"Modelle auf derselben Sensor-Population (Split) trainiert/evaluiert werden."
        )

    by_version: dict[str, dict] = {}
    for run in runs:
        v = run["version"]
        other = by_version.setdefault(v, run)
        if other is run:
            continue
        if run["covariate_columns"] != other["covariate_columns"]:
            raise ValueError(
                f"Configs '{run['version_suffix']}' und '{other['version_suffix']}' teilen sich "
                f"version='{v}', haben aber unterschiedliche covariate_columns - wuerde zu falsch "
                f"gecachten Trainingsdaten fuehren. Entweder version-Werte trennen oder die Felder angleichen."
            )


def resolve_runs(index_path: Path) -> list[dict]:
    """Lädt und prüft alle in configs/runs.yml gelisteten Configs."""
    runs = [load_model_config(p) for p in load_run_index(index_path)]
    _validate_runs(runs)
    return runs


# --- Benennung der Sample-Caches ---------------------------------------------------------------
#
#   "deployment" - Einsatzfall: mit prev_h23 über das Kalenderjahr verkettet, ohne prev_h23
#                  unabhängige Tage.
#   "oracle"     - unabhängige Tage mit echtem Vortageswert als prev_h23; nur für Modelle mit
#                  prev_h23.
CACHE_VARIANTS = ("deployment", "oracle")


def model_chains(covariate_columns: list[str]) -> bool:
    """Ob das Modell autoregressiv verkettet werden kann - nur mit prev_h23 als Kovariate."""
    return "prev_h23" in covariate_columns


def cache_variants(covariate_columns: list[str]) -> tuple[str, ...]:
    """Verfügbare Cache-Varianten für dieses Modell."""
    return CACHE_VARIANTS if model_chains(covariate_columns) else ("deployment",)


def cache_stem(variant: str, *, chained: bool, checkpoint: str = "best", population: str = "test",
               max_chain_depth: int | None = None, anchor_year: bool = True, seed: int = 0,
               noise_seed: int | None = None, sensor_limit: int | None = None,
               cluster_source: str = "true") -> str:
    """Dateiname eines Sample-Caches (ohne Verzeichnis und Endung).

    Alle ergebnisrelevanten Optionen sind im Namen enthalten; seed/noise_seed nur bei Abweichung
    vom Default. max_chain_depth und anchor_year gelten nur bei verketteter Generierung (chained).
    cluster_source: "true" = beobachtetes Cluster des Sensors, "sampled" = je Sensor aus
    cluster_proportions.csv gezogen.
    """
    if variant not in CACHE_VARIANTS:
        raise ValueError(f"Unbekannte Cache-Variante {variant!r} (erwartet {CACHE_VARIANTS})")
    if chained and variant != "deployment":
        raise ValueError(f"chained=True ist nur fuer 'deployment' sinnvoll, nicht fuer {variant!r}")
    name = f"{checkpoint}_{variant}"
    if chained and max_chain_depth is not None:
        name += f"_maxdepth{max_chain_depth}"
    if chained and anchor_year:
        name += "_yearanchor"
    if population != "all":
        name += f"_{population}"
    if seed != 0:
        name += f"_wseed{seed}"
    if noise_seed is not None:
        name += f"_nseed{noise_seed}"
    if sensor_limit is not None:
        name += f"_limit{sensor_limit}"
    if cluster_source != "true":
        name += f"_cluster{cluster_source}"
    return name
