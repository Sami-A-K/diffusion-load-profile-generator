import traceback
import json
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm

import src.model.diffusion_model as dm
from src.training.measurement_window import sample_measurement_window, scale_window, n_series_features
from src.model.measurement_series_patch_encoder import MeasurementSeriesPatchEncoder
from src.training.run_config import load_model_config, resolve_runs


class MeasurementSeriesDataset(Dataset):
    """
    Liefert je Sample (x0, cond, series, static). x0 und cond stammen aus den x_/y_-Arrays,
    series und static werden je Sample als zufälliges Messfenster gezogen und skaliert.

    encoder=None: es wird kein Fenster gezogen, __getitem__ liefert nur (y, x).
    deterministic=True (Validierung): das Fenster wird mit einem aus idx abgeleiteten Seed
    gezogen und ist damit in jeder Epoche gleich.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, index_df: pd.DataFrame, sensor_arrays: dict,
        min_days: int, max_days: int,
        encoder: str | None, token_columns: list[str] | None = None, deterministic: bool = False,
        window_scales: np.ndarray | None = None,
    ):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.sensor_ids = index_df["sensor_id"].to_numpy()
        self.days = index_df["day"].to_numpy()
        # Jahresmittellast des Zieltags je Zeile, zur Skalierung des Lastkanals im Fenster.
        if encoder is not None and window_scales is None:
            raise ValueError("window_scales (yscale_<split>_<version>.npy) fehlt - ohne sie kann das Encoder-Fenster nicht skaliert werden.")
        self.window_scales = None if window_scales is None else window_scales.astype(np.float64)
        self.sensor_arrays = sensor_arrays
        self.min_days = min_days
        self.max_days = max_days
        self.encoder = encoder
        self.token_columns = token_columns
        self.deterministic = deterministic
        self.rng = np.random.default_rng()
        self.static_dim = next(iter(sensor_arrays.values()))["category"].shape[0] if encoder is not None else 0
        self.n_series_features = n_series_features(token_columns) if encoder is not None else 0

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.encoder is None:
            return torch.from_numpy(self.y[idx]), torch.from_numpy(self.x[idx])

        sensor_id = self.sensor_ids[idx]
        arrays = self.sensor_arrays.get(sensor_id)

        result = None
        if arrays is not None:
            rng = np.random.default_rng(idx) if self.deterministic else self.rng
            target_day = pd.Timestamp(self.days[idx])
            result = sample_measurement_window(
                arrays, target_day, self.min_days, self.max_days, rng, token_columns=self.token_columns,
            )

        if result is None:
            # Kein Fenster verfügbar: einzelne Nullzeile als Fenster.
            series = np.zeros((1, self.n_series_features), dtype=np.float32)
            static = np.zeros(self.static_dim, dtype=np.float32)
        else:
            series, static = result
            series = scale_window(series, float(self.window_scales[idx]), token_columns=self.token_columns).astype(np.float32)
            static = static.astype(np.float32)

        return torch.from_numpy(self.y[idx]), torch.from_numpy(self.x[idx]), torch.from_numpy(series), torch.from_numpy(static)


def _seed_worker(worker_id: int) -> None:
    """Setzt den Zufallsgenerator des Datasets in jedem DataLoader-Worker neu."""
    seed = torch.initial_seed() % (2**32)
    get_worker_info().dataset.rng = np.random.default_rng(seed)


def collate_measurement_batch(batch, max_len: int):
    """
    Füllt alle Fenster auf die feste Länge max_len (max_days*24 Stunden) auf und erzeugt die
    Padding-Maske.
    """
    y_batch, cond_batch, series_list, static_list = zip(*batch)
    y_batch = torch.stack(y_batch)
    cond_batch = torch.stack(cond_batch)
    static_batch = torch.stack(static_list)

    n_series_features = series_list[0].shape[1]

    padded = torch.zeros(len(series_list), max_len, n_series_features)
    padding_mask = torch.ones(len(series_list), max_len, dtype=torch.bool)
    for i, s in enumerate(series_list):
        padded[i, : s.shape[0]] = s
        padding_mask[i, : s.shape[0]] = False

    return y_batch, cond_batch, padded, static_batch, padding_mask


def collate_plain_batch(batch):
    """Collate für Modelle ohne Encoder: batch ist eine Liste von (y, x)."""
    y_batch, cond_batch = zip(*batch)
    return torch.stack(y_batch), torch.stack(cond_batch), None, None, None


def _to_x0(y_norm: np.ndarray) -> np.ndarray:
    """
    Überführt das normierte Y (1.0 = Jahresmittellast) in den x0-Raum: x0 = y - 1.
    0 kWh entspricht damit x0 = -1, der Mittelwert liegt bei 0.
    """
    return y_norm - 1


def load_training_artifacts(data_dir: Path, version: str, needs_windows: bool) -> dict:
    """
    Lädt X/Y/Index einer Datenversion. Bei needs_windows=True zusätzlich
    measurement_windows.pkl und die Fensterskalen.
    """
    result = {
        "conditions_train": np.load(data_dir / f"x_train_{version}.npy"),
        "load_profiles_train": _to_x0(np.load(data_dir / f"y_train_{version}.npy")),
        "conditions_val": np.load(data_dir / f"x_val_{version}.npy"),
        "load_profiles_val": _to_x0(np.load(data_dir / f"y_val_{version}.npy")),
        "index_train": pd.read_csv(data_dir / f"index_train_{version}.csv", parse_dates=["day"]),
        "index_val": pd.read_csv(data_dir / f"index_val_{version}.csv", parse_dates=["day"]),
        "window_scales_train": None, "window_scales_val": None,
    }

    if needs_windows:
        result["sensor_arrays"] = joblib.load(data_dir / "measurement_windows.pkl")["sensor_arrays"]
        result["window_scales_train"] = np.load(data_dir / f"yscale_train_{version}.npy")
        result["window_scales_val"] = np.load(data_dir / f"yscale_val_{version}.npy")
    else:
        result["sensor_arrays"] = {}

    return result


def _encode_context(encoder, encoder_type: str | None, series, static, padding_mask):
    """Kontext-Tokens und Padding-Maske des Encoders; (None, None) für Modelle ohne Encoder."""
    if encoder_type is None:
        return None, None
    return encoder(series, static, padding_mask)


def _to_device(x, device):
    return x.to(device, non_blocking=True) if x is not None else None


_VAL_NOISE_PRECOMPUTE_MAX_ELEMENTS = 20_000_000  # Obergrenze für vorab erzeugtes Validierungsrauschen

# Feste Seeds für Zeitschritte und Rauschen der Validierung.
_VAL_PERM_SEED = 0
_VAL_NOISE_SEED = 1


def _deterministic_val_targets(n_val: int, seq_len: int, timesteps: int, device):
    """
    Feste (t, eps) je Validierungs-Sample, in allen Epochen gleich, damit val_loss zwischen
    Epochen vergleichbar ist. t ist gleichmäßig über [0, T) verteilt und permutiert.

    Returns: (t_fixed [n_val], eps_fixed). eps_fixed ist ein Tensor [n_val, seq_len] oder bei
    großen Splits eine Funktion (offset, count) -> Tensor[count, seq_len].
    """
    idx = np.arange(n_val)
    t = np.minimum((idx * timesteps) // max(n_val, 1), timesteps - 1)
    perm = np.random.default_rng(_VAL_PERM_SEED).permutation(n_val)
    t = t[perm]
    t_fixed = torch.from_numpy(t.astype(np.int64)).to(device)

    if n_val * seq_len <= _VAL_NOISE_PRECOMPUTE_MAX_ELEMENTS:
        gen = torch.Generator().manual_seed(_VAL_NOISE_SEED)
        eps_fixed = torch.randn(n_val, seq_len, generator=gen).to(device)
        return t_fixed, eps_fixed

    def eps_for_slice(offset: int, count: int) -> torch.Tensor:
        out = torch.empty(count, seq_len)
        for j in range(count):
            g = torch.Generator().manual_seed(_VAL_NOISE_SEED + offset + j)
            out[j] = torch.randn(seq_len, generator=g)
        return out.to(device)

    return t_fixed, eps_for_slice


class EMA:
    """
    Exponential Moving Average der Modellgewichte (Ho et al. 2020). Validierung und
    Checkpoints verwenden die EMA-Gewichte, das Training die aktuellen Gewichte.
    """

    def __init__(self, params: list[torch.nn.Parameter], decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]
        self._backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, params: list[torch.nn.Parameter]) -> None:
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self, params: list[torch.nn.Parameter]) -> None:
        """Ersetzt die Trainingsgewichte durch die EMA-Gewichte und sichert die Trainingsgewichte."""
        self._backup = [p.detach().clone() for p in params]
        for p, s in zip(params, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def swap_out(self, params: list[torch.nn.Parameter]) -> None:
        """Stellt die von swap_in() gesicherten Trainingsgewichte wieder her."""
        for p, b in zip(params, self._backup):
            p.data.copy_(b)
        self._backup = None


def build_run(data: dict, run_cfg: dict) -> None:
    """
    Trainiert Encoder und Denoiser (bzw. nur den Denoiser bei encoder=None) für eine Config und
    speichert Gewichte und Trainingshistorie unter run_cfg["version_suffix"].
    """
    encoder_type = run_cfg["encoder"]
    denoiser_cfg = run_cfg["denoiser"]
    version_suffix = run_cfg["version_suffix"]
    seq_len = run_cfg["seq_len"]

    seed = run_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    min_days = run_cfg["measurement_min_days"]
    max_days = run_cfg["measurement_max_days"]
    token_columns = run_cfg["token_columns"]
    train_dataset = MeasurementSeriesDataset(
        data["conditions_train"], data["load_profiles_train"], data["index_train"], data["sensor_arrays"],
        min_days, max_days, encoder=encoder_type, token_columns=token_columns, deterministic=False,
        window_scales=data["window_scales_train"],
    )
    val_dataset = MeasurementSeriesDataset(
        data["conditions_val"], data["load_profiles_val"], data["index_val"], data["sensor_arrays"],
        min_days, max_days, encoder=encoder_type, token_columns=token_columns, deterministic=True,
        window_scales=data["window_scales_val"],
    )

    if encoder_type is None:
        collate_fn = collate_plain_batch
        num_workers, val_num_workers = 0, 0
    else:
        max_window_hours = max_days * 24
        collate_fn = partial(collate_measurement_batch, max_len=max_window_hours)
        num_workers, val_num_workers = 2, 1

    train_dataloader = DataLoader(
        train_dataset, batch_size=run_cfg["batch_size"], shuffle=True, collate_fn=collate_fn, pin_memory=True,
        drop_last=True, num_workers=num_workers, persistent_workers=(num_workers > 0),
        worker_init_fn=(_seed_worker if num_workers > 0 else None),
    )
    # Feste Reihenfolge und alle Samples, passend zu den festen (t, eps) der Validierung.
    val_dataloader = DataLoader(
        val_dataset, batch_size=run_cfg["batch_size"], shuffle=False, collate_fn=collate_fn, pin_memory=True,
        drop_last=False, num_workers=val_num_workers, persistent_workers=(val_num_workers > 0),
        worker_init_fn=(_seed_worker if val_num_workers > 0 else None),
    )

    requested_device = run_cfg.get("device")
    if requested_device is not None:
        device = torch.device(requested_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    n_features = data["conditions_train"].shape[1]
    n_static_features = train_dataset.static_dim

    model = dm.ConditionedDenoiser(seq_len=seq_len, n_features=n_features, **denoiser_cfg).to(device)

    encoder = None
    if encoder_type == "patch":
        encoder = MeasurementSeriesPatchEncoder(
            n_series_features=train_dataset.n_series_features, n_static_features=n_static_features, **run_cfg["encoder_params"],
        ).to(device)

    params = list(model.parameters()) + (list(encoder.parameters()) if encoder is not None else [])
    optimizer = torch.optim.Adam(params, lr=run_cfg["learning_rate"])
    schedule = dm.DiffusionSchedule(timesteps=run_cfg["diffusion_steps"], schedule_type=run_cfg["noise_schedule"]).to(device)

    # Mixed Precision: bfloat16 ohne Loss-Scaling, falls unterstützt, sonst float16 mit GradScaler.
    bf16_supported = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    run_cfg["bf16_supported"] = bf16_supported
    run_cfg["amp_dtype"] = "bfloat16" if bf16_supported else "float16"
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and not bf16_supported))
    ema = EMA(params)

    val_t_fixed, val_eps_fixed = _deterministic_val_targets(
        len(val_dataset), seq_len, schedule.timesteps, device,
    )

    epochs = run_cfg["epochs"]
    print(f"\n=== {version_suffix}: encoder={encoder_type}, denoiser={denoiser_cfg}, epochs={epochs}, device={device} ===")

    list_avg_loss = []
    list_val_loss = []
    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        if encoder is not None:
            encoder.train()

        pbar = tqdm(train_dataloader, desc=f"[{version_suffix}] Epoch {epoch+1}/{epochs}", leave=True)
        for batch_idx, (x0_batch, cond_batch, series_batch, static_batch, padding_mask) in enumerate(pbar, start=1):
            x0_batch = x0_batch.to(device, non_blocking=True)
            cond_batch = cond_batch.to(device, non_blocking=True)
            series_batch = _to_device(series_batch, device)
            static_batch = _to_device(static_batch, device)
            padding_mask = _to_device(padding_mask, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                context, context_padding_mask = _encode_context(encoder, encoder_type, series_batch, static_batch, padding_mask)

            loss = dm.train_step(
                model, schedule, optimizer, x0_batch, cond_batch, scaler, context, context_padding_mask, amp_dtype,
            )
            total_loss += loss
            ema.update(params)

            pbar.set_postfix(batch=f"{batch_idx}/{len(train_dataloader)}", loss=f'{(total_loss/batch_idx):.4f}')

        avg_loss = total_loss / len(train_dataloader)
        list_avg_loss.append(avg_loss)
        pbar.set_description(f"[{version_suffix}] Epoch {epoch+1}/{epochs} | avg loss: {avg_loss:.4f}")

        # Validierung und Checkpoint mit EMA-Gewichten
        ema.swap_in(params)
        model.eval()
        if encoder is not None:
            encoder.eval()
        val_loss = 0.0
        val_offset = 0
        with torch.no_grad():
            for x0, cond, series, static, padding_mask in tqdm(val_dataloader, desc=f"[{version_suffix}] Val {epoch+1}/{epochs}"):
                x0 = x0.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                series = _to_device(series, device)
                static = _to_device(static, device)
                padding_mask = _to_device(padding_mask, device)

                batch_size_ = x0.size(0)
                t = val_t_fixed[val_offset: val_offset + batch_size_]
                noise = (
                    val_eps_fixed(val_offset, batch_size_) if callable(val_eps_fixed)
                    else val_eps_fixed[val_offset: val_offset + batch_size_]
                )
                val_offset += batch_size_

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                    context, context_padding_mask = _encode_context(encoder, encoder_type, series, static, padding_mask)

                    xt = dm.q_sample(schedule, x0, t, noise)
                    noise_pred = model(xt, t, cond, context, context_padding_mask)
                    loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item() * noise.size(0)

        val_loss /= len(val_dataloader.dataset)
        list_val_loss.append(val_loss)

        print(f'[{version_suffix}] Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                {"model": model.state_dict(), "encoder": encoder.state_dict() if encoder is not None else None},
                f'src/model/model_weights/model_weights_{version_suffix}_best.pth',
            )

        ema.swap_out(params)

    ema.swap_in(params)
    torch.save(
        {"model": model.state_dict(), "encoder": encoder.state_dict() if encoder is not None else None},
        f'src/model/model_weights/model_weights_{version_suffix}_last.pth',
    )
    ema.swap_out(params)

    # run_config und Feature-Anzahlen werden gespeichert, damit die Auswertung das Modell
    # rekonstruieren kann.
    history = {
        "train_loss": list_avg_loss,
        "val_loss": list_val_loss,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "run_config": run_cfg,
        "n_features": n_features,
        "n_static_features": n_static_features,
    }
    with open(f'src/model/model_weights/training_history_{version_suffix}.json', 'w') as f:
        json.dump(history, f, indent=2)


def main(config_paths: list[Path] | None = None, epochs: int | None = None, suffix: str | None = None):
    """
    config_paths: einzelne Config-Dateien statt configs/runs.yml.
    epochs: überschreibt training.epochs.
    suffix: Anhang an version_suffix für Checkpoint und Historie.
    """
    data_dir = Path("data/processed/openmeter/training_data")
    runs = ([load_model_config(p) for p in config_paths] if config_paths
            else resolve_runs(Path("configs/runs.yml")))
    for run in runs:
        if epochs is not None:
            run["epochs"] = epochs
        if suffix:
            run["version_suffix"] += suffix

    # Trainingsdaten einmal je Datenversion laden
    runs_by_version: dict[str, list[dict]] = {}
    for run in runs:
        runs_by_version.setdefault(run["version"], []).append(run)

    data_cache = {
        v: load_training_artifacts(data_dir, v, needs_windows=any(r["encoder"] is not None for r in version_runs))
        for v, version_runs in runs_by_version.items()
    }

    for run in runs:
        try:
            build_run(data_cache[run["version"]], run)
        except Exception:
            print(f"Lauf {run['version_suffix']} fehlgeschlagen, fahre mit dem naechsten fort:")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trainiert die Modelle aus configs/runs.yml.")
    parser.add_argument(
        "configs", nargs="*", type=Path,
        help="Optional: einzelne Modell-Config-Dateien statt configs/runs.yml (Probelaeufe).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Uebersteuert training.epochs - fuer kurze Diagnoselaeufe.",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Anhang an version_suffix, damit ein Probelauf Checkpoint/History des vollen Laufs "
             "nicht ueberschreibt (z.B. --suffix _probe).",
    )
    args = parser.parse_args()
    main(config_paths=args.configs or None, epochs=args.epochs, suffix=args.suffix)
