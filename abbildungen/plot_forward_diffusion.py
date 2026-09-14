"""
Vorwärts-Diffusionsprozess an einem Test-Tagesprofil: Zustand x_t bei ausgewählten
Diffusionsschritten (t=0 reales Profil bis t=T Rauschen).

Die Markov-Kette wird schrittweise simuliert (x_i = sqrt(alpha_i) * x_{i-1} + sqrt(beta_i) * z_i
mit neuem Rauschen in jedem Schritt).

    python -m abbildungen.plot_forward_diffusion
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import src.evaluation.compare_models as cm
import src.model.diffusion_model as dm
from src.training.train_model import _to_x0
from abbildungen.plot_style import (
    GENERATED_COLOR,
    REAL_COLOR,
    apply_thesis_style,
    despine,
    fig_size,
    save_figure,
)

OUTPUT_DIR = Path("abbildungen/outputs")
SNAPSHOT_STEPS = [0, 100, 250, 500]
SEED = cm.SEED
CONFIG = Path("configs/config_main.yml")  # für den Split-Seed


def pick_sample(sensor_table, split_seed: int):
    """Wählt reproduzierbar ein Test-Sample mit großer Spannweite im Tagesverlauf."""
    from src.training.build_training_data import load_splits, build_y

    splits = load_splits(cm.SPLIT_DIR / f"training_split_{split_seed}.json")
    mask = sensor_table.index.get_level_values("sensor_id").isin(splits[cm.SPLIT])
    index = sensor_table.loc[mask].index.sort_values()

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(index), size=min(50, len(index)), replace=False)
    candidates = index[idx]
    y_cand = build_y(sensor_table, candidates)
    spans = y_cand.max(axis=1) - y_cand.min(axis=1)
    best = int(np.argmax(spans))
    return candidates[best], y_cand[best]


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sensor_table = cm.get_sensor_table()

    from src.training.run_config import load_model_config
    split_seed = load_model_config(CONFIG)["split"]["seed"]

    key, y_real = pick_sample(sensor_table, split_seed)
    sensor_id, day = key
    print(f"Sample: sensor_id={sensor_id}, day={day.date()}")

    schedule = dm.DiffusionSchedule(timesteps=cm.DIFFUSION_STEPS, schedule_type="linear")
    x0 = torch.FloatTensor(_to_x0(y_real)).unsqueeze(0)  # x0 = y - 1 wie im Training

    torch.manual_seed(SEED)
    snapshots = {0: x0}
    xt = x0
    for i in range(1, schedule.timesteps + 1):
        beta_i = schedule.betas[i - 1]
        alpha_i = 1.0 - beta_i
        z = torch.randn_like(xt)
        xt = torch.sqrt(alpha_i) * xt + torch.sqrt(beta_i) * z
        if i in SNAPSHOT_STEPS:
            snapshots[i] = xt

    snapshots = {t: (xt[0].numpy() + 1.0) for t, xt in snapshots.items()}  # zurück in die Y-Skala

    fig, axes = plt.subplots(
        1, len(SNAPSHOT_STEPS), figsize=fig_size(width_fraction=1.0, aspect=0.38), sharey=True
    )
    hours = np.arange(24)
    for ax, t in zip(axes, SNAPSHOT_STEPS):
        ax.plot(hours, y_real, color=REAL_COLOR, linewidth=1.0, linestyle="--", alpha=0.6, label="Echt ($x_0$)")
        ax.plot(hours, snapshots[t], color=GENERATED_COLOR, linewidth=1.6, label="$x_t$")
        if t == 0:
            ax.set_title("$x_0=0$")
        elif t == schedule.timesteps:
            ax.set_title("$x_T=500$")
        else:
            ax.set_title(f"$x_t={t}$")
        ax.set_xlabel("Stunde")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 24])
    axes[0].set_ylabel("Last relativ zur\nJahresmittellast")
    axes[0].legend()
    despine(fig)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "diffusion_process_forward"
    save_figure(fig, out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
