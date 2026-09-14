"""
Trainings- und Validierungsverlust über die Epochen für baseline und main, je fünf Trainingsläufe
mit unterschiedlichem Seed (9/5/7/12/20). Datenquelle: training_history_<version>.json.

    python -m abbildungen.plot_training_loss
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from abbildungen.plot_style import ACCENT_CYCLE, apply_thesis_style, despine, fig_size, save_figure

OUTPUT_DIR = Path("abbildungen/outputs")
WEIGHTS_DIR = Path("src/model/model_weights")

# Seed 9 ist der Standardlauf je Gruppe, 5/7/12/20 die Wiederholungen.
GROUP_FILES = {
    "baseline": ["baseline", "baseline_seed05", "baseline_seed07", "baseline_seed12", "baseline_seed20"],
    "main": ["main", "patch_8h_seed05", "patch_8h_seed07", "patch_8h_seed12", "patch_8h_seed20"],
}
GROUP_LABELS = {"baseline": "baseline", "main": "main"}
GROUP_COLORS = {"baseline": ACCENT_CYCLE[1], "main": ACCENT_CYCLE[0]}
# Ein Linienstil je Seed
SEED_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]


def load_history(file_stem: str) -> dict:
    with open(WEIGHTS_DIR / f"training_history_{file_stem}.json", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_comparison(group_histories: dict[str, list[dict]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=fig_size(width_fraction=1.0, aspect=0.4), sharey=True)
    for ax, loss_key, title in ((axes[0], "train_loss", "Train-Loss"), (axes[1], "val_loss", "Val-Loss")):
        n_epochs = None
        for group, histories in group_histories.items():
            color = GROUP_COLORS[group]
            for i, history in enumerate(histories):
                losses = history[loss_key]
                n_epochs = len(losses)
                ax.plot(range(1, n_epochs + 1), losses, color=color, linewidth=1.1,
                        linestyle=SEED_LINESTYLES[i], alpha=0.85)
        ax.set_xlabel("Epoche")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xlim(0, 50)
        ax.set_xticks(range(0, 51, 10))

    # Vergrößerter Ausschnitt der letzten Epochen (Train-Loss, baseline) oberhalb der Kurven
    ax_train = axes[0]
    baseline_histories = group_histories["baseline"]
    zoom_start = max(1, n_epochs - 14)
    zoom_ys = [h["train_loss"][zoom_start - 1:] for h in baseline_histories]
    y_min = min(min(ys) for ys in zoom_ys)
    y_max = max(max(ys) for ys in zoom_ys)
    y_pad = (y_max - y_min) * 0.15

    axis_top = ax_train.get_ylim()[1]
    gap_bottom = max(max(h["train_loss"][zoom_start - 1:]) for h in baseline_histories)
    gap_margin = (axis_top - gap_bottom) * 0.1
    inset_x0, inset_x1 = zoom_start + 3, n_epochs
    inset_y0, inset_y1 = gap_bottom + gap_margin, axis_top - gap_margin
    axins = ax_train.inset_axes([inset_x0, inset_y0, inset_x1 - inset_x0, inset_y1 - inset_y0],
                                 transform=ax_train.transData)
    for i, history in enumerate(baseline_histories):
        losses = history["train_loss"]
        axins.plot(range(1, len(losses) + 1), losses, color=GROUP_COLORS["baseline"], linewidth=1.2,
                   linestyle=SEED_LINESTYLES[i], alpha=0.9)
    axins.set_xlim(zoom_start, n_epochs)
    axins.set_ylim(y_min - y_pad, y_max + y_pad)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_color("#8C8C8C")
    mark_inset(ax_train, axins, loc1=3, loc2=4, fc="none", ec="#8C8C8C", linewidth=0.7)

    legend_handles = [Line2D([0], [0], color=GROUP_COLORS[g], linewidth=1.6, label=GROUP_LABELS[g])
                       for g in GROUP_FILES]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    despine(fig)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    group_histories = {group: [load_history(f) for f in files] for group, files in GROUP_FILES.items()}
    for group, histories in group_histories.items():
        for file_stem, history in zip(GROUP_FILES[group], histories):
            print(f"{file_stem}: best_epoch={history['best_epoch']}, best_val_loss={history['best_val_loss']:.5f}")
    plot_loss_comparison(group_histories, OUTPUT_DIR / "training_loss_baseline_vs_main_seeds")
    print(f"Abbildung gespeichert: {OUTPUT_DIR}/training_loss_baseline_vs_main_seeds.pdf")


if __name__ == "__main__":
    main()
