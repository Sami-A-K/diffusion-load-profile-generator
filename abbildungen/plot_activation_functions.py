"""
Aktivierungsfunktionen ReLU, SiLU und GELU im Vergleich. SiLU wird im Diffusions-Decoder
verwendet, GELU im FFN des Encoders, ReLU dient als Referenz.

    python -m abbildungen.plot_activation_functions
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from abbildungen.plot_style import COLORS, apply_thesis_style, despine, fig_size, save_figure

OUT = Path("abbildungen/outputs/activation_functions")
X_RANGE = (-5, 5)
Y_RANGE = (-1, 5)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exakte GELU wie nn.GELU()."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def main() -> None:
    apply_thesis_style()
    x = np.linspace(*X_RANGE, 601)

    fig, ax = plt.subplots(figsize=fig_size(width_fraction=0.75, aspect=0.72))
    ax.axhline(0, color=COLORS["grid"], lw=0.8, zorder=1)
    ax.axvline(0, color=COLORS["grid"], lw=0.8, zorder=1)
    ax.plot(x, relu(x), color=COLORS["black"], lw=1.6, label="ReLU", zorder=2)
    ax.plot(x, silu(x), color=COLORS["red"], lw=1.8, label="SiLU", zorder=3)
    ax.plot(x, gelu(x), color=COLORS["orange"], lw=1.8, label="GELU", zorder=3)

    ax.set_xlabel("$x$")
    ax.set_ylabel("Ausgabe")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_xticks(range(X_RANGE[0], X_RANGE[1] + 1))
    ax.set_yticks(range(Y_RANGE[0], Y_RANGE[1] + 1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(True, which="major", axis="both", color=COLORS["grid"], linewidth=0.6)
    ax.grid(True, which="minor", axis="both", color=COLORS["grid"], linewidth=0.4)
    ax.legend(loc="upper left")
    despine(ax)
    fig.tight_layout()
    save_figure(fig, OUT)
    print(f"-> {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
