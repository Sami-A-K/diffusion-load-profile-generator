"""
Gemeinsames Matplotlib-Styling für alle Abbildungen: Arial 12pt, TH-Köln-Farben als Akzente,
Achsen nur unten/links. apply_thesis_style() vor dem ersten plt.subplots() aufrufen.
"""
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

COLORS = {
    "black": "#1A1A1A",
    "red": "#D6212B",
    "orange": "#EE7A29",
    "magenta": "#9B2F7F",
    "gray": "#8C8C8C",
    "grid": "#D9D9D9",
}

# Reale Daten schwarz, generierte Daten in Akzentfarben (bei mehreren Modellen aus ACCENT_CYCLE).
REAL_COLOR = COLORS["black"]
GENERATED_COLOR = COLORS["red"]
ACCENT_CYCLE = [COLORS["red"], COLORS["magenta"], COLORS["orange"], COLORS["gray"]]

# Textbreite der Arbeit in inch
TEXTWIDTH_IN = 6.3

# Anzeigenamen für die Zeitkontext-Konstanten aus evaluation_metrics
CONTEXT_LABELS = {"Uebergang": "Übergang", "Sonntag": "Sonn-/Feiertag"}


def label(name: str) -> str:
    """Anzeigename eines Zeitkontext-Werts."""
    return CONTEXT_LABELS.get(name, name)


def fig_size(width_fraction: float = 1.0, aspect: float = 0.62) -> tuple[float, float]:
    """Abbildungsgröße in inch als Anteil von TEXTWIDTH_IN; aspect = Höhe/Breite."""
    width = TEXTWIDTH_IN * width_fraction
    return width, width * aspect


def apply_thesis_style() -> None:
    """Setzt die globalen rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.edgecolor": COLORS["black"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",  # nur horizontale Gitterlinien
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        # Mathtext in Arial
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # TrueType-Schriften im PDF
        "ps.fonttype": 42,
    })


def despine(target) -> None:
    """Entfernt obere und rechte Achsenlinie für eine Axes oder alle Axes einer Figure."""
    axes = target.axes if isinstance(target, matplotlib.figure.Figure) else [target]
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def save_figure(fig: matplotlib.figure.Figure, path, formats=("pdf", "png")) -> None:
    """Speichert eine Figure in allen angegebenen Formaten."""
    from pathlib import Path

    path = Path(path)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"))
