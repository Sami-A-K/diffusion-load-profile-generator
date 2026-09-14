"""
Lastform-Clustering der Sensoren (k-means auf dem mittleren Werktags-Tagesprofil).

Übernimmt das Cluster-Schema des Vorgängers:
- Strom wird nach location_category gruppiert (Privat/Gewerblich/Oeffentlich), Wärme bildet
  eine gemeinsame Gruppe.
- Feste Clusteranzahl je Gruppe (CLUSTER_K: 3/3/7/6 = 19 Profiltypen).
- Ein Profil je Sensor und Jahr.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.data.encode_covariates import resolve_category

# Clusteranzahl je Gruppe nach dem Vorgänger (13 Strom- und 6 Wärmeprofile).
CLUSTER_K = {
    "Privat": 3,
    "Gewerblich": 3,
    "Oeffentlich": 7,
    "Waerme": 6,
}

CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def cluster_group_for_row(location_category: str, energy_type: str) -> str:
    """Cluster-Gruppe eines Sensors: location_category bei Strom, eine gemeinsame Gruppe bei Wärme."""
    if energy_type == "Waerme":
        return "Waerme"
    return location_category


def weekday_profiles_per_year(df_hourly: pd.Series) -> dict[int, np.ndarray]:
    """
    Mittleres Werktags-Tagesprofil (24 Werte) je Jahr, normiert auf das Jahresmaximum.
    Jahre mit unvollständigem oder konstantem Profil werden ausgelassen.
    """
    normalized = df_hourly / df_hourly.groupby(df_hourly.index.year).transform("max")
    weekday = normalized[normalized.index.dayofweek < 5]

    profiles: dict[int, np.ndarray] = {}
    for year, group in weekday.groupby(weekday.index.year):
        profile = group.groupby(group.index.hour).mean().to_numpy()
        if len(profile) < 24 or np.isnan(profile).any() or np.all(profile == 0):
            continue
        profiles[int(year)] = profile
    return profiles


def z_normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-Normalisierung je Profil, damit k-means nach der Form und nicht nach dem Niveau clustert."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + eps
    return (arr - mean) / std


def cluster_cell(cell: pd.DataFrame, k: int, random_state: int = 42) -> pd.DataFrame:
    """Clustert eine Gruppe mit festem k; ergänzt 'cluster_local' (Index innerhalb der Gruppe) und 'k_used'."""
    cell = cell.copy()
    arr = z_normalize_rows(np.vstack(cell["profile"].to_numpy()))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)

    cell["cluster_local"] = kmeans.fit_predict(arr)
    cell["k_used"] = k
    return cell


def sample_cluster(category: str, energy_type: str, proportions: pd.DataFrame, rng: np.random.Generator) -> int:
    """
    Zieht ein Cluster gemäß der empirischen Häufigkeit innerhalb einer (category, energy_type)-Zelle.

    Wird bei der Generierung ohne Lasthistorie verwendet: dort ist nur die Kategorie bekannt,
    nicht die Cluster-Zugehörigkeit. Wirft KeyError, wenn die Zelle nicht existiert.
    """
    cell = proportions[(proportions["category"] == category) & (proportions["energy_type"] == energy_type)]
    if cell.empty:
        raise KeyError(f"Keine Cluster-Proportionen für (category={category!r}, energy_type={energy_type!r}).")
    return int(rng.choice(cell["cluster"].to_numpy(), p=cell["proportion"].to_numpy()))


def plot_clusters(df_clustered: pd.DataFrame, output_path: Path) -> None:
    """Ein Subplot je Cluster-Gruppe mit Mittelwertprofil (± 1 Std) je Cluster."""
    groups = sorted(df_clustered["cluster_group"].unique())
    n = len(groups)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.axis("off")

    hours = np.arange(24)
    for ax, group_name in zip(axes, groups):
        cell = df_clustered[df_clustered["cluster_group"] == group_name]
        for cluster_local, group in cell.groupby("cluster_local"):
            arr = np.vstack(group["profile"].to_numpy())
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            color = CLUSTER_COLORS[cluster_local % len(CLUSTER_COLORS)]
            ax.plot(hours, mean, color=color, linewidth=2, label=f"Cluster {cluster_local} (n={len(group)})")
            ax.fill_between(hours, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

        ax.set_title(f"{group_name} (n={len(cell)} Sensor-Jahre)", fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="upper left")

    fig.supxlabel("Stunde")
    fig.supylabel("Anteil Jahresmaximum")
    fig.suptitle("Werktags-Lastprofile je Cluster-Gruppe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_clusters(
    df_sensors: pd.DataFrame,
    load_hourly: "callable",
    output_cluster_path: Path,
    output_proportions_path: Path,
    output_plot_path: Path,
) -> pd.DataFrame:
    """
    Clustert die Sensoren aus df_sensors (Index = sensor_id) und schreibt Zuordnung,
    Sampling-Proportionen und Plot. load_hourly(sensor_id) liefert die stündliche kWh-Reihe.

    Returns: Zuordnungstabelle (sensor_id, year) -> cluster.
    """
    rows = []
    for sensor_id, row in tqdm(df_sensors.iterrows(), total=len(df_sensors), desc="Cluster-Profile"):
        profiles = weekday_profiles_per_year(load_hourly(sensor_id))
        for year, profile in profiles.items():
            rows.append({
                "sensor_id": sensor_id,
                "year": year,
                "cluster_group": cluster_group_for_row(row["location_category"], row["energy_type"]),
                "category": resolve_category(row["location_usage"], row["location_usage_detail"]),
                "energy_type": row["energy_type"],
                "profile": profile,
            })

    df_profiles = pd.DataFrame(rows)

    clustered, offset = [], 0
    for group_name, cell in df_profiles.groupby("cluster_group"):
        cell = cluster_cell(cell, CLUSTER_K[group_name])
        # Offset je Gruppe ergibt einen global eindeutigen Cluster-Code.
        cell["cluster"] = cell["cluster_local"] + offset
        offset += CLUSTER_K[group_name]
        clustered.append(cell)

    df_clustered = pd.concat(clustered, ignore_index=True)

    df_cluster_out = df_clustered.set_index(["sensor_id", "year"])[["cluster_group", "energy_type", "cluster_local", "cluster"]]
    df_cluster_out.to_csv(output_cluster_path)

    # Proportionen je Sensor (häufigstes Cluster über die Jahre), damit Sensoren mit langer
    # Historie nicht mehrfach zählen.
    modal = (
        df_clustered.groupby(["sensor_id", "category", "energy_type"])["cluster"]
        .agg(lambda s: s.mode().iat[0])
        .reset_index()
    )
    proportions = (
        modal.groupby(["category", "energy_type", "cluster"]).size().rename("n_sensors").reset_index()
    )
    proportions["n_cell"] = proportions.groupby(["category", "energy_type"])["n_sensors"].transform("sum")
    proportions["proportion"] = proportions["n_sensors"] / proportions["n_cell"]
    proportions.to_csv(output_proportions_path, index=False)

    plot_clusters(df_clustered, output_plot_path)

    print(f"  Cluster-Gruppen: {df_profiles['cluster_group'].nunique()}, Cluster gesamt: {offset}")
    print(f"  Sensor-Jahre geclustert: {len(df_clustered)} über {df_clustered['sensor_id'].nunique()} Sensoren")
    return df_cluster_out
