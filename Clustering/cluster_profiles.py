"""Build readable summaries of per-cluster standardized performance profiles."""

from collections.abc import Sequence

import pandas as pd


DEFAULT_KEY_COLUMNS = ("league", "season", "team", "id_jugador")


def _validate_row_alignment(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    key_columns: Sequence[str],
) -> None:
    missing_data = [column for column in key_columns if column not in data.columns]
    missing_assignments = [column for column in key_columns if column not in assignments.columns]
    if missing_data or missing_assignments:
        raise ValueError(
            "Faltan columnas clave para validar el orden: "
            f"datos={missing_data}, asignaciones={missing_assignments}"
        )
    if len(data) != len(assignments):
        raise ValueError(
            "La matriz de métricas y las asignaciones tienen distinta cantidad de filas."
        )

    data_keys = data.loc[:, key_columns].astype(str).reset_index(drop=True)
    assignment_keys = assignments.loc[:, key_columns].astype(str).reset_index(drop=True)
    if not data_keys.equals(assignment_keys):
        raise ValueError(
            "La matriz de métricas y las asignaciones no tienen el mismo orden de filas."
        )


def build_cluster_profiles(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    metric_columns: Sequence[str],
    key_columns: Sequence[str] = DEFAULT_KEY_COLUMNS,
) -> pd.DataFrame:
    """Return one row per cluster with metric means and player counts.

    ``N_registros`` counts player-team-season rows, while
    ``N_jugadores_unicos`` counts distinct player IDs across those rows.
    """
    if not metric_columns:
        raise ValueError("Se necesita al menos una métrica para resumir los clústeres.")
    _validate_row_alignment(data, assignments, key_columns)

    required_assignments = {"Cluster", "Rol", "id_jugador"}
    missing = sorted(required_assignments.difference(assignments.columns))
    if missing:
        raise ValueError(f"Faltan columnas en las asignaciones: {missing}")
    missing_metrics = [column for column in metric_columns if column not in data.columns]
    if missing_metrics:
        raise ValueError(f"Faltan métricas en la matriz: {missing_metrics}")
    if assignments["Cluster"].isna().any():
        raise ValueError("Hay filas sin asignación de clúster.")
    if (assignments.groupby("Cluster")["Rol"].nunique(dropna=False) > 1).any():
        raise ValueError("Un clúster tiene más de un nombre de rol asignado.")

    numeric_metrics = (
        data.loc[:, metric_columns]
        .apply(pd.to_numeric, errors="raise")
        .reset_index(drop=True)
    )
    cluster_ids = assignments["Cluster"].reset_index(drop=True)
    grouped_means = numeric_metrics.groupby(cluster_ids, sort=True).mean()
    grouped_means.index.name = "Cluster"

    role_by_cluster = assignments.groupby("Cluster", sort=True)["Rol"].first()
    records_by_cluster = assignments.groupby("Cluster", sort=True).size()
    unique_players_by_cluster = assignments.groupby("Cluster", sort=True)["id_jugador"].nunique()

    profiles = grouped_means.reset_index()
    profiles.insert(1, "Rol", profiles["Cluster"].map(role_by_cluster))
    profiles.insert(
        2,
        "N_registros",
        profiles["Cluster"].map(records_by_cluster).astype(int),
    )
    profiles.insert(
        3,
        "N_jugadores_unicos",
        profiles["Cluster"].map(unique_players_by_cluster).astype(int),
    )
    return profiles


def build_cluster_summary(
    profiles: pd.DataFrame,
    metric_columns: Sequence[str],
    top_n: int = 3,
    bottom_n: int = 2,
) -> pd.DataFrame:
    """Summarize the highest and lowest mean metrics for each cluster."""
    if not metric_columns:
        raise ValueError("Se necesita al menos una métrica para resumir los clústeres.")
    if top_n < 1 or bottom_n < 1:
        raise ValueError("top_n y bottom_n deben ser mayores que cero.")
    required = [
        "Cluster",
        "Rol",
        "N_registros",
        "N_jugadores_unicos",
        *metric_columns,
    ]
    missing = [column for column in required if column not in profiles.columns]
    if missing:
        raise ValueError(f"Faltan columnas en los perfiles: {missing}")

    rows = []
    for _, profile in profiles.iterrows():
        means = pd.to_numeric(profile.loc[list(metric_columns)], errors="raise")
        highest = means.sort_values(ascending=False, kind="stable").head(top_n)
        lowest = means.sort_values(ascending=True, kind="stable").head(bottom_n)
        rows.append(
            {
                "Cluster": int(profile["Cluster"]),
                "Rol": profile["Rol"],
                "N_registros": int(profile["N_registros"]),
                "N_jugadores_unicos": int(profile["N_jugadores_unicos"]),
                "metricas_mas_altas": "; ".join(
                    f"{metric} ({value:+.2f})" for metric, value in highest.items()
                ),
                "metricas_mas_bajas": "; ".join(
                    f"{metric} ({value:+.2f})" for metric, value in lowest.items()
                ),
            }
        )
    return pd.DataFrame(rows)
