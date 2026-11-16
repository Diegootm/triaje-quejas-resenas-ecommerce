"""Evaluación de la calidad y estabilidad del agrupamiento.

Métricas internas (silueta), externas opcionales (ARI frente a etiquetas
manuales o heurísticas) y estabilidad entre ejecuciones o submuestras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp


def coeficiente_silueta(
    matriz: "sp.csr_matrix | np.ndarray",
    etiquetas: Sequence[int],
) -> float:
    """Calcula el coeficiente de silueta medio.

    TODO: delegar en ``sklearn.metrics.silhouette_score`` y documentar la
    métrica de distancia usada.
    """
    raise NotImplementedError("Pendiente: implementar el coeficiente de silueta.")


def indice_rand_ajustado(
    etiquetas_reales: Sequence[int],
    etiquetas_predichas: Sequence[int],
) -> float:
    """Calcula el Adjusted Rand Index (ARI) entre dos particiones.

    TODO: validar longitudes y usar ``sklearn.metrics.adjusted_rand_score``.
    """
    raise NotImplementedError("Pendiente: implementar el ARI.")


def evaluar_estabilidad(
    matriz: "sp.csr_matrix | np.ndarray",
    k: int,
    n_repeticiones: int = 10,
    semilla: int = 42,
) -> "pd.DataFrame":
    """Estima la estabilidad del agrupamiento entre repeticiones.

    TODO: entrenar KMeans con distintas semillas/submuestras, comparar
    particiones con ARI y devolver media y desviación por repetición.
    """
    raise NotImplementedError("Pendiente: implementar la evaluación de estabilidad.")


def comparar_etiquetas_manuales(
    df: "pd.DataFrame",
    columna_manual: str,
    columna_cluster: str,
) -> "pd.DataFrame":
    """Cruza clústeres con etiquetas manuales o heurísticas.

    TODO: devolver una tabla de contingencia y métricas externas asociadas.
    """
    raise NotImplementedError("Pendiente: implementar la comparación de etiquetas.")


def resumen_evaluacion(
    matriz: "sp.csr_matrix | np.ndarray",
    etiquetas: Sequence[int],
) -> dict[str, float]:
    """Agrupa las métricas principales en un diccionario.

    TODO: incluir silueta, tamaño de clústeres y proporción de ruido/outliers.
    """
    raise NotImplementedError("Pendiente: implementar el resumen de evaluación.")
