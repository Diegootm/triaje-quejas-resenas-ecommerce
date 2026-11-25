"""Evaluación de la calidad y estabilidad del agrupamiento (cuaderno 04)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from codigo.config import MUESTRA_SILUETA, N_INIT_ESTABILIDAD, N_REPETICIONES_ESTABILIDAD, SEMILLA


def coeficiente_silueta(
    matriz: sp.csr_matrix | np.ndarray,
    etiquetas: Sequence[int],
    muestra: int = MUESTRA_SILUETA,
    semilla: int = SEMILLA,
) -> float:
    """Silueta promedio (Fórmula 17) con distancia euclídea, sobre una muestra."""
    return float(silhouette_score(matriz, etiquetas, sample_size=min(muestra, matriz.shape[0]), random_state=semilla))


def indice_rand_ajustado(etiquetas_a: Sequence[int], etiquetas_b: Sequence[int]) -> float:
    """ARI entre dos particiones del mismo conjunto (Fórmula 18)."""
    if len(etiquetas_a) != len(etiquetas_b):
        raise ValueError("Las particiones deben tener la misma longitud")
    return float(adjusted_rand_score(etiquetas_a, etiquetas_b))


def evaluar_estabilidad(
    matriz: sp.csr_matrix | np.ndarray,
    k: int,
    n_repeticiones: int = N_REPETICIONES_ESTABILIDAD,
) -> pd.DataFrame:
    """Entrena K-means con semillas 0..n-1 y ``n_init=1`` y calcula el ARI de cada par.

    Se usa ``n_init=1`` a propósito: con ``n_init=10`` cada corrida elegiría su
    mejor inicialización y la estabilidad saldría artificialmente alta.
    """
    particiones = [KMeans(n_clusters=k, random_state=s, n_init=N_INIT_ESTABILIDAD).fit_predict(matriz) for s in range(n_repeticiones)]
    filas = [
        {"semilla_i": i, "semilla_j": j, "ari": adjusted_rand_score(particiones[i], particiones[j])}
        for i in range(n_repeticiones)
        for j in range(i + 1, n_repeticiones)
    ]
    return pd.DataFrame(filas)


def comparar_etiquetas_manuales(df: pd.DataFrame, columna_manual: str, columna_cluster: str) -> pd.DataFrame:
    """Tabla de contingencia normalizada por fila (cada grupo suma 1).

    Con ``columna_manual="Department Name"`` sirve para comprobar que los grupos
    no separan por tipo de prenda (Tabla 42 del informe).
    """
    tabla = pd.crosstab(df[columna_cluster], df[columna_manual])
    return tabla.div(tabla.sum(axis=1), axis=0)
