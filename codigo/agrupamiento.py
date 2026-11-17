"""Reducción con SVD truncada y agrupamiento con K-means (cuaderno 03).

Pipeline final del informe: TF-IDF (5000) -> TruncatedSVD (100) -> K-means (K=6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

SEMILLA = 42
N_COMPONENTES_SVD = 100
K_ELEGIDO = 6


def reducir_svd(
    matriz: sp.csr_matrix,
    n_componentes: int = N_COMPONENTES_SVD,
    semilla: int = SEMILLA,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Ajusta ``TruncatedSVD`` (LSA) y devuelve ``(matriz_reducida, svd)``."""
    svd = TruncatedSVD(n_components=n_componentes, random_state=semilla)
    return svd.fit_transform(matriz), svd


def seleccionar_k(
    matriz: sp.csr_matrix | np.ndarray,
    rango_k: Sequence[int] = range(2, 13),
    semilla: int = SEMILLA,
    muestra_silueta: int = 4000,
) -> pd.DataFrame:
    """Barrido de K: devuelve una tabla con inercia (codo) y silueta para cada K."""
    filas = []
    n = matriz.shape[0]
    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=semilla, n_init=10)
        etiquetas = km.fit_predict(matriz)
        sil = silhouette_score(matriz, etiquetas, sample_size=min(muestra_silueta, n), random_state=semilla)
        filas.append({"K": k, "inercia": km.inertia_, "silueta": sil})
    return pd.DataFrame(filas)


def entrenar_kmeans(
    matriz: sp.csr_matrix | np.ndarray,
    k: int = K_ELEGIDO,
    semilla: int = SEMILLA,
    n_init: int = 10,
    **kwargs: Any,
) -> KMeans:
    """Entrena K-means (inicialización k-means++, algoritmo de Lloyd)."""
    if k < 2:
        raise ValueError("k debe ser al menos 2")
    modelo = KMeans(n_clusters=k, random_state=semilla, n_init=n_init, **kwargs)
    modelo.fit(matriz)
    return modelo


def asignar_clusters(modelo: KMeans, matriz: sp.csr_matrix | np.ndarray) -> np.ndarray:
    """Asigna cada fila al centroide más cercano (Fórmula 11)."""
    return modelo.predict(matriz)


def guardar_modelo(modelo: Any, ruta: str | Path) -> None:
    """Serializa un objeto (K-means o SVD) con ``joblib``."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: str | Path) -> Any:
    """Carga un objeto serializado con ``joblib``."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero el entrenamiento.")
    return joblib.load(ruta)
