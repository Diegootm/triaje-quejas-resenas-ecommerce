"""Reducción con SVD truncada y agrupamiento con K-means (cuaderno 03).

Pipeline del informe: TF-IDF (5000) -> TruncatedSVD (100) -> K-means.
El modelo del informe usa K=6, pero K se puede elegir en el rango
``K_MIN``..``K_MAX`` (``codigo.config``) y comparar con :func:`barrido_k`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from codigo.config import (
    K_MAX,
    K_MIN,
    K_MIN_RECOMENDADO,
    K_REFERENCIA_INFORME,
    MUESTRA_SILUETA,
    N_COMPONENTES_SVD,
    N_INIT,
    PROPORCION_MINIMA_GRUPO,
    RANGO_K,
    SEMILLA,
    TOLERANCIA_SILUETA,
)

def reducir_svd(
    matriz: sp.csr_matrix,
    n_componentes: int = N_COMPONENTES_SVD,
    semilla: int = SEMILLA,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Ajusta ``TruncatedSVD`` (LSA) y devuelve ``(matriz_reducida, svd)``."""
    svd = TruncatedSVD(n_components=n_componentes, random_state=semilla)
    return svd.fit_transform(matriz), svd


def validar_k(k: int, n_muestras: int | None = None, k_min: int = K_MIN, k_max: int = K_MAX) -> int:
    """Comprueba que ``k`` sea un entero dentro de ``[k_min, k_max]`` y menor que ``n_muestras``."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"K debe ser un entero; se recibió {k!r}")
    k = int(k)
    if not k_min <= k <= k_max:
        raise ValueError(f"K debe estar entre {k_min} y {k_max}; se recibió {k}")
    if n_muestras is not None and k >= n_muestras:
        raise ValueError(f"K ({k}) debe ser menor que el número de reseñas ({n_muestras})")
    return k


def entrenar_kmeans(
    matriz: sp.csr_matrix | np.ndarray,
    k: int = K_REFERENCIA_INFORME,
    semilla: int = SEMILLA,
    n_init: int = N_INIT,
    **kwargs: Any,
) -> KMeans:
    """Entrena K-means (inicialización k-means++, algoritmo de Lloyd)."""
    if k < 2:
        raise ValueError("k debe ser al menos 2")
    if k >= matriz.shape[0]:
        raise ValueError(f"k ({k}) debe ser menor que el número de filas ({matriz.shape[0]})")
    modelo = KMeans(n_clusters=k, random_state=semilla, n_init=n_init, **kwargs)
    modelo.fit(matriz)
    return modelo


def barrido_k(
    matriz: sp.csr_matrix | np.ndarray,
    rango_k: Sequence[int] = RANGO_K,
    semilla: int = SEMILLA,
    muestra_silueta: int = MUESTRA_SILUETA,
    progreso: Callable[[int, int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """Entrena K-means para cada K y devuelve ``(tabla, modelos)``.

    Cada modelo se entrena con :func:`entrenar_kmeans` (mismos parámetros que el
    modelo final), así que puede reutilizarse sin volver a entrenarlo.

    La tabla tiene una fila por K con inercia (codo), silueta (sobre una muestra)
    y el tamaño del grupo más chico y más grande. ``progreso(i, total, k)`` se
    llama después de terminar cada K, para mostrar el avance real.
    """
    rango_k = list(rango_k)
    n = matriz.shape[0]
    filas, modelos = [], {}
    for i, k in enumerate(rango_k, start=1):
        km = entrenar_kmeans(matriz, k=k, semilla=semilla)
        etiquetas = km.labels_
        sil = silhouette_score(matriz, etiquetas, sample_size=min(muestra_silueta, n), random_state=semilla)
        tamanos = np.bincount(etiquetas, minlength=k)
        filas.append({
            "K": k,
            "inercia": float(km.inertia_),
            "silueta": float(sil),
            "grupo_menor": int(tamanos.min()),
            "grupo_mayor": int(tamanos.max()),
            "proporcion_menor": float(tamanos.min() / n),
        })
        modelos[k] = km
        if progreso is not None:
            progreso(i, len(rango_k), k)
    return pd.DataFrame(filas), modelos


@dataclass(frozen=True)
class RecomendacionK:
    """Resultado de :func:`recomendar_k`. ``k`` es ``None`` si ningún K cumple los filtros."""

    k: int | None
    k_mejor_silueta: int | None
    silueta_mejor: float | None
    umbral_silueta: float | None
    candidatos: list[int] = field(default_factory=list)
    descartados: dict[int, str] = field(default_factory=dict)


def recomendar_k(
    tabla: pd.DataFrame,
    k_min: int = K_MIN_RECOMENDADO,
    proporcion_minima: float = PROPORCION_MINIMA_GRUPO,
    tolerancia: float = TOLERANCIA_SILUETA,
) -> RecomendacionK:
    """Recomendación HEURÍSTICA de K a partir de la tabla de :func:`barrido_k`.

    Regla (parsimonia sobre la silueta):

    1. Se descartan los K menores que ``k_min`` (dos grupos no sirven para
       distinguir tipos de queja) y los K cuyo grupo más chico tenga menos de
       ``proporcion_minima`` del corpus (grupos degenerados).
    2. Entre los restantes se toma la mejor silueta ``s*``.
    3. Se recomienda el K MÁS PEQUEÑO cuya silueta sea al menos
       ``s* - tolerancia * |s*|``.

    No es una garantía de calidad: en este corpus la silueta es baja y casi
    plana, y se calcula sobre una muestra, por lo que diferencias pequeñas
    pueden ser ruido. La recomendación es un punto de partida para la
    selección manual, no un reemplazo de la interpretación de los grupos.
    """
    descartados: dict[int, str] = {}
    candidatos = []
    for fila in tabla.sort_values("K").itertuples(index=False):
        k = int(fila.K)
        if k < k_min:
            descartados[k] = f"K menor que {k_min}"
        elif fila.proporcion_menor < proporcion_minima:
            descartados[k] = f"grupo más chico con {fila.proporcion_menor:.1%} del corpus (< {proporcion_minima:.0%})"
        else:
            candidatos.append(k)

    if not candidatos:
        return RecomendacionK(None, None, None, None, [], descartados)

    siluetas = tabla.set_index("K")["silueta"]
    k_mejor = max(candidatos, key=lambda k: siluetas[k])
    s_mejor = float(siluetas[k_mejor])
    umbral = s_mejor - tolerancia * abs(s_mejor)
    k_rec = min(k for k in candidatos if siluetas[k] >= umbral)
    return RecomendacionK(k_rec, k_mejor, s_mejor, umbral, candidatos, descartados)


def guardar_modelo(modelo: Any, ruta: str | Path) -> None:
    """Serializa un objeto (K-means o SVD) con ``joblib``."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
