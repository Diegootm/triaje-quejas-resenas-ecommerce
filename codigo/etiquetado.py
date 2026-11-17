"""Interpretación de los grupos: términos representativos y nombres (cuaderno 04).

Los nombres de :data:`NOMBRES_CLUSTER` son una interpretación humana posterior
(post-hoc) del modelo de referencia (K=6, semilla 42). NO se usan para
entrenar: K-means solo ve la matriz TF-IDF/SVD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

NOMBRES_CLUSTER: dict[int, str] = {
    0: "Insatisfacción general de ajuste, color y tela",
    1: "Calidad y precio (tela pobre, no vale lo que cuesta)",
    2: "Expectativa vs. realidad ('quería que me gustara')",
    3: "Silueta poco favorecedora (not flattering)",
    4: "Tallaje (la talla corre pequeña o grande)",
    5: "Apariencia vs. modelo (forma holgada / boxy)",
}


def terminos_top_por_cluster(
    matriz: sp.csr_matrix,
    etiquetas: Sequence[int],
    vectorizador: TfidfVectorizer,
    n_terminos: int = 10,
) -> dict[int, list[str]]:
    """Términos con mayor peso TF-IDF promedio dentro de cada grupo."""
    etiquetas = np.asarray(etiquetas)
    vocab = np.array(vectorizador.get_feature_names_out())
    resultado = {}
    for c in np.unique(etiquetas):
        medio = np.asarray(matriz[etiquetas == c].mean(axis=0)).ravel()
        resultado[int(c)] = list(vocab[medio.argsort()[::-1][:n_terminos]])
    return resultado


def resenas_cercanas(distancias: np.ndarray, etiquetas: Sequence[int], cluster: int, n: int = 4) -> np.ndarray:
    """Índices de las ``n`` reseñas del grupo más cercanas a su centroide (``kmeans.transform``)."""
    etiquetas = np.asarray(etiquetas)
    idx = np.where(etiquetas == cluster)[0]
    return idx[np.argsort(distancias[idx, cluster])[:n]]


def etiquetar_clusters(
    terminos: Mapping[int, Sequence[str]],
    nombres_manuales: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Nombre manual si existe; si no, una etiqueta provisional con los 3 primeros términos."""
    nombres_manuales = nombres_manuales or {}
    return {
        c: nombres_manuales.get(c) or f"Grupo {c}: " + ", ".join(list(terms)[:3])
        for c, terms in terminos.items()
    }


def resumen_clusters(
    df: pd.DataFrame,
    columna_cluster: str,
    columna_texto: str,
    etiquetas: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Tabla por grupo: nombre, tamaño, proporción, rating medio, % que recomienda y un ejemplo."""
    filas = []
    total = len(df)
    for c, grupo in df.groupby(columna_cluster):
        fila = {
            "cluster": int(c),
            "nombre": (etiquetas or {}).get(int(c), f"Grupo {c}"),
            "n_resenas": len(grupo),
            "proporcion": len(grupo) / total,
            "ejemplo": str(grupo[columna_texto].iloc[0])[:200],
        }
        if "Rating" in grupo:
            fila["rating_medio"] = grupo["Rating"].mean()
        if "Recommended IND" in grupo:
            fila["tasa_recomienda"] = grupo["Recommended IND"].mean()
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("n_resenas", ascending=False).reset_index(drop=True)


def exportar_etiquetas(resumen: pd.DataFrame, ruta: str | Path) -> None:
    """Guarda el resumen de grupos en CSV (UTF-8)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(ruta, index=False, encoding="utf-8")
