"""Etiquetado e interpretación de los clústeres.

Extrae los términos con mayor peso TF-IDF por clúster para proponer nombres
descriptivos (por ejemplo, "tallas", "calidad de tela", "envío") y facilitar
la revisión manual por parte del equipo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer


def terminos_top_por_cluster(
    matriz: "sp.csr_matrix",
    etiquetas: Sequence[int],
    vectorizador: "TfidfVectorizer",
    n_terminos: int = 10,
) -> dict[int, list[str]]:
    """Devuelve los ``n_terminos`` más representativos de cada clúster.

    TODO: calcular el peso medio por término y clúster, excluyendo términos
    demasiado frecuentes o poco informativos.
    """
    raise NotImplementedError("Pendiente: implementar la extracción de términos.")


def etiquetar_clusters(
    terminos: Mapping[int, Sequence[str]],
    nombres_manuales: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Propone un nombre corto por clúster a partir de sus términos.

    TODO: usar ``nombres_manuales`` cuando exista y, si no, generar una
    etiqueta provisional legible con los términos top.
    """
    raise NotImplementedError("Pendiente: implementar el etiquetado de clústeres.")


def resumen_clusters(
    df: "pd.DataFrame",
    columna_cluster: str,
    columna_texto: str,
    etiquetas: Mapping[int, str] | None = None,
) -> "pd.DataFrame":
    """Construye una tabla resumen por clúster.

    TODO: incluir tamaño, proporción, rating medio, tasa de recomendación y
    un ejemplo de reseña por clúster.
    """
    raise NotImplementedError("Pendiente: implementar el resumen por clúster.")


def exportar_etiquetas(
    resumen: "pd.DataFrame",
    ruta: str,
) -> None:
    """Guarda el resumen de clústeres en CSV para revisión.

    TODO: crear la carpeta destino y usar codificación UTF-8.
    """
    raise NotImplementedError("Pendiente: implementar la exportación de etiquetas.")
