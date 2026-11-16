"""Extracción de características de texto mediante TF-IDF.

Convierte el texto preprocesado en una matriz numérica dispersa apta para
KMeans. El vectorizador entrenado debe persistirse junto al modelo para que
la interfaz transforme nuevas reseñas de la misma forma.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer


def construir_tfidf(
    textos: Sequence[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 5,
    max_df: float = 0.9,
) -> tuple["sp.csr_matrix", "TfidfVectorizer"]:
    """Ajusta un ``TfidfVectorizer`` y devuelve ``(matriz, vectorizador)``.

    TODO: validar hiperparámetros y documentar los valores elegidos tras
    el análisis exploratorio.
    """
    raise NotImplementedError("Pendiente: implementar la vectorización TF-IDF.")


def transformar_textos(
    textos: Sequence[str],
    vectorizador: "TfidfVectorizer",
) -> "sp.csr_matrix":
    """Transforma textos nuevos con un vectorizador ya ajustado.

    TODO: manejar textos vacíos sin romper la interfaz.
    """
    raise NotImplementedError("Pendiente: implementar la transformación de textos.")


def guardar_vectorizador(vectorizador: "TfidfVectorizer", ruta: str) -> None:
    """Serializa el vectorizador (por ejemplo con ``joblib``).

    TODO: crear la carpeta destino si no existe.
    """
    raise NotImplementedError("Pendiente: implementar el guardado del vectorizador.")


def cargar_vectorizador(ruta: str) -> Any:
    """Carga un vectorizador serializado desde disco.

    TODO: validar que el archivo exista y devolver el objeto.
    """
    raise NotImplementedError("Pendiente: implementar la carga del vectorizador.")
