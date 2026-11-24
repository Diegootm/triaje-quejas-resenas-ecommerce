"""Extracción de características de texto mediante TF-IDF (cuaderno 03).

Convierte el texto preprocesado en una matriz dispersa documento-término.
El vectorizador entrenado se guarda junto al modelo para que la interfaz
transforme las quejas nuevas con el MISMO vocabulario e IDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Nombres de prenda (lematizados) que dominan por frecuencia sin describir un problema.
STOPWORDS_DOMINIO = [
    "dress", "top", "shirt", "sweater", "jean", "pant", "skirt",
    "blouse", "jacket", "coat", "cardigan", "blazer",
]

PARAMS_TFIDF = dict(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.9)


def construir_tfidf(
    textos: Sequence[str],
    max_features: int = PARAMS_TFIDF["max_features"],
    ngram_range: tuple[int, int] = PARAMS_TFIDF["ngram_range"],
    min_df: int | float = PARAMS_TFIDF["min_df"],
    max_df: float = PARAMS_TFIDF["max_df"],
    stop_words: Sequence[str] | None = tuple(STOPWORDS_DOMINIO),
) -> tuple[sp.csr_matrix, TfidfVectorizer]:
    """Ajusta un ``TfidfVectorizer`` y devuelve ``(matriz, vectorizador)``.

    Valores por defecto = configuración del informe (Tabla 36): unigramas y
    bigramas, min_df=5, max_df=0.9, 5000 términos, IDF suavizado y norma L2.
    """
    vectorizador = TfidfVectorizer(
        stop_words=list(stop_words) if stop_words else None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    matriz = vectorizador.fit_transform(textos)
    return matriz, vectorizador


def guardar_vectorizador(vectorizador: TfidfVectorizer, ruta: str | Path) -> None:
    """Serializa el vectorizador con ``joblib``."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizador, ruta)
