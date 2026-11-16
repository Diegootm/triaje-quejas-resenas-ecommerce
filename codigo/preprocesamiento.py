"""Preprocesamiento de reseñas de e-commerce.

Responsabilidades previstas:

1. Cargar el dataset crudo (``datos/crudos/archive.zip`` o el CSV extraído).
2. Unir los campos de texto (``Title`` y ``Review Text``).
3. Limpiar el texto conservando el idioma original (inglés).
4. Tokenizar, eliminar *stopwords* y lematizar.

Nota: el texto NO se traduce; se mantiene en inglés para no perder matices
y para que TF-IDF y los términos por clúster sean interpretables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    import pandas as pd

# Columnas de texto habituales del dataset de Women's E-Commerce Clothing Reviews.
COLUMNAS_TEXTO: tuple[str, ...] = ("Title", "Review Text")


def cargar_resenas(ruta: str, **kwargs) -> "pd.DataFrame":
    """Carga el CSV de reseñas desde ``ruta``.

    TODO: detectar separador/codificación, validar columnas esperadas y
    devolver un ``DataFrame`` sin filas con texto vacío.
    """
    raise NotImplementedError("Pendiente: implementar la carga del dataset.")


def unir_texto(
    df: "pd.DataFrame",
    columnas: Sequence[str] = COLUMNAS_TEXTO,
) -> "pd.Series":
    """Concatena las columnas de texto disponibles en una sola serie.

    TODO: unir ``Title`` + ``Review Text`` tratando valores nulos como cadena
    vacía y normalizando espacios.
    """
    raise NotImplementedError("Pendiente: implementar la unión de columnas de texto.")


def limpiar_texto(texto: str) -> str:
    """Limpia un texto individual (minúsculas, puntuación, espacios).

    TODO: decidir si se eliminan números/URLs y documentar las reglas.
    """
    raise NotImplementedError("Pendiente: implementar la limpieza de texto.")


def tokenizar(texto: str) -> list[str]:
    """Divide un texto limpio en tokens.

    TODO: definir tokenizador (NLTK o expresión regular) y probarlo con
    ejemplos del dataset.
    """
    raise NotImplementedError("Pendiente: implementar la tokenización.")


def eliminar_stopwords(
    tokens: Iterable[str],
    stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Elimina *stopwords* del inglés y tokens muy cortos.

    TODO: cargar la lista de NLTK y permitir una lista personalizada.
    """
    raise NotImplementedError("Pendiente: implementar la eliminación de stopwords.")


def lematizar(tokens: Iterable[str]) -> list[str]:
    """Lematiza una secuencia de tokens en inglés.

    TODO: usar ``WordNetLemmatizer`` y descargar los recursos de NLTK si faltan.
    """
    raise NotImplementedError("Pendiente: implementar la lematización.")


def preprocesar_dataframe(
    df: "pd.DataFrame",
    columna_texto: str = "texto",
    columna_salida: str = "texto_limpio",
) -> "pd.DataFrame":
    """Aplica el pipeline de limpieza a todo el ``DataFrame``.

    TODO: orquestar ``unir_texto`` -> ``limpiar_texto`` -> ``tokenizar`` ->
    ``eliminar_stopwords`` -> ``lematizar`` y guardar el resultado.
    """
    raise NotImplementedError("Pendiente: implementar el pipeline de preprocesamiento.")
