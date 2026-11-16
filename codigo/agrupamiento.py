"""Agrupamiento de reseñas con KMeans.

Incluye la selección del número de clústeres ``k`` mediante una métrica
interna (por defecto, coeficiente de silueta) y el entrenamiento/persistencia
del modelo final.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from sklearn.cluster import KMeans


def seleccionar_k(
    matriz: "sp.csr_matrix | np.ndarray",
    rango_k: Sequence[int] = range(2, 11),
    metrica: str = "silhouette",
    semilla: int = 42,
) -> "pd.DataFrame":
    """Evalúa varios valores de ``k`` y devuelve una tabla comparativa.

    TODO: probar ``k`` en ``rango_k``, calcular la métrica elegida y devolver
    columnas ``k``, ``metrica`` y ``mejor_k`` sugerido.
    """
    raise NotImplementedError("Pendiente: implementar la selección de k.")


def entrenar_kmeans(
    matriz: "sp.csr_matrix | np.ndarray",
    k: int,
    semilla: int = 42,
    n_init: int = 10,
    **kwargs: Any,
) -> "KMeans":
    """Entrena y devuelve un modelo ``KMeans`` con ``k`` clústeres.

    TODO: fijar ``random_state``, validar ``k`` y documentar los parámetros.
    """
    raise NotImplementedError("Pendiente: implementar el entrenamiento de KMeans.")


def asignar_clusters(
    modelo: "KMeans",
    matriz: "sp.csr_matrix | np.ndarray",
) -> "np.ndarray":
    """Asigna cada fila de ``matriz`` al clúster más cercano.

    TODO: devolver un arreglo de etiquetas alineado con las filas de entrada.
    """
    raise NotImplementedError("Pendiente: implementar la asignación de clústeres.")


def guardar_modelo(modelo: "KMeans", ruta: str) -> None:
    """Serializa el modelo KMeans en ``ruta`` (por ejemplo con ``joblib``).

    TODO: crear la carpeta destino si no existe.
    """
    raise NotImplementedError("Pendiente: implementar el guardado del modelo.")


def cargar_modelo(ruta: str) -> Any:
    """Carga un modelo KMeans serializado desde disco.

    TODO: validar que el archivo exista y devolver el objeto.
    """
    raise NotImplementedError("Pendiente: implementar la carga del modelo.")
