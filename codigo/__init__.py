"""Paquete ``codigo`` del proyecto de triaje de quejas de reseñas.

Agrupa los módulos reutilizables del flujo de trabajo:

- :mod:`codigo.preprocesamiento`: carga y limpieza del texto de las reseñas.
- :mod:`codigo.caracteristicas`: vectorización TF-IDF.
- :mod:`codigo.agrupamiento`: entrenamiento de KMeans y selección de ``k``.
- :mod:`codigo.evaluacion`: métricas de calidad y estabilidad de clústeres.
- :mod:`codigo.etiquetado`: términos representativos y nombres de clústeres.
- :mod:`codigo.triaje`: agente que sugiere el grupo de una queja nueva.

La lógica es la misma de los cuadernos 01-04 (mismos parámetros y semilla).
"""

__version__ = "0.2.0"
