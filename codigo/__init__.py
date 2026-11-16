"""Paquete ``codigo`` del proyecto de triaje de quejas de reseñas.

Agrupa los módulos reutilizables del flujo de trabajo:

- :mod:`codigo.preprocesamiento`: carga y limpieza del texto de las reseñas.
- :mod:`codigo.caracteristicas`: vectorización TF-IDF.
- :mod:`codigo.agrupamiento`: entrenamiento de KMeans y selección de ``k``.
- :mod:`codigo.evaluacion`: métricas de calidad y estabilidad de clústeres.
- :mod:`codigo.etiquetado`: términos representativos y nombres de clústeres.

Los módulos son esqueletos documentados: las firmas públicas están definidas,
pero la lógica se implementará en etapas posteriores.
"""

__version__ = "0.1.0"
