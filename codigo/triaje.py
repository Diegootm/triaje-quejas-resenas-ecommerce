"""Agente de triaje: sugiere el grupo de una queja nueva (apartado 5.8 del informe).

Percepción: el texto de una queja. Acción: sugerir el grupo al que más se
parece, mostrando sus términos y reseñas similares para que una persona
verifique la sugerencia. El agente NO resuelve la queja ni responde al cliente
y NO se reentrena con cada consulta.

Flujo (Ejemplo 10 / Tabla 44 del informe):
    1. preprocesar()                -> mismo pipeline del cuaderno 02
    2. vectorizador.transform()     -> TF-IDF con el IDF aprendido (no se recalcula)
    3. svd.transform()              -> proyección a 100 componentes (Fórmula 25)
    4. kmeans.transform()           -> distancia a los K centroides; gana la menor (Fórmula 11)
    5. similitud del coseno         -> reseñas del corpus más parecidas (Fórmula 7)

El vectorizador y el SVD son fijos. El modelo K-means se puede reemplazar por
otro con distinto K (:meth:`AgenteTriaje.con_kmeans`) sin recalcular TF-IDF ni SVD.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from codigo.etiquetado import (
    cargar_nombres,
    elegir_nombres_manuales,
    etiquetar_clusters,
    huella_particion,
    terminos_top_por_cluster,
)
from codigo.preprocesamiento import preprocesar

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "modelos"
RUTA_LIMPIAS = RAIZ / "datos" / "procesados" / "resenas_limpias.csv"

# Si la diferencia entre la distancia al 1.er y al 2.º grupo es menor que esto,
# la sugerencia se marca como "poco segura" (reseña de frontera).
MARGEN_DUDOSO = 0.02


@dataclass
class ResultadoTriaje:
    queja: str
    texto_limpio: str
    valido: bool                      # False si ninguna palabra está en el vocabulario
    grupo: int | None = None
    nombre: str | None = None
    distancias: list[float] = field(default_factory=list)
    segundo_grupo: int | None = None
    margen: float | None = None
    poco_seguro: bool = False
    terminos: list[str] = field(default_factory=list)
    similares: list[dict] = field(default_factory=list)
    aviso: str | None = None


class AgenteTriaje:
    """Carga los modelos entrenados y sugiere el grupo de quejas nuevas."""

    def __init__(self, dir_modelos: str | Path = DIR_MODELOS, ruta_corpus: str | Path = RUTA_LIMPIAS):
        dir_modelos = Path(dir_modelos)
        faltan = [n for n in ("vectorizador.joblib", "svd_lsa.joblib", "modelo_kmeans.joblib")
                  if not (dir_modelos / n).exists()]
        if faltan:
            raise FileNotFoundError(
                f"Faltan modelos en {dir_modelos}: {', '.join(faltan)}. "
                "Entrena primero con 'python entrenar_agente.py' o ejecutando los cuadernos 01-03."
            )
        self.vectorizador = joblib.load(dir_modelos / "vectorizador.joblib")
        self.svd = joblib.load(dir_modelos / "svd_lsa.joblib")
        kmeans = joblib.load(dir_modelos / "modelo_kmeans.joblib")

        # Corpus de quejas ya preprocesadas: sirve para mostrar reseñas parecidas
        # y para reentrenar K-means con otro K sobre la misma representación.
        corpus = pd.read_csv(ruta_corpus)
        self.corpus = corpus.dropna(subset=["texto_limpio"]).reset_index(drop=True)
        self.X_corpus = self.vectorizador.transform(self.corpus["texto_limpio"])
        self.X_svd_corpus = self.svd.transform(self.X_corpus)

        # Nombres guardados por entrenar_agente.py; solo se aplican a la partición
        # cuya huella coincide (ver etiquetado.elegir_nombres_manuales).
        try:
            self.nombres_persistidos = cargar_nombres(dir_modelos)
            aviso_lectura = None
        except ValueError as error:
            self.nombres_persistidos = None
            aviso_lectura = f"Se ignoraron los nombres guardados: {error}"

        self._usar_kmeans(kmeans)
        self.aviso_nombres = aviso_lectura or self.aviso_nombres
        self.kmeans_referencia = kmeans
        self.etiquetas_referencia = self.etiquetas_corpus

    def _usar_kmeans(self, kmeans: KMeans) -> None:
        """Asigna el corpus a los grupos de ``kmeans`` y recalcula términos y nombres.

        Los nombres persistidos o manuales solo se aplican si la huella de la
        partición coincide; si no, se usan etiquetas derivadas de los términos.
        """
        if kmeans.n_features_in_ != self.X_svd_corpus.shape[1]:
            raise ValueError(
                f"El modelo K-means espera {kmeans.n_features_in_} componentes y el SVD produce "
                f"{self.X_svd_corpus.shape[1]}."
            )
        self.kmeans = kmeans
        self.etiquetas_corpus = kmeans.predict(self.X_svd_corpus)
        self.terminos = terminos_top_por_cluster(self.X_corpus, self.etiquetas_corpus, self.vectorizador, 10)
        self.huella = huella_particion(self.etiquetas_corpus)
        manuales, self.origen_nombres, self.aviso_nombres = elegir_nombres_manuales(
            self.etiquetas_corpus, self.nombres_persistidos)
        self.nombres_manuales = self.origen_nombres != "terminos"
        self.nombres = etiquetar_clusters(self.terminos, manuales)
        # Un grupo sin reseñas del corpus no tiene términos; se le da un nombre genérico.
        for g in range(self.n_grupos):
            self.nombres.setdefault(g, f"Grupo {g}")
            self.terminos.setdefault(g, [])

    def con_kmeans(self, kmeans: KMeans) -> "AgenteTriaje":
        """Devuelve un agente que usa ``kmeans`` y comparte vectorizador, SVD y corpus.

        No modifica el agente original ni recalcula TF-IDF ni SVD.
        """
        nuevo = copy.copy(self)
        nuevo._usar_kmeans(kmeans)
        nuevo.aviso_nombres = None   # el aviso sobre nombres_grupos.json solo concierne al modelo guardado
        return nuevo

    @property
    def n_grupos(self) -> int:
        return int(self.kmeans.n_clusters)

    @property
    def k_referencia(self) -> int:
        return int(self.kmeans_referencia.n_clusters)

    def distribucion(self) -> pd.DataFrame:
        """Cantidad de quejas del corpus en cada grupo (para el gráfico de la interfaz)."""
        conteo = pd.Series(self.etiquetas_corpus).value_counts().sort_index()
        return pd.DataFrame({
            "grupo": conteo.index,
            "nombre": [self.nombres[int(g)] for g in conteo.index],
            "quejas": conteo.values,
            "porcentaje": (conteo.values / conteo.values.sum() * 100).round(1),
        })

    def sugerir(self, queja: str, n_similares: int = 3) -> ResultadoTriaje:
        limpio = preprocesar(queja)
        x = self.vectorizador.transform([limpio])
        if x.nnz == 0:
            return ResultadoTriaje(
                queja=queja, texto_limpio=limpio, valido=False,
                aviso=("Ninguna palabra de la queja está en el vocabulario del modelo (entrenado en inglés). "
                       "No se sugiere grupo para no dar una respuesta sin sentido."),
            )
        dist = self.kmeans.transform(self.svd.transform(x))[0]
        orden = np.argsort(dist)
        g, g2 = int(orden[0]), int(orden[1])
        margen = float(dist[g2] - dist[g])
        sims = (self.X_corpus @ x.T).toarray().ravel()   # coseno: ambos vectores tienen norma 1
        similares = [
            {
                "similitud": float(sims[i]),
                "grupo": int(self.etiquetas_corpus[i]),
                "rating": int(self.corpus.loc[i, "Rating"]) if "Rating" in self.corpus else None,
                "texto": str(self.corpus.loc[i, "Review Text"]),
            }
            for i in sims.argsort()[::-1][:n_similares]
        ]
        return ResultadoTriaje(
            queja=queja, texto_limpio=limpio, valido=True, grupo=g, nombre=self.nombres[g],
            distancias=[float(d) for d in dist], segundo_grupo=g2, margen=margen,
            poco_seguro=margen < MARGEN_DUDOSO, terminos=self.terminos[g], similares=similares,
            aviso=(f"Sugerencia poco segura: el grupo {g2} ({self.nombres[g2]}) está casi igual de cerca."
                   if margen < MARGEN_DUDOSO else None),
        )
