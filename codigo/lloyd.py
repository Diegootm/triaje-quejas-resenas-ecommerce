"""Algoritmo de Lloyd (K-means) paso a paso, para mostrar cómo se forman los grupos.

Es una réplica en numpy de K-means de scikit-learn: ``N_INIT`` arranques k-means++
(la MISMA secuencia de semillas que usa ``KMeans(random_state=SEMILLA, n_init=N_INIT)``) y
el criterio de parada de scikit-learn. El agente se queda con el arranque de menor
inercia final, que es exactamente el modelo guardado.

Las proyecciones a 2D (:class:`Proyeccion2D`) son solo para dibujar: el agrupamiento se
hace en las 100 dimensiones del SVD, y en 2D los grupos se ven más mezclados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from codigo.config import MAX_ITER_LLOYD, MUESTRA_SILUETA, N_INIT, SEMILLA, TOL_LLOYD
from codigo.evaluacion import indice_rand_ajustado

NOMBRE_PCA_PUNTOS = "PCA de todos los puntos"
NOMBRE_PCA_CENTROIDES = "PCA de los centroides finales (ponderado por tamaño)"


@dataclass(frozen=True)
class EstadoLloyd:
    """Estado tras una iteración: centroides y asignación de cada reseña a ellos."""

    iteracion: int          # 0 = centroides iniciales (k-means++), sin movimiento previo
    centros: np.ndarray     # (k, d)
    etiquetas: np.ndarray   # (n,) grupo más cercano a ``centros``
    inercia: float          # suma de distancias² de cada punto a su centroide
    cambios: int            # reseñas que cambiaron de grupo respecto del estado anterior (0 en la iteración 0)


@dataclass(frozen=True)
class Arranque:
    """Una inicialización completa de K-means: su trayectoria y su resultado."""

    indice: int
    trayectoria: list[EstadoLloyd]

    @property
    def final(self) -> EstadoLloyd:
        return self.trayectoria[-1]

    @property
    def iteraciones(self) -> int:
        return self.final.iteracion

    @property
    def inercia(self) -> float:
        return self.final.inercia


def _asignar(X: np.ndarray, centros: np.ndarray) -> tuple[np.ndarray, float]:
    """Grupo más cercano de cada punto y la inercia resultante."""
    d2 = (X ** 2).sum(axis=1)[:, None] - 2.0 * X @ centros.T + (centros ** 2).sum(axis=1)[None, :]
    etiquetas = d2.argmin(axis=1)
    inercia = float(np.maximum(d2[np.arange(len(X)), etiquetas], 0.0).sum())
    return etiquetas, inercia


def _actualizar_centros(X: np.ndarray, etiquetas: np.ndarray, centros: np.ndarray) -> np.ndarray:
    """Cada centroide pasa a ser la media de sus reseñas (si un grupo queda vacío, no se mueve)."""
    nuevos = centros.copy()
    for g in range(len(centros)):
        miembros = etiquetas == g
        if miembros.any():
            nuevos[g] = X[miembros].mean(axis=0)
    return nuevos


def _lloyd_desde(X: np.ndarray, centros: np.ndarray, max_iter: int, tol: float) -> Iterator[EstadoLloyd]:
    """Itera Lloyd desde unos centroides iniciales.

    Termina cuando ninguna reseña cambia de grupo, cuando los centroides se mueven menos
    que ``tol`` (relativo a la varianza media, como scikit-learn) o al llegar a ``max_iter``.
    """
    umbral_movimiento = tol * float(X.var(axis=0).mean())
    etiquetas, inercia = _asignar(X, centros)
    yield EstadoLloyd(0, centros, etiquetas, inercia, 0)
    for iteracion in range(1, max_iter + 1):
        nuevos = _actualizar_centros(X, etiquetas, centros)
        movimiento = float(((nuevos - centros) ** 2).sum())
        nuevas_etiquetas, inercia = _asignar(X, nuevos)
        cambios = int((nuevas_etiquetas != etiquetas).sum())
        centros, etiquetas = nuevos, nuevas_etiquetas
        yield EstadoLloyd(iteracion, centros, etiquetas, inercia, cambios)
        if cambios == 0 or movimiento <= umbral_movimiento:
            return


def iterar_lloyd(
    X: np.ndarray,
    k: int,
    semilla: int = SEMILLA,
    max_iter: int = MAX_ITER_LLOYD,
    tol: float = TOL_LLOYD,
) -> Iterator[EstadoLloyd]:
    """Estados de UN arranque k-means++ con la semilla dada, hasta converger."""
    X = np.asarray(X, dtype=np.float64)
    centros, _ = kmeans_plusplus(X, n_clusters=k, random_state=semilla)
    yield from _lloyd_desde(X, centros, max_iter, tol)


def trayectoria_lloyd(X: np.ndarray, k: int, semilla: int = SEMILLA, **kwargs) -> list[EstadoLloyd]:
    """Todos los estados de :func:`iterar_lloyd`."""
    return list(iterar_lloyd(X, k, semilla, **kwargs))


def iterar_arranques(
    X: np.ndarray,
    k: int,
    semilla: int = SEMILLA,
    n_init: int = N_INIT,
    max_iter: int = MAX_ITER_LLOYD,
    tol: float = TOL_LLOYD,
) -> Iterator[Arranque]:
    """Los ``n_init`` arranques de ``KMeans(k, random_state=semilla, n_init=n_init)``, de uno en uno.

    Se reproduce la secuencia de inicializaciones de scikit-learn: un único
    ``RandomState(semilla)`` que se consume en orden, un k-means++ por arranque. Depende de
    cómo la versión fijada de scikit-learn deriva las inicializaciones (ver
    ``pruebas/test_lloyd.py``, que falla si cambia). Cada arranque se calcula al pedirlo.
    """
    X = np.asarray(X, dtype=np.float64)
    estado_aleatorio = np.random.RandomState(semilla)
    for indice in range(n_init):
        centros, _ = kmeans_plusplus(X, n_clusters=k, random_state=estado_aleatorio)
        yield Arranque(indice, list(_lloyd_desde(X, centros, max_iter, tol)))


def calcular_arranques(X: np.ndarray, k: int, semilla: int = SEMILLA, n_init: int = N_INIT, **kwargs) -> list[Arranque]:
    """Todos los arranques de :func:`iterar_arranques`."""
    return list(iterar_arranques(X, k, semilla, n_init, **kwargs))


def elegir_ganador(arranques: Sequence[Arranque]) -> Arranque:
    """El arranque de menor inercia final (el primero si hay empate), como hace K-means."""
    return min(arranques, key=lambda a: (a.inercia, a.indice))


def ari_trayectoria_vs_modelo(etiquetas_anim: Sequence[int], kmeans: KMeans) -> float:
    """ARI entre una partición de la animación y la del modelo guardado."""
    return indice_rand_ajustado(etiquetas_anim, kmeans.labels_)


def resumen_arranques(arranques: Sequence[Arranque], kmeans: KMeans) -> list[dict]:
    """Una fila por arranque (para mostrar "probó 10 arranques y se quedó con el de menor inercia")."""
    ganador = elegir_ganador(arranques).indice
    return [
        {
            "arranque": a.indice,
            "iteraciones": a.iteraciones,
            "inercia_final": a.inercia,
            "ari_vs_modelo": ari_trayectoria_vs_modelo(a.final.etiquetas, kmeans),
            "ganador": a.indice == ganador,
        }
        for a in arranques
    ]


# --------------------------------------------------------------------------
# Proyección a 2D (solo para dibujar)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Proyeccion2D:
    nombre: str
    puntos2d: np.ndarray             # (n, 2)
    centros2d: np.ndarray            # (T, k, 2) centroides de cada iteración
    varianza_explicada: float        # fracción de la varianza total de X que conservan los 2 ejes
    silueta_2d: float                # silueta de la partición final medida EN estos 2 ejes
    media: np.ndarray                # (d,) centro de la proyección
    ejes: np.ndarray                 # (2, d) ejes ortonormales

    def proyectar(self, puntos: np.ndarray) -> np.ndarray:
        """Proyecta puntos de 100 dimensiones (``(..., d)``) al mismo plano 2D que ``puntos2d``."""
        return (np.asarray(puntos, dtype=np.float64) - self.media) @ self.ejes.T


def _silueta_2d(puntos2d: np.ndarray, etiquetas: np.ndarray, semilla: int) -> float:
    return float(silhouette_score(
        puntos2d, etiquetas, sample_size=min(MUESTRA_SILUETA, len(puntos2d)), random_state=semilla))


def _construir(nombre: str, X: np.ndarray, media: np.ndarray, ejes: np.ndarray,
               centros: Sequence[np.ndarray], etiquetas_finales: np.ndarray, semilla: int) -> Proyeccion2D:
    """``ejes`` (2, d) ortonormales: proyecta puntos y centroides con la misma transformación."""
    puntos2d = (X - media) @ ejes.T
    pila = np.stack([np.asarray(c, dtype=np.float64) for c in centros])            # (T, k, d)
    centros2d = ((pila.reshape(-1, X.shape[1]) - media) @ ejes.T).reshape(len(pila), -1, 2)
    varianza = float(puntos2d.var(axis=0).sum() / X.var(axis=0).sum())
    return Proyeccion2D(nombre, puntos2d, centros2d, varianza, _silueta_2d(puntos2d, etiquetas_finales, semilla),
                        media, ejes)


def proyectar_pca_puntos(
    X: np.ndarray, centros: Sequence[np.ndarray], etiquetas_finales: np.ndarray, semilla: int = SEMILLA,
) -> Proyeccion2D:
    """PCA ajustado UNA vez sobre todos los puntos; los centroides de cada iteración se proyectan."""
    X = np.asarray(X, dtype=np.float64)
    pca = PCA(n_components=2, svd_solver="full", random_state=semilla).fit(X)
    return _construir(NOMBRE_PCA_PUNTOS, X, pca.mean_, pca.components_, centros, etiquetas_finales, semilla)


def proyectar_pca_centroides(
    X: np.ndarray, centros: Sequence[np.ndarray], etiquetas_finales: np.ndarray, semilla: int = SEMILLA,
) -> Proyeccion2D:
    """PCA de los centroides finales ponderado por el tamaño de cada grupo.

    Los dos ejes son las direcciones en las que los grupos (contados con su tamaño) más se
    separan entre sí, así que es el plano donde mejor se distinguen. Se ajusta UNA vez.
    """
    X = np.asarray(X, dtype=np.float64)
    finales = np.asarray(centros[-1], dtype=np.float64)                              # (k, d)
    tamanos = np.bincount(etiquetas_finales, minlength=len(finales)).astype(np.float64)
    pesos = tamanos / tamanos.sum()
    media = pesos @ finales                                                          # = media de X
    centrados = finales - media
    covarianza = (centrados * pesos[:, None]).T @ centrados                          # (d, d), rango <= k-1
    valores, vectores = np.linalg.eigh(covarianza)
    ejes = vectores[:, np.argsort(valores)[::-1][:2]].T                              # (2, d)
    return _construir(NOMBRE_PCA_CENTROIDES, X, media, ejes, centros, etiquetas_finales, semilla)


def elegir_proyeccion(candidatas: Sequence[Proyeccion2D]) -> Proyeccion2D:
    """La que mejor separa los grupos medidos en 2D (mayor silueta)."""
    return max(candidatas, key=lambda p: p.silueta_2d)
