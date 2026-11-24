"""Lloyd paso a paso: convergencia, inercia, determinismo, 10 arranques, proyecciones 2D y ARI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.datasets import make_blobs

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from codigo import config  # noqa: E402
from codigo.lloyd import (  # noqa: E402
    ari_trayectoria_vs_modelo,
    calcular_arranques,
    elegir_ganador,
    elegir_proyeccion,
    proyectar_pca_centroides,
    proyectar_pca_puntos,
    resumen_arranques,
    trayectoria_lloyd,
)

HAY_MODELOS = (RAIZ / "modelos" / "modelo_kmeans.joblib").exists()


@pytest.fixture(scope="module")
def blobs():
    return make_blobs(n_samples=400, centers=4, n_features=6, cluster_std=2.5, random_state=3)


@pytest.fixture(scope="module")
def trayectoria(blobs):
    return trayectoria_lloyd(blobs[0], k=4, semilla=7)


# --------------------------------------------------------------------------
# Un arranque
# --------------------------------------------------------------------------
def test_converge_antes_del_maximo(trayectoria):
    assert trayectoria[0].iteracion == 0
    assert [e.iteracion for e in trayectoria] == list(range(len(trayectoria)))
    assert len(trayectoria) - 1 < config.MAX_ITER_LLOYD
    assert trayectoria[-1].cambios == 0   # blobs separados: converge sin reasignaciones pendientes


def test_la_inercia_no_aumenta(trayectoria):
    inercias = [e.inercia for e in trayectoria]
    assert all(b <= a + 1e-9 for a, b in zip(inercias, inercias[1:]))
    assert inercias[-1] < inercias[0]


def test_misma_semilla_misma_salida(blobs):
    a = trayectoria_lloyd(blobs[0], k=4, semilla=11)
    b = trayectoria_lloyd(blobs[0], k=4, semilla=11)
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        np.testing.assert_array_equal(ea.etiquetas, eb.etiquetas)
        np.testing.assert_array_equal(ea.centros, eb.centros)
        assert ea.inercia == eb.inercia and ea.cambios == eb.cambios


def test_cambios_cuenta_las_resenas_que_cambian_de_grupo(trayectoria):
    assert trayectoria[0].cambios == 0
    for anterior, actual in zip(trayectoria, trayectoria[1:]):
        assert actual.cambios == int((actual.etiquetas != anterior.etiquetas).sum())
    assert sum(e.cambios for e in trayectoria) > 0


def test_cada_estado_es_coherente_consigo_mismo(blobs, trayectoria):
    X = blobs[0]
    for e in trayectoria:
        d2 = ((X[:, None, :] - e.centros[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_array_equal(e.etiquetas, d2.argmin(axis=1))
        assert e.inercia == pytest.approx(d2.min(axis=1).sum())


def test_coincide_con_kmeans_de_sklearn_desde_la_misma_inicializacion(blobs, trayectoria):
    X = blobs[0]
    centros0, _ = kmeans_plusplus(X, n_clusters=4, random_state=7)
    np.testing.assert_allclose(trayectoria[0].centros, centros0)
    km = KMeans(n_clusters=4, init=centros0, n_init=1, algorithm="lloyd", tol=config.TOL_LLOYD).fit(X)
    np.testing.assert_array_equal(trayectoria[-1].etiquetas, km.labels_)
    assert trayectoria[-1].inercia == pytest.approx(km.inertia_, rel=1e-6)


def test_respeta_max_iter(blobs):
    from codigo.lloyd import iterar_lloyd
    estados = list(iterar_lloyd(blobs[0], k=4, semilla=7, max_iter=1, tol=0.0))
    assert [e.iteracion for e in estados] == [0, 1]


# --------------------------------------------------------------------------
# Los N_INIT arranques y el ganador
# --------------------------------------------------------------------------
def test_arranques_reproducen_kmeans_de_sklearn(blobs):
    """GUARDA: depende de cómo la versión fijada de scikit-learn deriva las inicializaciones.

    ``KMeans(random_state=s, n_init=n)`` consume un único ``RandomState(s)`` en orden, un
    k-means++ por arranque, y se queda con el de menor inercia. Si una versión nueva cambia
    esa derivación, este test falla y la animación dejaría de coincidir con el modelo guardado.
    """
    X = blobs[0]
    k, semilla = 4, config.SEMILLA
    arranques = calcular_arranques(X, k, semilla=semilla, n_init=config.N_INIT)
    ganador = elegir_ganador(arranques)
    km = KMeans(n_clusters=k, random_state=semilla, n_init=config.N_INIT).fit(X)
    assert len(arranques) == config.N_INIT
    assert ari_trayectoria_vs_modelo(ganador.final.etiquetas, km) == pytest.approx(1.0)
    np.testing.assert_allclose(ganador.final.centros, km.cluster_centers_, rtol=1e-6, atol=1e-8)
    assert ganador.inercia == pytest.approx(km.inertia_, rel=1e-6)


def test_el_ganador_es_el_de_menor_inercia_final(blobs):
    arranques = calcular_arranques(blobs[0], 4, n_init=5)
    assert elegir_ganador(arranques).inercia == min(a.inercia for a in arranques)
    assert [a.indice for a in arranques] == list(range(5))


def test_resumen_marca_un_solo_ganador(blobs):
    X = blobs[0]
    km = KMeans(n_clusters=4, random_state=config.SEMILLA, n_init=3).fit(X)
    filas = resumen_arranques(calcular_arranques(X, 4, n_init=3), km)
    assert [f["arranque"] for f in filas] == [0, 1, 2]
    assert sum(f["ganador"] for f in filas) == 1
    assert all(f["iteraciones"] > 0 and -1.0 <= f["ari_vs_modelo"] <= 1.0 for f in filas)
    assert next(f for f in filas if f["ganador"])["inercia_final"] == min(f["inercia_final"] for f in filas)


# --------------------------------------------------------------------------
# Proyecciones 2D
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def proyecciones(blobs, trayectoria):
    X = blobs[0]
    centros = [e.centros for e in trayectoria]
    etiquetas = trayectoria[-1].etiquetas
    return proyectar_pca_puntos(X, centros, etiquetas), proyectar_pca_centroides(X, centros, etiquetas)


def test_proyecciones_tienen_las_formas_y_metricas_esperadas(blobs, trayectoria, proyecciones):
    for p in proyecciones:
        assert p.puntos2d.shape == (len(blobs[0]), 2)
        assert p.centros2d.shape == (len(trayectoria), 4, 2)
        assert 0.0 < p.varianza_explicada <= 1.0
        assert -1.0 <= p.silueta_2d <= 1.0
    assert proyecciones[0].nombre != proyecciones[1].nombre


def test_pca_de_puntos_maximiza_la_varianza_explicada(proyecciones):
    assert proyecciones[0].varianza_explicada >= proyecciones[1].varianza_explicada - 1e-12


def test_pca_de_puntos_proyecta_la_media_al_origen(blobs, trayectoria):
    X = blobs[0]
    p = proyectar_pca_puntos(X, [X.mean(axis=0, keepdims=True)], trayectoria[-1].etiquetas)
    np.testing.assert_allclose(p.centros2d, 0.0, atol=1e-9)


def test_pca_de_centroides_ejes_ortonormales_y_origen_en_la_media_ponderada(blobs, trayectoria):
    X = blobs[0]
    etiquetas = trayectoria[-1].etiquetas
    p = proyectar_pca_centroides(X, [e.centros for e in trayectoria], etiquetas)
    tamanos = np.bincount(etiquetas, minlength=4)
    final2d = p.centros2d[-1]
    np.testing.assert_allclose((tamanos[:, None] * final2d).sum(axis=0) / tamanos.sum(), 0.0, atol=1e-8)
    # Los puntos proyectados también quedan centrados (la media de los puntos es la media ponderada).
    np.testing.assert_allclose(p.puntos2d.mean(axis=0), 0.0, atol=1e-8)


def test_pca_de_centroides_primer_eje_maximiza_la_separacion_ponderada(blobs, trayectoria):
    X = blobs[0]
    etiquetas = trayectoria[-1].etiquetas
    p = proyectar_pca_centroides(X, [e.centros for e in trayectoria], etiquetas)
    pesos = np.bincount(etiquetas, minlength=4) / len(etiquetas)
    finales = trayectoria[-1].centros
    centrados = finales - pesos @ finales
    separacion_eje1 = float((pesos * p.centros2d[-1][:, 0] ** 2).sum())
    azar = np.random.RandomState(0).normal(size=(50, X.shape[1]))
    azar /= np.linalg.norm(azar, axis=1, keepdims=True)
    for direccion in azar:
        assert separacion_eje1 >= float((pesos * (centrados @ direccion) ** 2).sum()) - 1e-9


def test_elegir_proyeccion_toma_la_de_mayor_silueta(proyecciones):
    elegida = elegir_proyeccion(proyecciones)
    assert elegida.silueta_2d == max(p.silueta_2d for p in proyecciones)


# --------------------------------------------------------------------------
# ARI y modelo real
# --------------------------------------------------------------------------
def test_ari_de_particiones_iguales_es_uno_aunque_cambien_los_ids(blobs):
    km = KMeans(n_clusters=4, n_init=1, random_state=0).fit(blobs[0])
    assert ari_trayectoria_vs_modelo(km.labels_, km) == pytest.approx(1.0)
    assert ari_trayectoria_vs_modelo((km.labels_ + 1) % 4, km) == pytest.approx(1.0)
    assert ari_trayectoria_vs_modelo(np.zeros_like(km.labels_), km) == pytest.approx(0.0)


@pytest.mark.skipif(not HAY_MODELOS, reason="Primero entrena: python entrenar_agente.py")
def test_el_arranque_ganador_reproduce_el_modelo_guardado():
    """GUARDA con los datos reales: la trayectoria animada termina EXACTAMENTE en el modelo del agente.

    Depende de cómo la versión fijada de scikit-learn deriva las inicializaciones por arranque
    (ver ``calcular_arranques``); si cambia, este test falla con claridad.
    """
    from codigo.triaje import AgenteTriaje
    agente = AgenteTriaje()
    arranques = calcular_arranques(agente.X_svd_corpus, agente.k_referencia)
    ganador = elegir_ganador(arranques)
    assert ari_trayectoria_vs_modelo(ganador.final.etiquetas, agente.kmeans) == pytest.approx(1.0)
    np.testing.assert_allclose(ganador.final.centros, agente.kmeans.cluster_centers_, rtol=1e-6, atol=1e-8)
    assert ganador.inercia == pytest.approx(agente.kmeans.inertia_, rel=1e-6)
