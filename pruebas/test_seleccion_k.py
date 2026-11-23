"""Pruebas de la selección configurable de K. Ejecutar desde la raíz:  python -m pytest -q

Las pruebas de funciones usan datos sintéticos (no necesitan el dataset ni
modelos). Las del agente se omiten si todavía no se entrenó el modelo.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from codigo import agrupamiento, etiquetado  # noqa: E402

HAY_MODELOS = (RAIZ / "modelos" / "modelo_kmeans.joblib").exists()


@pytest.fixture(scope="module")
def datos():
    X, _ = make_blobs(n_samples=300, centers=4, n_features=5, random_state=0)
    return X


# --------------------------------------------------------------------------
# validar_k
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 6, 12])
def test_validar_k_acepta_rango(k):
    assert agrupamiento.validar_k(k) == k


@pytest.mark.parametrize("k", [1, 13, 0, -3, 2.5, "6", True])
def test_validar_k_rechaza_fuera_de_rango_o_no_entero(k):
    with pytest.raises(ValueError):
        agrupamiento.validar_k(k)


def test_validar_k_rechaza_k_mayor_o_igual_que_muestras():
    with pytest.raises(ValueError):
        agrupamiento.validar_k(5, n_muestras=5)


# --------------------------------------------------------------------------
# barrido_k / seleccionar_k
# --------------------------------------------------------------------------
def test_barrido_devuelve_tabla_y_modelos(datos):
    avances = []
    tabla, modelos = agrupamiento.barrido_k(datos, range(2, 6), progreso=lambda i, n, k: avances.append((i, n, k)))
    assert tabla["K"].tolist() == [2, 3, 4, 5]
    assert {"inercia", "silueta", "grupo_menor", "grupo_mayor", "proporcion_menor"} <= set(tabla.columns)
    assert sorted(modelos) == [2, 3, 4, 5]
    assert all(modelos[k].n_clusters == k for k in modelos)
    assert avances == [(1, 4, 2), (2, 4, 3), (3, 4, 4), (4, 4, 5)]


def test_barrido_es_reproducible_y_coincide_con_entrenar_kmeans(datos):
    t1, m1 = agrupamiento.barrido_k(datos, [3, 4])
    t2, _ = agrupamiento.barrido_k(datos, [3, 4])
    pd.testing.assert_frame_equal(t1, t2)
    directo = agrupamiento.entrenar_kmeans(datos, k=4)
    np.testing.assert_array_equal(m1[4].labels_, directo.labels_)


def test_seleccionar_k_conserva_su_interfaz(datos):
    tabla = agrupamiento.seleccionar_k(datos, range(2, 4))
    assert isinstance(tabla, pd.DataFrame) and tabla["K"].tolist() == [2, 3]


# --------------------------------------------------------------------------
# recomendar_k
# --------------------------------------------------------------------------
def _tabla(siluetas, proporciones):
    ks = range(2, 2 + len(siluetas))
    return pd.DataFrame({"K": list(ks), "silueta": siluetas, "proporcion_menor": proporciones})


def test_recomendar_k_aplica_filtros_y_parsimonia():
    # K=2 se descarta por k_min; K=6 por grupo demasiado chico.
    tabla = _tabla([0.50, 0.30, 0.32, 0.31, 0.10], [0.2, 0.1, 0.1, 0.1, 0.01])
    rec = agrupamiento.recomendar_k(tabla, k_min=3, proporcion_minima=0.03, tolerancia=0.10)
    assert rec.candidatos == [3, 4, 5]
    assert set(rec.descartados) == {2, 6}
    assert rec.k_mejor_silueta == 4
    assert rec.k == 3                     # 0.30 >= 0.32 - 0.1 * 0.32
    assert rec.umbral_silueta == pytest.approx(0.288)


def test_recomendar_k_sin_tolerancia_elige_la_mejor_silueta():
    tabla = _tabla([0.50, 0.30, 0.32, 0.31], [0.2, 0.1, 0.1, 0.1])
    assert agrupamiento.recomendar_k(tabla, tolerancia=0.0).k == 4


def test_recomendar_k_sin_candidatos_devuelve_none():
    tabla = _tabla([0.5, 0.4], [0.001, 0.001])
    rec = agrupamiento.recomendar_k(tabla)
    assert rec.k is None and rec.candidatos == []


# --------------------------------------------------------------------------
# Nombres manuales solo para el modelo de referencia
# --------------------------------------------------------------------------
# Partición artificial de 6 grupos y dos variantes con LOS MISMOS tamaños por
# grupo (3, 2, 2, 1, 1, 1) pero distinta asignación de reseñas.
PARTICION = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 5])
MISMOS_TAMANOS_FILAS_CAMBIADAS = np.array([0, 0, 1, 0, 1, 2, 2, 3, 4, 5])   # dos reseñas intercambian grupo
MISMOS_TAMANOS_IDS_CAMBIADOS = np.array([0, 0, 0, 2, 2, 1, 1, 3, 4, 5])     # grupos 1 y 2 renombrados


def test_particiones_artificiales_tienen_los_mismos_tamanos():
    for otra in (MISMOS_TAMANOS_FILAS_CAMBIADAS, MISMOS_TAMANOS_IDS_CAMBIADOS):
        assert np.bincount(otra).tolist() == np.bincount(PARTICION).tolist()


def test_huella_es_determinista_e_independiente_del_tipo():
    h = etiquetado.huella_particion(PARTICION)
    assert h == etiquetado.huella_particion(PARTICION.tolist())
    assert h == etiquetado.huella_particion(PARTICION.astype(np.int32))
    assert len(h) == 64


def test_huella_distingue_particiones_con_los_mismos_tamanos():
    h = etiquetado.huella_particion(PARTICION)
    assert etiquetado.huella_particion(MISMOS_TAMANOS_FILAS_CAMBIADAS) != h
    assert etiquetado.huella_particion(MISMOS_TAMANOS_IDS_CAMBIADOS) != h
    assert etiquetado.huella_particion(PARTICION[:-1]) != h                  # otro corpus


def test_referencia_solo_con_huella_exacta():
    ref = etiquetado.huella_particion(PARTICION)
    assert etiquetado.es_modelo_referencia(PARTICION, ref)
    assert etiquetado.nombres_manuales_validos(PARTICION, huella_referencia=ref) == etiquetado.NOMBRES_CLUSTER
    for otra in (MISMOS_TAMANOS_FILAS_CAMBIADAS, MISMOS_TAMANOS_IDS_CAMBIADOS):
        assert not etiquetado.es_modelo_referencia(otra, ref)
        assert etiquetado.nombres_manuales_validos(otra, huella_referencia=ref) == {}


def test_sin_huella_de_referencia_registrada_nunca_se_aplican_nombres_manuales():
    assert not etiquetado.es_modelo_referencia(PARTICION, None)
    assert etiquetado.nombres_manuales_validos(PARTICION, huella_referencia=None) == {}


# --------------------------------------------------------------------------
# nombres_grupos.json + huella
# --------------------------------------------------------------------------
NOMBRES_PROPIOS = {g: f"Nombre propio {g}" for g in range(6)}


def test_guardar_nombres_conserva_el_formato_y_agrega_huella(tmp_path):
    etiquetado.guardar_nombres(NOMBRES_PROPIOS, PARTICION, tmp_path)
    crudo = json.loads((tmp_path / etiquetado.ARCHIVO_NOMBRES).read_text(encoding="utf-8"))
    assert crudo == {str(g): n for g, n in NOMBRES_PROPIOS.items()}         # mismo formato de antes
    meta = json.loads((tmp_path / etiquetado.ARCHIVO_HUELLA_NOMBRES).read_text(encoding="utf-8"))
    assert meta == {"huella_particion": etiquetado.huella_particion(PARTICION), "k": 6, "n_resenas": 10}


def test_nombres_guardados_se_aplican_a_su_misma_particion(tmp_path):
    etiquetado.guardar_nombres(NOMBRES_PROPIOS, PARTICION, tmp_path)
    persistidos = etiquetado.cargar_nombres(tmp_path)
    nombres, origen, aviso = etiquetado.elegir_nombres_manuales(PARTICION, persistidos, huella_referencia=None)
    assert (nombres, origen, aviso) == (NOMBRES_PROPIOS, "archivo", None)


def test_nombres_guardados_no_pasan_a_otra_particion_con_los_mismos_tamanos(tmp_path):
    etiquetado.guardar_nombres(NOMBRES_PROPIOS, PARTICION, tmp_path)
    persistidos = etiquetado.cargar_nombres(tmp_path)
    for otra in (MISMOS_TAMANOS_FILAS_CAMBIADAS, MISMOS_TAMANOS_IDS_CAMBIADOS):
        nombres, origen, aviso = etiquetado.elegir_nombres_manuales(otra, persistidos, huella_referencia=None)
        assert nombres == {} and origen == "terminos" and "otra partición" in aviso


def test_nombres_guardados_tienen_prioridad_sobre_los_del_informe(tmp_path):
    etiquetado.guardar_nombres(NOMBRES_PROPIOS, PARTICION, tmp_path)
    persistidos = etiquetado.cargar_nombres(tmp_path)
    ref = etiquetado.huella_particion(PARTICION)
    nombres, origen, _ = etiquetado.elegir_nombres_manuales(PARTICION, persistidos, huella_referencia=ref)
    assert origen == "archivo" and nombres == NOMBRES_PROPIOS


def test_archivo_ajeno_no_impide_usar_la_referencia(tmp_path):
    etiquetado.guardar_nombres(NOMBRES_PROPIOS, MISMOS_TAMANOS_FILAS_CAMBIADAS, tmp_path)
    persistidos = etiquetado.cargar_nombres(tmp_path)
    ref = etiquetado.huella_particion(PARTICION)
    nombres, origen, aviso = etiquetado.elegir_nombres_manuales(PARTICION, persistidos, huella_referencia=ref)
    assert origen == "referencia" and nombres == etiquetado.NOMBRES_CLUSTER and aviso


def test_archivo_del_formato_anterior_sin_huella_se_ignora(tmp_path):
    (tmp_path / etiquetado.ARCHIVO_NOMBRES).write_text(json.dumps({"0": "a", "1": "b"}), encoding="utf-8")
    persistidos = etiquetado.cargar_nombres(tmp_path)
    assert persistidos == etiquetado.NombresPersistidos({0: "a", 1: "b"}, None)
    nombres, origen, aviso = etiquetado.elegir_nombres_manuales(PARTICION, persistidos, huella_referencia=None)
    assert nombres == {} and origen == "terminos" and etiquetado.ARCHIVO_HUELLA_NOMBRES in aviso


def test_archivo_con_ids_fuera_de_rango_se_ignora(tmp_path):
    etiquetado.guardar_nombres({**NOMBRES_PROPIOS, 9: "sobra"}, PARTICION, tmp_path)
    persistidos = etiquetado.cargar_nombres(tmp_path)
    nombres, origen, aviso = etiquetado.elegir_nombres_manuales(PARTICION, persistidos, huella_referencia=None)
    assert nombres == {} and origen == "terminos" and "fuera de" in aviso


def test_cargar_nombres_sin_archivo_o_invalido(tmp_path):
    assert etiquetado.cargar_nombres(tmp_path) is None
    (tmp_path / etiquetado.ARCHIVO_NOMBRES).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        etiquetado.cargar_nombres(tmp_path)


def test_etiquetas_automaticas_usan_terminos():
    nombres = etiquetado.etiquetar_clusters({0: ["small", "size", "tight", "short"]}, {})
    assert nombres == {0: "Grupo 0: small, size, tight"}


# --------------------------------------------------------------------------
# Agente con otro K (requiere modelos entrenados)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HAY_MODELOS, reason="Primero entrena: python entrenar_agente.py")
class TestAgenteConOtroK:
    @pytest.fixture(scope="class")
    def base(self):
        from codigo.triaje import AgenteTriaje
        return AgenteTriaje()

    def test_otro_k_asigna_con_el_modelo_elegido(self, base):
        km = agrupamiento.entrenar_kmeans(base.X_svd_corpus, k=8)
        agente = base.con_kmeans(km)
        r = agente.sugerir("It runs very small, I had to return it for a larger size.")
        assert agente.n_grupos == 8 and len(r.distancias) == 8 and 0 <= r.grupo < 8
        assert agente.origen_nombres == "terminos" and not agente.nombres_manuales
        assert agente.aviso_nombres is None
        assert set(agente.nombres) == set(range(8))
        assert len(agente.etiquetas_corpus) == len(base.corpus)

    def test_el_agente_base_no_cambia(self, base):
        base.con_kmeans(agrupamiento.entrenar_kmeans(base.X_svd_corpus, k=3))
        assert base.n_grupos == base.k_referencia
        assert base.kmeans is base.kmeans_referencia

    def test_huella_del_agente_corresponde_a_su_particion(self, base):
        assert base.huella == etiquetado.huella_particion(base.etiquetas_corpus)

    def test_nombres_guardados_corresponden_al_modelo_cargado(self, base):
        if not (RAIZ / "modelos" / etiquetado.ARCHIVO_HUELLA_NOMBRES).exists():
            pytest.skip("Modelo entrenado antes de la huella: vuelve a ejecutar python entrenar_agente.py")
        persistidos = etiquetado.cargar_nombres(RAIZ / "modelos")
        assert persistidos.huella == base.huella
        assert base.origen_nombres in ("archivo", "referencia")
        assert all(base.nombres[g] == n for g, n in persistidos.nombres.items())

    def test_nombres_persistidos_solo_para_su_particion(self, base):
        con_nombres = copy.copy(base)
        con_nombres.nombres_persistidos = etiquetado.NombresPersistidos(
            {g: f"Nombre propio {g}" for g in range(base.n_grupos)}, base.huella)
        propio = con_nombres.con_kmeans(base.kmeans_referencia)
        assert propio.origen_nombres == "archivo" and propio.nombres[0] == "Nombre propio 0"
        otro = con_nombres.con_kmeans(agrupamiento.entrenar_kmeans(base.X_svd_corpus, k=base.k_referencia, semilla=7))
        if otro.huella == base.huella:
            pytest.skip("La semilla 7 reprodujo exactamente la misma partición")
        assert otro.origen_nombres != "archivo"

    def test_modelo_de_referencia_usa_nombres_del_informe(self, base):
        if etiquetado.HUELLA_REFERENCIA is None:
            pytest.skip("HUELLA_REFERENCIA aún no registrada en codigo/etiquetado.py")
        if base.huella != etiquetado.HUELLA_REFERENCIA:
            pytest.skip("El modelo guardado no es el de referencia")
        assert base.nombres_manuales
        if base.origen_nombres == "referencia":
            assert base.nombres[4] == etiquetado.NOMBRES_CLUSTER[4]
