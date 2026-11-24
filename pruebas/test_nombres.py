"""Nombres de grupo: el automático es el principal y el del informe es secundario por huella."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from codigo import agrupamiento, etiquetado  # noqa: E402
from codigo.presentacion import numero_grupo  # noqa: E402

HAY_MODELOS = (RAIZ / "modelos" / "modelo_kmeans.joblib").exists()

# Términos principales de cada grupo del informe (K = 6); deben aparecer entre los 10 primeros.
TERMINOS_INFORME = {
    0: ["not", "fit", "color", "would", "fabric"],
    1: ["quality", "price", "poor", "poor quality", "worth"],
    2: ["wanted", "wanted love", "really wanted"],
    3: ["flattering", "not flattering"],
    4: ["size", "small", "run", "large", "run small"],
    5: ["look", "like", "look like", "boxy"],
}


@pytest.fixture(scope="module")
def agente():
    if not HAY_MODELOS:
        pytest.skip("Primero entrena: python entrenar_agente.py")
    from codigo.triaje import AgenteTriaje
    return AgenteTriaje()


@pytest.fixture
def es_referencia(agente):
    if agente.k_referencia != 6 or agente.huella != etiquetado.HUELLA_REFERENCIA:
        pytest.skip("El modelo guardado no es la partición de referencia (K = 6)")


def test_los_terminos_de_la_particion_de_referencia_coinciden_con_el_informe(agente, es_referencia):
    for grupo, esperados in TERMINOS_INFORME.items():
        faltan = [t for t in esperados if t not in agente.terminos[grupo]]
        assert not faltan, f"grupo {grupo}: faltan {faltan} en {agente.terminos[grupo]}"


def test_nombre_automatico_principal_y_del_informe_secundario(agente, es_referencia):
    assert agente.origen_nombres in ("referencia", "archivo")
    for g in range(agente.n_grupos):
        assert agente.nombres[g].startswith(f"Grupo {numero_grupo(g)}: ")
        assert agente.nombres_secundarios[g] == etiquetado.NOMBRES_CLUSTER[g]
    r = agente.sugerir("It runs very small, I had to return it for a larger size.")
    assert r.nombre.startswith("Grupo 5: ") and r.nombre_informe == etiquetado.NOMBRES_CLUSTER[4]


def test_otra_particion_solo_tiene_nombre_automatico(agente):
    otro = agente.con_kmeans(agrupamiento.entrenar_kmeans(agente.X_svd_corpus, k=8))
    assert otro.nombres_secundarios == {} and otro.origen_nombres == "terminos"
    r = otro.sugerir("It runs very small, I had to return it for a larger size.")
    assert r.nombre.startswith("Grupo ") and r.nombre_informe is None


def test_la_numeracion_visible_no_cambia_ids_huella_ni_archivos(agente, es_referencia):
    import json
    assert sorted(set(agente.etiquetas_corpus.tolist())) == list(range(6))          # IDs internos desde 0
    assert agente.huella == etiquetado.HUELLA_REFERENCIA
    assert agente.nombres_secundarios.keys() == set(range(6))
    guardado = RAIZ / "modelos" / "nombres_grupos.json"
    if guardado.exists():
        assert set(json.loads(guardado.read_text(encoding="utf-8"))) == {str(g) for g in range(6)}


def test_la_distribucion_expone_ambos_nombres(agente):
    assert {"nombre", "nombre_informe"} <= set(agente.distribucion().columns)
