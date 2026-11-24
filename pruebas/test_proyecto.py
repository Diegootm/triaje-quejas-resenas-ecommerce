"""Pruebas automáticas del proyecto. Ejecutar desde la raíz:  python -m pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from codigo.preprocesamiento import expandir_contracciones, preprocesar  # noqa: E402

HAY_MODELOS = (RAIZ / "modelos" / "modelo_kmeans.joblib").exists()


def _k_elegido() -> int | None:
    """K que eligió el agente al entrenar (modelos/seleccion_k.json); None si no hay decisión."""
    from codigo.decision_k import cargar_decision
    try:
        return cargar_decision(RAIZ / "modelos").k_elegido
    except FileNotFoundError:
        return None


# Los valores exactos del informe (Tabla 44, tamaños) son de la partición con K = 6.
SOLO_K6 = pytest.mark.skipif(
    not HAY_MODELOS or _k_elegido() != 6,
    reason="Los valores del informe corresponden a K = 6; el agente eligió otro K (o no hay modelos)",
)


def test_contracciones_se_expanden():
    assert expandir_contracciones("i didn't like it, it wasn't my size and won't fit") == \
        "i did not like it, it was not my size and will not fit"


def test_negacion_se_conserva():
    # Tabla 34 del informe
    assert preprocesar("This dress did not fit me at all, the fabric was cheap!") == "dress not fit fabric cheap"


def test_lematizacion_basica():
    assert preprocesar("The dress runs small") == "dress run small"


@pytest.mark.skipif(not HAY_MODELOS, reason="Primero entrena: python entrenar_agente.py")
class TestAgente:
    @pytest.fixture(scope="class")
    def agente(self):
        from codigo.triaje import AgenteTriaje
        return AgenteTriaje()

    @SOLO_K6
    def test_queja_de_talla(self, agente):
        r = agente.sugerir("It runs very small, I had to return it for a larger size.")
        assert r.valido and r.grupo == 4
        assert r.distancias[4] == pytest.approx(0.4520, abs=1e-4)   # Tabla 44

    @SOLO_K6
    def test_queja_de_calidad(self, agente):
        r = agente.sugerir("Poor quality fabric, not worth the price.")
        assert r.valido and r.grupo == 1
        assert r.distancias[1] == pytest.approx(0.6099, abs=1e-4)   # Tabla 44

    def test_el_modelo_guardado_usa_el_k_elegido_por_el_agente(self, agente):
        if _k_elegido() is None:
            pytest.skip("No hay modelos/seleccion_k.json: entrena con python entrenar_agente.py")
        assert agente.k_referencia == _k_elegido()

    def test_texto_sin_vocabulario_no_se_asigna(self, agente):
        r = agente.sugerir("asdf qwerty")
        assert not r.valido and r.grupo is None

    @SOLO_K6
    def test_tamanos_de_grupo_del_informe(self, agente):
        assert agente.distribucion()["quejas"].tolist() == [1873, 435, 361, 300, 720, 986]
