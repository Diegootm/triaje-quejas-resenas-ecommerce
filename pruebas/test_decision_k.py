"""Decisión de K: persistencia, campos y explicación con las cifras reales."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codigo import config, decision_k  # noqa: E402
from pruebas.test_regla_k import SILUETAS_CUADERNO_03, _tabla  # noqa: E402


@pytest.fixture
def decision():
    return decision_k.decidir_k(_tabla(SILUETAS_CUADERNO_03, {12: 0.01}))


def test_decide_k6_con_los_campos_de_la_regla(decision):
    assert decision.k_elegido == 6 and decision.origen == decision_k.ORIGEN_AGENTE
    assert decision.k_mejor_silueta == 11
    assert decision.silueta_mejor == pytest.approx(0.0249)
    assert decision.umbral == pytest.approx(0.0249 * 0.85)
    assert decision.margen == pytest.approx(0.0216 - 0.0249 * 0.85)
    assert decision.margen >= 0
    assert decision.tolerancia == config.TOLERANCIA_SILUETA and decision.semilla == config.SEMILLA
    assert set(decision.descartados) == {2, 12}
    assert decision.candidatos == list(range(3, 12))
    assert decision.fecha


def test_guardar_y_cargar_conserva_todo(decision, tmp_path):
    ruta = decision_k.guardar_decision(decision, tmp_path)
    assert ruta.name == config.ARCHIVO_DECISION_K
    leida = decision_k.cargar_decision(tmp_path)
    pd.testing.assert_frame_equal(leida.tabla, decision.tabla)
    for campo in ("descartados", "k_elegido", "k_mejor_silueta", "silueta_mejor", "umbral",
                  "margen", "tolerancia", "semilla", "fecha", "origen"):
        assert getattr(leida, campo) == getattr(decision, campo), campo


def test_cargar_sin_archivo_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="python entrenar_agente.py"):
        decision_k.cargar_decision(tmp_path)


def test_explicacion_incluye_las_cifras_reales(decision):
    texto = decision_k.explicar_decision(decision)
    assert "Elegí K=6" in texto
    assert f"{decision.silueta_mejor:.4f}" in texto and f"{decision.umbral:.4f}" in texto
    assert f"{decision.margen:+.4f}" in texto
    assert "K=2 (K menor que 3)" in texto and "K=12" in texto and "15%" in texto


def test_k_manual_queda_registrado_como_manual(decision):
    manual = decision_k.con_k_manual(decision, 8)
    assert manual.k_elegido == 8 and manual.origen == decision_k.ORIGEN_MANUAL
    assert manual.tabla.equals(decision.tabla) and manual.umbral == decision.umbral
    assert "impuesto manualmente" in decision_k.explicar_decision(manual)
    with pytest.raises(ValueError):
        decision_k.con_k_manual(decision, 99)


def test_sin_candidatos_falla_con_los_motivos():
    with pytest.raises(ValueError, match="Ningún K cumple"):
        decision_k.decidir_k(_tabla({2: 0.1, 3: 0.2}, {3: 0.0}))
