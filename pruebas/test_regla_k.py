"""Comportamiento de la regla de elección de K sobre la tabla real del cuaderno 03."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codigo import config  # noqa: E402
from codigo.agrupamiento import recomendar_k  # noqa: E402

# Silueta (SVD) por K de la tabla del cuaderno 03.
SILUETAS_CUADERNO_03 = {
    2: 0.0268, 3: 0.0157, 4: 0.0156, 5: 0.0175, 6: 0.0216, 7: 0.0176,
    8: 0.0184, 9: 0.0196, 10: 0.0192, 11: 0.0249, 12: 0.0030,
}
# Fracción del grupo menor: todos pasan el filtro del 3 % salvo que el caso diga lo contrario.
PROPORCION_SANA = 0.05


def _tabla(siluetas: dict[int, float], proporciones: dict[int, float] | None = None) -> pd.DataFrame:
    proporciones = proporciones or {}
    return pd.DataFrame({
        "K": list(siluetas),
        "silueta": list(siluetas.values()),
        "proporcion_menor": [proporciones.get(k, PROPORCION_SANA) for k in siluetas],
    })


def test_con_15_por_ciento_elige_k6_en_la_tabla_del_cuaderno_03():
    rec = recomendar_k(_tabla(SILUETAS_CUADERNO_03), tolerancia=0.15)
    assert rec.k == 6
    assert rec.k_mejor_silueta == 11
    assert rec.umbral_silueta == pytest.approx(0.0249 * 0.85)
    assert SILUETAS_CUADERNO_03[6] >= rec.umbral_silueta


def test_con_10_por_ciento_elige_k11_si_k11_pasa_el_filtro_del_3_por_ciento():
    rec = recomendar_k(_tabla(SILUETAS_CUADERNO_03), tolerancia=0.10)
    assert rec.k == 11


def test_k11_con_grupo_menor_bajo_el_3_por_ciento_se_descarta():
    tabla = _tabla(SILUETAS_CUADERNO_03, {11: config.PROPORCION_MINIMA_GRUPO / 2})
    rec = recomendar_k(tabla, tolerancia=0.10)
    assert 11 in rec.descartados and rec.k != 11


def test_k_menor_que_el_minimo_nunca_se_elige_aunque_tenga_la_mejor_silueta():
    rec = recomendar_k(_tabla(SILUETAS_CUADERNO_03), tolerancia=0.0)
    assert SILUETAS_CUADERNO_03[2] > max(v for k, v in SILUETAS_CUADERNO_03.items() if k != 2)
    assert 2 in rec.descartados and rec.k == 11


def test_la_tolerancia_por_defecto_es_15_por_ciento():
    assert config.TOLERANCIA_SILUETA == 0.15
    tabla = _tabla(SILUETAS_CUADERNO_03)
    por_defecto = recomendar_k(tabla)
    assert por_defecto == recomendar_k(tabla, tolerancia=0.15)
    assert por_defecto.k == 6
